"""自我改进（记忆层）：权重口径、不一致检测修复、规则管理、schema 迁移。全部使用临时库。"""
import asyncio
import sqlite3

import pytest

from Data import memory as M


@pytest.fixture
def mm(tmp_path):
    return M.MemoryManager(db_path=str(tmp_path / "t.db"))


def _run(coro):
    return asyncio.run(coro)


# ---------- P2-1 schema ----------
def test_fresh_db_is_at_latest_schema(mm):
    v = sqlite3.connect(mm.db_path).execute("PRAGMA user_version").fetchone()[0]
    assert v == M.SCHEMA_VERSION


def test_record_analysis_writes_vote_rows(mm):
    rid = _run(mm.record_analysis(
        session_id="s1", stock_symbol="nvda", query="q",
        agent_votes={"A": "BUY", "B": "HOLD"}, final_decision="BUY",
        user_id="u1", agent_details={"A": {"target_position_pct": 0.5, "confidence": 0.8}},
    ))
    rows = sqlite3.connect(mm.db_path).execute(
        "SELECT agent_name, vote, target_pct, confidence FROM agent_votes WHERE analysis_id=? ORDER BY agent_name",
        (rid,),
    ).fetchall()
    assert rows == [("A", "BUY", 0.5, 0.8), ("B", "HOLD", None, None)]


def test_migration_backfills_legacy_json_votes(tmp_path):
    """旧库（user_version=0，只有 analysis_history 的 JSON 投票）应被回填到 agent_votes。"""
    db = str(tmp_path / "legacy.db")
    c = sqlite3.connect(db)
    c.executescript("""
        CREATE TABLE analysis_history (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL DEFAULT 'anonymous',
          session_id TEXT NOT NULL, stock_symbol TEXT NOT NULL, query TEXT, agent_votes TEXT, final_decision TEXT,
          reasoning_summary TEXT, user_feedback TEXT, market_outcome TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        INSERT INTO analysis_history (session_id, stock_symbol, agent_votes, final_decision)
          VALUES ('s', 'X', '{"A":"BUY","B":"SELL"}', 'HOLD');
    """)
    c.commit(); c.close()
    M.MemoryManager(db_path=db)
    c = sqlite3.connect(db)
    assert c.execute("PRAGMA user_version").fetchone()[0] == M.SCHEMA_VERSION
    assert c.execute("SELECT agent_name, vote FROM agent_votes ORDER BY agent_name").fetchall() == [("A", "BUY"), ("B", "SELL")]


# ---------- P0-1 权重口径 ----------
def test_feedback_weight_is_visible_to_voting_for_same_user(mm):
    """旧问题：反馈写入用户行，投票却读 global，反馈从不生效。"""
    _run(mm.adjust_weight("TECHNICAL_NERD", -0.10, "user_A"))
    assert _run(mm.resolve_weights("user_A", ["TECHNICAL_NERD"]))["TECHNICAL_NERD"] == pytest.approx(0.9)
    # 其他用户不受影响
    assert _run(mm.resolve_weights("user_B", ["TECHNICAL_NERD"]))["TECHNICAL_NERD"] == pytest.approx(1.0)


def test_objective_and_personal_weights_multiply_and_clamp(mm):
    _run(mm.adjust_weight("A", -0.10, "u"))           # 个人 0.9
    with sqlite3.connect(mm.db_path) as c:
        c.execute("INSERT INTO agent_stats (agent_name, user_id, current_weight) VALUES ('A','global',1.5)")
    assert _run(mm.resolve_weights("u", ["A"]))["A"] == pytest.approx(1.35)  # 1.5 * 0.9
    with sqlite3.connect(mm.db_path) as c:
        c.execute("UPDATE agent_stats SET current_weight=0.2 WHERE agent_name='A' AND user_id='global'")
    assert _run(mm.resolve_weights("u", ["A"]))["A"] == pytest.approx(M.WEIGHT_MIN)  # 下限截断


def test_resolve_weights_global_user_no_double_count(mm):
    _run(mm.adjust_weight("A", -0.10, "global"))
    assert _run(mm.resolve_weights("global", ["A"]))["A"] == pytest.approx(0.9)


# ---------- P0-2 不一致检测修复 ----------
def test_inconsistency_not_comparing_with_itself(mm):
    _run(mm.record_analysis("s1", "NVDA", "q", {"A": "BUY"}, "BUY", user_id="u"))
    # 本次是反转，且已写入本次记录（rid_new）；应与 rid_old 比较而非自己
    rid_new = _run(mm.record_analysis("s2", "NVDA", "q", {"A": "SELL"}, "SELL", user_id="u"))
    warn = _run(mm.find_inconsistencies("u", "NVDA", "SELL", exclude_id=rid_new))
    assert warn and "方向反转" in warn


def test_inconsistency_none_when_consistent(mm):
    _run(mm.record_analysis("s1", "NVDA", "q", {"A": "BUY"}, "BUY", user_id="u"))
    rid = _run(mm.record_analysis("s2", "NVDA", "q", {"A": "BUY"}, "BUY", user_id="u"))
    assert _run(mm.find_inconsistencies("u", "NVDA", "BUY", exclude_id=rid)) is None


def test_inconsistency_first_ever_analysis_has_no_warning(mm):
    rid = _run(mm.record_analysis("s1", "NVDA", "q", {"A": "SELL"}, "SELL", user_id="u"))
    assert _run(mm.find_inconsistencies("u", "NVDA", "SELL", exclude_id=rid)) is None


# ---------- P2-3 规则管理 ----------
def test_rule_list_disable_delete(mm):
    r = _run(mm.add_or_update_rule("不要过度解读单日放量", "Morefit", "u"))
    rules = _run(mm.list_rules("u"))
    assert rules[0]["rule_text"] == "不要过度解读单日放量" and rules[0]["active"] is True
    assert _run(mm.set_rule_active(r.id, "u", False)) is True
    assert _run(mm.list_rules("u", include_inactive=False)) == []
    # 只能操作自己的规则
    assert _run(mm.delete_rule(r.id, "other")) is False
    assert _run(mm.delete_rule(r.id, "u")) is True
    assert _run(mm.list_rules("u")) == []


# ---------- Agent 名称统一（历史数据的类名/旧写法） ----------
@pytest.mark.parametrize("raw,canon", [
    ("TECHNICAL_NERD", "technical-nerd"), ("technical-nerd", "technical-nerd"),
    ("Morefit", "morefit"), ("RiskManager", "risk-manager"), ("risk-manager", "risk-manager"),
    ("SentimentAnalyzer", "sentiment-analyzer"), ("UNKNOWN_X", "UNKNOWN_X"),
])
def test_canonical_agent_name(raw, canon):
    assert M.canonical_agent_name(raw) == canon


def test_record_analysis_stores_canonical_names(mm):
    rid = _run(mm.record_analysis("s", "NVDA", "q", {"TECHNICAL_NERD": "BUY", "RiskManager": "HOLD"},
                                  "BUY", user_id="u"))
    names = [r[0] for r in sqlite3.connect(mm.db_path).execute(
        "SELECT agent_name FROM agent_votes WHERE analysis_id=? ORDER BY agent_name", (rid,))]
    assert names == ["risk-manager", "technical-nerd"]


def test_weight_lookup_with_legacy_name_hits_canonical_row(mm):
    _run(mm.adjust_weight("technical-nerd", -0.10, "u"))
    assert _run(mm.resolve_weights("u", ["TECHNICAL_NERD"]))["TECHNICAL_NERD"] == pytest.approx(0.9)


def test_migration_v3_canonicalizes_legacy_rows(tmp_path):
    db = str(tmp_path / "legacy_names.db")
    M.MemoryManager(db_path=db)  # 先建到最新版
    c = sqlite3.connect(db)
    c.execute("PRAGMA user_version = 2")  # 模拟停在 v2
    c.execute("INSERT INTO analysis_history (session_id, stock_symbol, agent_votes, final_decision) "
              "VALUES ('s','X','{\"TECHNICAL_NERD\":\"BUY\"}','BUY')")
    rid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
    c.execute("INSERT INTO agent_votes (analysis_id, agent_name, vote) VALUES (?, 'TECHNICAL_NERD', 'BUY')", (rid,))
    c.commit(); c.close()
    M.MemoryManager(db_path=db)  # 触发 v3 迁移
    c = sqlite3.connect(db)
    assert c.execute("PRAGMA user_version").fetchone()[0] == M.SCHEMA_VERSION
    assert c.execute("SELECT agent_name FROM agent_votes WHERE analysis_id=?", (rid,)).fetchall() == [("technical-nerd",)]
    assert '"technical-nerd"' in c.execute("SELECT agent_votes FROM analysis_history WHERE id=?", (rid,)).fetchone()[0]


def test_voting_uses_user_personal_weight(mm):
    """端到端口径：用户对某 Agent 不同意后，该用户投票时此 Agent 的权重下降；其他用户不变。"""
    from LLM.graph import FinGraph

    _run(mm.adjust_weight("morefit", -0.5, "user_A"))  # 个人权重 1.0 -> 0.5
    g = object.__new__(FinGraph)
    g.memory_manager = mm
    results = {
        "morefit": {"vote": "BUY", "target_position_pct": 0.5, "confidence": 0.8, "symbol": "NVDA"},
        "risk-manager": {"vote": "HOLD", "target_position_pct": 0.3, "confidence": 0.6, "symbol": "NVDA"},
    }

    def weights_used(uid):
        out = _run(g._voting({"agent_results": results, "user_id": uid, "thread_id": "t"}))
        return {d["vote"]: d["weight"] for d in out["final_decision"]["decisions"]} if "decisions" in out["final_decision"] \
            else {d["vote"]: d["weight"] for d in out["detailed_analysis"]["decisions"]}

    wa = weights_used("user_A")
    wb = weights_used("user_B")
    assert wa["BUY"] == pytest.approx(0.5)     # morefit 个人降权后用于投票
    assert wb["BUY"] == pytest.approx(1.0)     # 其他用户不受影响
    assert wa["HOLD"] == pytest.approx(1.0)    # 未调整的 Agent 保持 1.0


def test_rule_saved_by_canonical_name_is_injected_for_class_name(mm):
    """反馈规则存为标准名 morefit；注入时 Agent 传的是类名 Morefit，必须能取到，否则规则永远不生效。"""
    _run(mm.add_or_update_rule("不要过度解读单日放量", "morefit", "u"))
    got = _run(mm.get_rules("u", "Morefit"))
    assert [r.rule_text for r in got] == ["不要过度解读单日放量"]
    ctx = _run(mm.build_memory_context("u", None, "Morefit"))
    assert "不要过度解读单日放量" in ctx


def test_disabled_rule_is_not_injected(mm):
    rule = _run(mm.add_or_update_rule("规则A", "morefit", "u"))
    _run(mm.set_rule_active(rule.id, "u", False))
    assert "规则A" not in _run(mm.build_memory_context("u", None, "Morefit"))
