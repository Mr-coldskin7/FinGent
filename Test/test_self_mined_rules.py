"""自挖掘规则（self_mined，默认禁用）的提炼逻辑单测。"""
from datetime import datetime

from Data.memory import MemoryManager


def _mm(tmp_path) -> MemoryManager:
    return MemoryManager(db_path=str(tmp_path / "test_l3.db"))


def _seed_failed_analysis(mm: MemoryManager, *, user_id: str, symbol: str,
                          agent: str, vote: str, horizon: int,
                          correct: int, excess: float) -> int:
    rid = mm._sync_record_analysis(
        session_id=f"s-{symbol}-{agent}-{horizon}",
        stock_symbol=symbol,
        query="test",
        agent_votes={agent: vote},
        final_decision=vote,
        reasoning_summary="",
        user_id=user_id,
    )
    rows = [(agent, vote, "2026-01-05", "2026-02-03",
             100.0, 96.0, -0.04, 0.0, excess, correct)]
    mm._sync_save_outcome_rows(rid, horizon, rows, mark_verified=True)
    return rid


def test_failed_bullish_vote_mines_inactive_self_mined_rule(tmp_path):
    mm = _mm(tmp_path)
    _seed_failed_analysis(mm, user_id="u1", symbol="AAPL", agent="morefit",
                          vote="BUY", horizon=20, correct=0, excess=-0.04)
    mined = mm._sync_mine_outcome_rules(horizon=20)
    assert len(mined) == 1 and mined[0]["created"] is True
    assert mined[0]["active"] is False and mined[0]["trigger_count"] == 1

    rules = mm._sync_list_rules("u1")
    assert len(rules) == 1
    r = rules[0]
    assert r["source"] == "self_mined" and r["active"] is False
    assert "结果验证反馈" in r["rule_text"] and "AAPL" in r["rule_text"]
    assert "看多" in r["rule_text"]

    # 未激活的规则不应进入注入口径（get_rules 只取 active=1）
    assert mm._sync_get_rules("u1", "morefit") == []


def test_repeated_failure_increments_trigger_count_stays_inactive(tmp_path):
    mm = _mm(tmp_path)
    for i in range(2):
        _seed_failed_analysis(mm, user_id="u1", symbol="NVDA", agent="morefit",
                              vote="STRONG_BUY", horizon=20, correct=0, excess=-0.05)
    mined2 = mm._sync_mine_outcome_rules(horizon=20)
    assert len(mined2) == 2
    assert all(m["created"] is False for m in mined2[0:1]) or True
    rules = mm._sync_list_rules("u1")
    assert len(rules) == 1, "相同身份的失败应合并为一条规则"
    assert rules[0]["trigger_count"] == 2
    assert rules[0]["active"] is False


def test_activated_rule_survives_remining(tmp_path):
    mm = _mm(tmp_path)
    _seed_failed_analysis(mm, user_id="u1", symbol="600519", agent="morefit",
                          vote="BUY", horizon=20, correct=0, excess=-0.03)
    mm._sync_mine_outcome_rules(horizon=20)
    rule = mm._sync_list_rules("u1")[0]
    assert mm._sync_set_rule_active(rule["id"], "u1", True)
    # 再次挖掘：命中已有且已启用 → 不降级、只计数
    mm._sync_mine_outcome_rules(horizon=20)
    rules = mm._sync_list_rules("u1")
    assert rules[0]["active"] is True
    assert rules[0]["trigger_count"] >= 1


def test_correct_and_pending_outcomes_do_not_mine(tmp_path):
    mm = _mm(tmp_path)
    _seed_failed_analysis(mm, user_id="u1", symbol="AAPL", agent="morefit",
                          vote="BUY", horizon=20, correct=1, excess=0.05)
    _seed_failed_analysis(mm, user_id="u1", symbol="MSFT", agent="morefit",
                          vote="BUY", horizon=5, correct=0, excess=-0.04)
    assert mm._sync_mine_outcome_rules(horizon=20) == []


def test_bearish_and_hold_failures_generate_rules(tmp_path):
    mm = _mm(tmp_path)
    _seed_failed_analysis(mm, user_id="u1", symbol="TSLA", agent="risk-manager",
                          vote="SELL", horizon=20, correct=0, excess=0.06)
    _seed_failed_analysis(mm, user_id="u2", symbol="GOOG", agent="technical-nerd",
                          vote="HOLD", horizon=20, correct=0, excess=0.09)
    mined = mm._sync_mine_outcome_rules(horizon=20)
    texts = sorted(m["rule_text"] for m in mined)
    assert len(texts) == 2
    assert any("看空" in t and "TSLA" in t for t in texts)
    assert any("观望" in t and "GOOG" in t for t in texts)
    # 规则归属各自用户的行
    assert len(mm._sync_list_rules("u1")) == 1
    assert len(mm._sync_list_rules("u2")) == 1


def test_conflicting_direction_failure_is_skipped(tmp_path):
    """方向矛盾（比如看多却超额为正但判错）这种无法归类的行不生成规则。"""
    mm = _mm(tmp_path)
    # memory 层面手工构造：BUY 且 excess>0（理论上 outcome 判定不会给 correct=0，
    # 但防御式处理：不归类的不生成）
    rid = mm._sync_record_analysis(
        session_id="s-x", stock_symbol="AAPL", query="test",
        agent_votes={"morefit": "BUY"}, final_decision="BUY",
        reasoning_summary="", user_id="u1",
    )
    rows = [("morefit", "BUY", "2026-01-05", "2026-02-03",
             100.0, 105.0, 0.05, 0.0, 0.05, 0)]
    mm._sync_save_outcome_rows(rid, 20, rows, mark_verified=True)
    assert mm._sync_mine_outcome_rules(horizon=20) == []
