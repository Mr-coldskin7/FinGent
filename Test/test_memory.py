"""
Tests for Data/memory.py — MemoryManager (SQLite, no external deps).
"""

import asyncio
import pytest
from Data.memory import (
    MemoryManager,
    AnalysisRecord,
    AgentStat,
    UserRule,
    WEIGHT_AGREE_DELTA,
    WEIGHT_DISAGREE_DELTA,
    WEIGHT_MAX,
    WEIGHT_MIN,
    WEIGHT_RULE_PROMOTE_THRESHOLD,
    get_memory_manager,
)


# ── helpers ──────────────────────────────────────────────────────────
def _run(coro):
    """Run an async coroutine in a fresh event loop (for sync test functions)."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Initialization ───────────────────────────────────────────────────
class TestInit:
    def test_creates_db_file(self, tmp_db, memory_manager):
        import os

        assert os.path.exists(tmp_db)

    def test_tables_exist(self, tmp_db, memory_manager):
        import sqlite3

        conn = sqlite3.connect(tmp_db)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "analysis_history" in tables
        assert "agent_stats" in tables
        assert "user_rules" in tables


# ── Analysis CRUD ────────────────────────────────────────────────────
class TestAnalysisCRUD:
    def test_record_and_get(self, memory_manager):
        mid = _run(
            memory_manager.record_analysis(
                session_id="s1",
                stock_symbol="aapl",
                query="分析AAPL",
                agent_votes={"TECHNICAL_NERD": "BUY", "Morefit": "HOLD"},
                final_decision="BUY",
                reasoning_summary="tech bullish",
                user_id="u1",
            )
        )
        assert mid > 0

        record = _run(memory_manager.get_analysis_record("s1", "aapl"))
        assert record is not None
        assert record.stock_symbol == "AAPL"  # uppercased
        assert record.agent_votes["TECHNICAL_NERD"] == "BUY"
        assert record.final_decision == "BUY"

    def test_record_symbol_uppercased(self, memory_manager):
        _run(
            memory_manager.record_analysis(
                session_id="s2",
                stock_symbol="msft",
                query="test",
                agent_votes={},
                final_decision="HOLD",
                user_id="u1",
            )
        )
        record = _run(memory_manager.get_analysis_record("s2", "msft"))
        assert record.stock_symbol == "MSFT"

    def test_get_nonexistent(self, memory_manager):
        record = _run(memory_manager.get_analysis_record("no_such_session"))
        assert record is None

    def test_update_feedback(self, memory_manager):
        mid = _run(
            memory_manager.record_analysis(
                session_id="s3",
                stock_symbol="TSLA",
                query="test",
                agent_votes={},
                final_decision="SELL",
                user_id="u1",
            )
        )
        _run(memory_manager.update_feedback(mid, "agree"))
        record = _run(memory_manager.get_analysis_record("s3", "TSLA"))
        assert record.user_feedback == "agree"

    def test_record_feedback(self, memory_manager):
        _run(
            memory_manager.record_analysis(
                session_id="s4",
                stock_symbol="GOOG",
                query="test",
                agent_votes={},
                final_decision="BUY",
                user_id="u1",
            )
        )
        rid = _run(memory_manager.record_feedback("s4", "GOOG", "disagree"))
        assert rid is not None

        record = _run(memory_manager.get_analysis_record("s4", "GOOG"))
        assert record.user_feedback == "disagree"

    def test_record_feedback_no_match(self, memory_manager):
        rid = _run(memory_manager.record_feedback("none", "NONE", "agree"))
        assert rid is None

    def test_get_recent_analyses(self, memory_manager):
        for i in range(5):
            _run(
                memory_manager.record_analysis(
                    session_id=f"recent_{i}",
                    stock_symbol="AAPL",
                    query=f"q{i}",
                    agent_votes={},
                    final_decision="BUY",
                    user_id="u1",
                )
            )
        records = _run(memory_manager.get_recent_analyses("u1", "AAPL", limit=3))
        assert len(records) == 3
        # All should belong to the correct user/stock
        for r in records:
            assert r.stock_symbol == "AAPL"
            assert r.user_id == "u1"


# ── Agent Weights ────────────────────────────────────────────────────
class TestAgentWeights:
    def test_default_weight(self, memory_manager):
        weights = _run(
            memory_manager.get_agent_weights("global", ["TECHNICAL_NERD"])
        )
        assert weights["TECHNICAL_NERD"] == 1.0

    def test_adjust_weight_increases(self, memory_manager):
        new_w = _run(
            memory_manager.adjust_weight("TECHNICAL_NERD", WEIGHT_AGREE_DELTA)
        )
        assert new_w == pytest.approx(1.0 + WEIGHT_AGREE_DELTA, abs=0.001)

    def test_weight_clamped_at_max(self, memory_manager):
        # Push above max
        for _ in range(100):
            _run(memory_manager.adjust_weight("AGENT_A", 0.1))
        w = _run(memory_manager.get_agent_weights("global", ["AGENT_A"]))
        assert w["AGENT_A"] <= WEIGHT_MAX

    def test_weight_clamped_at_min(self, memory_manager):
        # Push below min
        for _ in range(100):
            _run(memory_manager.adjust_weight("AGENT_B", -0.1))
        w = _run(memory_manager.get_agent_weights("global", ["AGENT_B"]))
        assert w["AGENT_B"] >= WEIGHT_MIN

    def test_record_agent_outcome_agree(self, memory_manager):
        _run(memory_manager.record_agent_outcome("TECH", "agree"))
        w = _run(memory_manager.get_agent_weights("global", ["TECH"]))
        assert w["TECH"] == pytest.approx(1.0 + WEIGHT_AGREE_DELTA, abs=0.001)

    def test_record_agent_outcome_disagree(self, memory_manager):
        _run(memory_manager.record_agent_outcome("TECH", "disagree"))
        w = _run(memory_manager.get_agent_weights("global", ["TECH"]))
        assert w["TECH"] == pytest.approx(1.0 + WEIGHT_DISAGREE_DELTA, abs=0.001)


# ── User Rules ───────────────────────────────────────────────────────
class TestUserRules:
    def test_add_rule(self, memory_manager):
        rule = _run(
            memory_manager.add_or_update_rule(
                rule_text="不要推荐银行股",
                agent_name="Morefit",
                user_id="u1",
            )
        )
        assert rule.id > 0
        assert rule.trigger_count == 1
        assert rule.active is True

    def test_duplicate_rule_increments_count(self, memory_manager):
        _run(
            memory_manager.add_or_update_rule(
                rule_text="不要推荐银行股", agent_name="Morefit", user_id="u1"
            )
        )
        rule2 = _run(
            memory_manager.add_or_update_rule(
                rule_text="不要推荐银行股", agent_name="Morefit", user_id="u1"
            )
        )
        assert rule2.trigger_count == 2

    def test_get_rules(self, memory_manager):
        _run(
            memory_manager.add_or_update_rule(
                rule_text="Rule A", agent_name="Morefit", user_id="u1"
            )
        )
        _run(
            memory_manager.add_or_update_rule(
                rule_text="Rule B", agent_name="TECHNICAL_NERD", user_id="u1"
            )
        )
        # Get only Morefit rules
        rules = _run(memory_manager.get_rules("u1", agent_name="Morefit"))
        assert all(r.agent_name == "Morefit" for r in rules)

    def test_get_rules_min_trigger_count(self, memory_manager):
        _run(
            memory_manager.add_or_update_rule(
                rule_text="Low count rule", agent_name="A", user_id="u1"
            )
        )
        for _ in range(5):
            _run(
                memory_manager.add_or_update_rule(
                    rule_text="High count rule", agent_name="A", user_id="u1"
                )
            )
        rules = _run(memory_manager.get_rules("u1", min_trigger_count=3))
        assert all(r.trigger_count >= 3 for r in rules)


# ── Inconsistency Detection ──────────────────────────────────────────
class TestInconsistency:
    def test_buy_then_sell_warning(self, memory_manager):
        _run(
            memory_manager.record_analysis(
                session_id="inc1",
                stock_symbol="AAPL",
                query="q1",
                agent_votes={},
                final_decision="BUY",
                user_id="u1",
            )
        )
        warning = _run(
            memory_manager.find_inconsistencies("u1", "AAPL", "SELL")
        )
        assert warning is not None
        assert "方向反转" in warning

    def test_sell_then_buy_warning(self, memory_manager):
        _run(
            memory_manager.record_analysis(
                session_id="inc2",
                stock_symbol="AAPL",
                query="q1",
                agent_votes={},
                final_decision="STRONG_SELL",
                user_id="u1",
            )
        )
        warning = _run(
            memory_manager.find_inconsistencies("u1", "AAPL", "STRONG_BUY")
        )
        assert warning is not None

    def test_hold_then_buy_no_warning(self, memory_manager):
        _run(
            memory_manager.record_analysis(
                session_id="inc3",
                stock_symbol="AAPL",
                query="q1",
                agent_votes={},
                final_decision="HOLD",
                user_id="u1",
            )
        )
        warning = _run(
            memory_manager.find_inconsistencies("u1", "AAPL", "BUY")
        )
        assert warning is None

    def test_no_history_no_warning(self, memory_manager):
        warning = _run(
            memory_manager.find_inconsistencies("u1", "NEWSTOCK", "SELL")
        )
        assert warning is None


# ── build_memory_context ─────────────────────────────────────────────
class TestBuildMemoryContext:
    def test_empty_context(self, memory_manager):
        ctx = _run(memory_manager.build_memory_context("new_user"))
        assert ctx == ""

    def test_context_with_history(self, memory_manager):
        _run(
            memory_manager.record_analysis(
                session_id="ctx1",
                stock_symbol="AAPL",
                query="q",
                agent_votes={},
                final_decision="BUY",
                user_id="u1",
            )
        )
        ctx = _run(memory_manager.build_memory_context("u1", "AAPL"))
        assert "AAPL" in ctx
        assert "BUY" in ctx

    def test_context_with_rules(self, memory_manager):
        _run(
            memory_manager.add_or_update_rule(
                rule_text="Always check PE ratio", agent_name="Morefit", user_id="u1"
            )
        )
        ctx = _run(memory_manager.build_memory_context("u1", agent_name="Morefit"))
        assert "PE ratio" in ctx


# ── Singleton ────────────────────────────────────────────────────────
class TestSingleton:
    def test_get_memory_manager_returns_same_instance(self, tmp_path):
        import Data.memory as mod

        # Reset singleton
        mod._memory_manager = None
        # Point to temp dir
        import os

        db = str(tmp_path / "singleton_test.db")
        os.environ["FINGENT_MEMORY_DB"] = db
        try:
            m1 = get_memory_manager()
            m2 = get_memory_manager()
            assert m1 is m2
        finally:
            del os.environ["FINGENT_MEMORY_DB"]
            mod._memory_manager = None
