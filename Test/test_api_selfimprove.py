"""自我改进接口：反馈归因、复盘、规则管理。使用临时库，不碰真实记忆数据。"""
import asyncio

import pytest
from fastapi.testclient import TestClient

import api_server
from Data.memory import MemoryManager


@pytest.fixture
def client(tmp_path, monkeypatch):
    mm = MemoryManager(db_path=str(tmp_path / "api.db"))
    monkeypatch.setattr(api_server, "memory_manager", mm)
    return TestClient(api_server.app), mm


def _run(coro):
    return asyncio.run(coro)


def test_feedback_multi_agent_attribution_adjusts_only_selected(client):
    c, mm = client
    r = c.post("/api/v1/feedback", json={
        "session_id": "s", "stock_symbol": "NVDA", "feedback": "disagree",
        "agent_names": ["TECHNICAL_NERD", "RISK_MANAGER"], "user_id": "u1",
    }).json()
    assert r["success"] is True and r["action"] == "weight_decreased"
    assert set(r["weights"]) == {"technical-nerd", "risk-manager"}  # 统一为标准名
    assert _run(mm.resolve_weights("u1", ["SENTIMENT"]))["SENTIMENT"] == pytest.approx(1.0)


def test_feedback_without_agents_records_history_only(client):
    c, _ = client
    r = c.post("/api/v1/feedback", json={"session_id": "s", "stock_symbol": "NVDA",
                                         "feedback": "agree", "user_id": "u1"}).json()
    assert r["success"] is True and r["action"] == "feedback_recorded" and r["weights"] == {}


def test_feedback_weight_delta_applied_once_not_twice(client):
    """回归：同一反馈的 delta 曾被 adjust_weight + record_agent_outcome 叠加两次。
    现在只应变化一次：agree +0.05，disagree -0.10（Data.memory WEIGHT_*_DELTA）。"""
    c, mm = client
    r = c.post("/api/v1/feedback", json={
        "session_id": "s", "stock_symbol": "NVDA", "feedback": "agree",
        "agent_names": ["morefit"], "user_id": "u1",
    }).json()
    assert r["success"]
    assert r["weights"]["morefit"] == pytest.approx(1.0 + 0.05)

    r = c.post("/api/v1/feedback", json={
        "session_id": "s2", "stock_symbol": "AAPL", "feedback": "disagree",
        "agent_names": ["morefit"], "user_id": "u2",
    }).json()
    assert r["weights"]["morefit"] == pytest.approx(1.0 - 0.10)


def test_feedback_unknown_type_rejected(client):
    c, _ = client
    assert c.post("/api/v1/feedback", json={"session_id": "s", "stock_symbol": "X",
                                            "feedback": "meh"}).json()["success"] is False


def test_review_returns_votes_dissenters_and_history(client):
    c, mm = client
    _run(mm.record_analysis("sess1", "NVDA", "q", {"A": "BUY", "B": "SELL", "C": "BUY"},
                            "BUY", user_id="u1",
                            agent_details={"A": {"target_position_pct": 0.5, "confidence": 0.7}}))
    r = c.post("/api/v1/review", json={"session_id": "sess1", "stock_symbol": "NVDA", "user_id": "u1"}).json()
    assert r["success"] is True
    rv = r["review"]
    assert rv["current_decision"] == "BUY"
    assert rv["agent_votes"] == {"A": "BUY", "B": "SELL", "C": "BUY"}
    assert rv["dissenting_agents"] == ["B"]
    assert rv["outcomes"] == []  # 尚未验证
    assert rv["recent_history"] and rv["recent_history"][0]["final_decision"] == "BUY"


def test_review_flags_direction_reversal(client):
    c, mm = client
    _run(mm.record_analysis("s1", "NVDA", "q", {"A": "BUY"}, "BUY", user_id="u1"))
    _run(mm.record_analysis("s2", "NVDA", "q", {"A": "SELL"}, "SELL", user_id="u1"))
    rv = c.post("/api/v1/review", json={"session_id": "s2", "stock_symbol": "NVDA", "user_id": "u1"}).json()["review"]
    assert rv["inconsistency_warning"] and "方向反转" in rv["inconsistency_warning"]


def test_review_missing_history_is_clear_error(client):
    c, _ = client
    r = c.post("/api/v1/review", json={"session_id": "none", "stock_symbol": "ZZZ", "user_id": "u1"}).json()
    assert r["success"] is False and "没有找到" in r["error"]


def test_rules_list_disable_delete_via_api(client):
    c, mm = client
    rule = _run(mm.add_or_update_rule("不要过度解读单日放量", "Morefit", "u1"))
    rules = c.get("/api/v1/memory/rules", params={"user_id": "u1"}).json()["rules"]
    assert rules[0]["id"] == rule.id
    assert c.patch(f"/api/v1/memory/rules/{rule.id}", json={"user_id": "u1", "active": False}).json()["success"]
    assert c.get("/api/v1/memory/rules", params={"user_id": "u1", "include_inactive": False}).json()["rules"] == []
    # 越权：其他用户不能删
    assert c.delete(f"/api/v1/memory/rules/{rule.id}", params={"user_id": "other"}).json()["success"] is False
    assert c.delete(f"/api/v1/memory/rules/{rule.id}", params={"user_id": "u1"}).json()["success"] is True
