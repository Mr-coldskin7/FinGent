"""
Tests for Trade/adapter.py — GraphSignal, clamp_pct, map_vote_to_target_pct, etc.
"""

import json
import pytest
from Trade.adapter import (
    GraphSignal,
    build_graph_input,
    clamp_pct,
    map_vote_to_target_pct,
    extract_graph_signal,
)


class TestClampPct:
    def test_normal_value(self):
        assert clamp_pct(0.5) == 0.5

    def test_below_zero(self):
        assert clamp_pct(-0.1) == 0.0

    def test_above_one(self):
        assert clamp_pct(1.5) == 1.0

    def test_exact_zero(self):
        assert clamp_pct(0.0) == 0.0

    def test_exact_one(self):
        assert clamp_pct(1.0) == 1.0

    def test_int_input(self):
        assert clamp_pct(2) == 1.0

    def test_string_input(self):
        assert clamp_pct("0.7") == 0.7


class TestMapVoteToTargetPct:
    def test_strong_buy(self):
        assert map_vote_to_target_pct("STRONG_BUY", 0.0) == 0.60

    def test_buy(self):
        assert map_vote_to_target_pct("BUY", 0.0) == 0.30

    def test_strong_sell(self):
        assert map_vote_to_target_pct("STRONG_SELL", 0.5) == 0.00

    def test_sell(self):
        assert map_vote_to_target_pct("SELL", 0.5) == 0.10

    def test_hold_returns_current(self):
        assert map_vote_to_target_pct("HOLD", 0.42) == 0.42

    def test_hold_clamped_above(self):
        assert map_vote_to_target_pct("HOLD", 1.5) == 1.0

    def test_hold_clamped_below(self):
        assert map_vote_to_target_pct("HOLD", -0.1) == 0.0

    def test_none_vote_defaults_to_hold(self):
        assert map_vote_to_target_pct(None, 0.3) == 0.3

    def test_case_insensitive(self):
        assert map_vote_to_target_pct("buy", 0.0) == 0.30


class TestBuildGraphInput:
    def test_output_contains_all_fields(self):
        result = build_graph_input(
            symbol="AAPL",
            date_str="2025-01-15",
            close_price=150.5,
            cash=10000.0,
            portfolio_value=10000.0,
            shares=0,
            avg_cost=0.0,
        )
        assert "symbol=AAPL" in result
        assert "date=2025-01-15" in result
        assert "close_price=150.5" in result
        assert "portfolio_context=" in result

    def test_portfolio_context_is_valid_json(self):
        result = build_graph_input("AAPL", "2025-01-15", 150.0, 10000, 10000, 0)
        # Extract JSON after "portfolio_context="
        json_str = result.split("portfolio_context=")[1].strip()
        parsed = json.loads(json_str)
        assert parsed["cash"] == 10000.0
        assert parsed["holdings"][0]["symbol"] == "AAPL"

    def test_format_is_key_value_lines(self):
        result = build_graph_input("TSLA", "2025-06-01", 250.0, 5000, 5000, 10)
        lines = result.strip().split("\n")
        for line in lines:
            assert "=" in line


class TestExtractGraphSignal:
    def _make_result(self, vote, confidence=0.8, target_pct=None, symbol="AAPL"):
        return {
            "final_decision": {
                "final_vote": vote,
                "confidence": confidence,
                "target_position_pct": target_pct,
                "symbol": symbol,
                "details": {
                    "morefit": {"reason": "Fundamentals are strong"},
                    "technical": {"reason": "RSI oversold"},
                },
            }
        }

    def test_basic_extraction(self):
        result = self._make_result("BUY", confidence=0.75, target_pct=0.30)
        sig = extract_graph_signal(result, 0.0, "FALLBACK")
        assert sig.vote == "BUY"
        assert sig.confidence == 0.75
        assert sig.target_position_pct == 0.30
        assert sig.symbol == "AAPL"

    def test_reason_concatenation(self):
        result = self._make_result("HOLD")
        sig = extract_graph_signal(result, 0.0, "X")
        assert "Fundamentals" in sig.reason
        assert "RSI" in sig.reason
        assert " | " in sig.reason

    def test_fallback_when_no_final_decision(self):
        sig = extract_graph_signal({}, 0.5, "FALLBACK")
        assert sig.vote == "HOLD"
        assert sig.symbol == "FALLBACK"
        assert sig.target_position_pct == 0.5  # current_pct via map_vote_to_target_pct

    def test_fallback_when_target_is_none(self):
        result = {"final_decision": {"final_vote": "BUY", "confidence": 0.6}}
        sig = extract_graph_signal(result, 0.0, "FB")
        # BUY → map_vote_to_target_pct → 0.30
        assert sig.target_position_pct == 0.30

    def test_target_pct_clamped(self):
        result = self._make_result("BUY", target_pct=2.0)
        sig = extract_graph_signal(result, 0.0, "X")
        assert sig.target_position_pct == 1.0

    def test_raw_preserved(self):
        result = self._make_result("SELL")
        sig = extract_graph_signal(result, 0.0, "X")
        assert sig.raw == result

    def test_empty_details(self):
        result = {"final_decision": {"final_vote": "HOLD", "confidence": 0.5}}
        sig = extract_graph_signal(result, 0.3, "X")
        assert sig.reason == ""
