"""
Tests for LLM/preprocess.py — _extract_json() and _parse_backtest_format().
These are pure logic tests that do NOT require an LLM or checkpointer.
"""

import json
import pytest
from unittest.mock import MagicMock

# We need to import the class, but constructing it requires a model.
# We'll test the static/instance methods by creating a minimal mock.
from LLM.preprocess import Preprocessor


@pytest.fixture
def preprocessor():
    """Create a Preprocessor with a mocked model (no real LLM calls)."""
    mock_model = MagicMock()
    mock_model.bind_tools = MagicMock(return_value=mock_model)
    return Preprocessor(mock_model, checkpointer=None)


# ── _extract_json ────────────────────────────────────────────────────
class TestExtractJson:
    def test_plain_json(self, preprocessor):
        raw = '{"status": "ready", "intent": {"type": "MARKET_DATA"}}'
        result = preprocessor._extract_json(raw)
        assert result["status"] == "ready"
        assert result["intent"]["type"] == "MARKET_DATA"

    def test_json_in_markdown_fence(self, preprocessor):
        raw = '```json\n{"status": "ready"}\n```'
        result = preprocessor._extract_json(raw)
        assert result["status"] == "ready"

    def test_json_in_plain_fence(self, preprocessor):
        raw = '```\n{"status": "ready"}\n```'
        result = preprocessor._extract_json(raw)
        assert result["status"] == "ready"

    def test_json_with_surrounding_text(self, preprocessor):
        raw = 'Here is the result:\n{"status": "ready", "intent": {"type": "BUY"}}\nDone.'
        result = preprocessor._extract_json(raw)
        assert result["status"] == "ready"

    def test_nested_json(self, preprocessor):
        raw = '{"intent": {"type": "RISK"}, "entities": {"symbols": ["AAPL"]}}'
        result = preprocessor._extract_json(raw)
        assert result["entities"]["symbols"] == ["AAPL"]

    def test_invalid_json_raises(self, preprocessor):
        with pytest.raises(ValueError, match="Invalid JSON"):
            preprocessor._extract_json("this is not json at all")

    def test_empty_string_raises(self, preprocessor):
        with pytest.raises(ValueError):
            preprocessor._extract_json("")

    def test_json_with_extra_whitespace(self, preprocessor):
        raw = '   \n  {"status": "ready"}  \n  '
        result = preprocessor._extract_json(raw)
        assert result["status"] == "ready"

    def test_markdown_fence_no_newline(self, preprocessor):
        raw = '```json{"status": "ready"}```'
        result = preprocessor._extract_json(raw)
        assert result["status"] == "ready"


# ── _parse_backtest_format ───────────────────────────────────────────
class TestParseBacktestFormat:
    def test_valid_backtest_input(self, preprocessor):
        user_input = (
            "symbol=AAPL\n"
            "date=2025-01-15\n"
            "close_price=150.5\n"
            "portfolio_context={}"
        )
        result = preprocessor._parse_backtest_format(user_input)
        assert result is not None
        assert result["status"] == "ready"
        assert result["intent"]["type"] == "MARKET_DATA"
        assert result["entities"]["symbols"] == ["AAPL"]
        assert result["time_range"]["start"] == "2025-01-15"
        assert result["time_range"]["end"] == "2025-01-15"

    def test_missing_symbol_returns_none(self, preprocessor):
        user_input = "date=2025-01-15\nclose_price=150.5"
        result = preprocessor._parse_backtest_format(user_input)
        assert result is None

    def test_missing_date_returns_none(self, preprocessor):
        user_input = "symbol=AAPL\nclose_price=150.5"
        result = preprocessor._parse_backtest_format(user_input)
        assert result is None

    def test_missing_close_price_returns_none(self, preprocessor):
        user_input = "symbol=AAPL\ndate=2025-01-15"
        result = preprocessor._parse_backtest_format(user_input)
        assert result is None

    def test_extra_fields_preserved(self, preprocessor):
        user_input = (
            "symbol=TSLA\n"
            "date=2025-06-01\n"
            "close_price=250.0\n"
            "extra_field=hello"
        )
        result = preprocessor._parse_backtest_format(user_input)
        assert result is not None
        assert result["entities"]["symbols"] == ["TSLA"]

    def test_non_backtest_input_returns_none(self, preprocessor):
        result = preprocessor._parse_backtest_format("分析AAPL的技术面")
        assert result is None

    def test_original_input_preserved(self, preprocessor):
        user_input = "symbol=AAPL\ndate=2025-01-15\nclose_price=150"
        result = preprocessor._parse_backtest_format(user_input)
        assert result["original_input"] == user_input
