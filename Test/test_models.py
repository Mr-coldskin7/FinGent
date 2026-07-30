"""
Tests for Data/models.py — UnifiedStockData, to_dict(), to_json().
"""

import json
import pytest
from datetime import datetime
from Data.models import (
    BasicInfo,
    PriceData,
    CompanyDescription,
    FinancialReport,
    NewsData,
    TechnicalIndicators,
    DataSource,
    UnifiedStockData,
    DataType,
)


class TestDataType:
    def test_enum_values(self):
        assert DataType.BASIC_INFO.value == "basic_info"
        assert DataType.PRICE_DATA.value == "price_data"
        assert DataType.COMPANY_DESCRIPTION.value == "company_description"
        assert DataType.FINANCIAL_REPORT.value == "financial_report"
        assert DataType.NEWS_DATA.value == "news_data"
        assert DataType.TECHNICAL_INDICATORS.value == "technical_indicators"

    def test_enum_count(self):
        assert len(DataType) == 6


class TestBasicInfo:
    def test_required_fields(self):
        info = BasicInfo(
            symbol="AAPL",
            name="Apple Inc.",
            currency="USD",
            exchange="NASDAQ",
            exchange_full_name="NASDAQ Global Select",
            market_type="US",
        )
        assert info.symbol == "AAPL"
        assert info.name == "Apple Inc."
        assert info.sector is None

    def test_optional_fields(self):
        info = BasicInfo(
            symbol="AAPL",
            name="Apple Inc.",
            currency="USD",
            exchange="NASDAQ",
            exchange_full_name="NASDAQ Global Select",
            market_type="US",
            sector="Technology",
            market_cap=3000000000000,
        )
        assert info.sector == "Technology"
        assert info.market_cap == 3_000_000_000_000


class TestPriceData:
    def test_creation(self):
        p = PriceData(
            symbol="AAPL",
            date="2025-01-01",
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=1000000,
        )
        assert p.symbol == "AAPL"
        assert p.adjusted_close is None
        assert p.market_type == ""


class TestUnifiedStockData:
    def _make_data(self, num_prices=7):
        """Helper: build a UnifiedStockData with N price records."""
        prices = [
            PriceData(
                symbol="AAPL",
                date=f"2025-01-{i:02d}",
                open=150.0 + i,
                high=155.0 + i,
                low=149.0 + i,
                close=153.0 + i,
                volume=1000000 + i * 10000,
            )
            for i in range(1, num_prices + 1)
        ]
        return UnifiedStockData(
            symbol="AAPL",
            market_type="US",
            price_data=prices,
            basic_info=BasicInfo(
                symbol="AAPL",
                name="Apple Inc.",
                currency="USD",
                exchange="NASDAQ",
                exchange_full_name="NASDAQ Global Select",
                market_type="US",
            ),
            company_description=CompanyDescription(
                symbol="AAPL",
                description="A" * 600,  # 600 chars, should be truncated to 500
            ),
        )

    def test_to_dict_truncates_prices_to_5(self):
        data = self._make_data(num_prices=7)
        d = data.to_dict()
        assert len(d["recent_prices"]) == 5
        # Should be the LAST 5
        assert d["recent_prices"][0]["date"] == "2025-01-03"
        assert d["recent_prices"][-1]["date"] == "2025-01-07"

    def test_to_dict_keeps_all_when_leq_5(self):
        data = self._make_data(num_prices=3)
        d = data.to_dict()
        assert len(d["recent_prices"]) == 3

    def test_to_dict_truncates_description(self):
        data = self._make_data()
        d = data.to_dict()
        assert len(d["description"]) == 500

    def test_to_dict_data_summary(self):
        data = self._make_data()
        d = data.to_dict()
        summary = d["data_summary"]
        assert summary["has_basic_info"] is True
        assert summary["price_records"] == 7
        assert summary["has_description"] is True
        assert summary["financial_reports"] == 0
        assert summary["news_count"] == 0

    def test_to_dict_no_optional(self):
        data = UnifiedStockData(symbol="TEST", market_type="US")
        d = data.to_dict()
        assert "basic_info" not in d
        assert "recent_prices" not in d
        assert "description" not in d
        assert d["data_summary"]["has_basic_info"] is False

    def test_to_json(self):
        data = self._make_data(num_prices=1)
        j = data.to_json()
        parsed = json.loads(j)
        assert parsed["symbol"] == "AAPL"

    def test_to_dict_basic_info_fields(self):
        data = self._make_data()
        d = data.to_dict()
        bi = d["basic_info"]
        assert bi["symbol"] == "AAPL"
        assert bi["name"] == "Apple Inc."
        assert bi["exchange"] == "NASDAQ"
        assert bi["currency"] == "USD"
