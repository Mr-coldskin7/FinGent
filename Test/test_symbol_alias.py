"""海力士 / 000660 映射到美股 SKHY；A 股代码不受影响。"""
import pytest

from Data import outcome as O
from Data.symbol_alias import alias_symbol


@pytest.mark.parametrize("text", ["海力士", "SK海力士", "sk hynix", "SKHYNIX", "000660", "skhy"])
def test_alias_maps_to_skhy(text):
    assert alias_symbol(text) == "SKHY"


@pytest.mark.parametrize("text", ["600519", "NVDA", "茅台", ""])
def test_non_alias_returns_none(text):
    assert alias_symbol(text) is None


def test_outcome_uses_us_listing_and_sp500_benchmark_for_hynix():
    assert O.yahoo_symbol("000660") == "SKHY"
    assert O.yahoo_symbol("海力士") == "SKHY"
    assert O.benchmark_for("000660") == "^GSPC"  # 不能落到沪深300
    assert O.benchmark_for("海力士") == "^GSPC"


def test_a_share_still_a_share():
    assert O.yahoo_symbol("600519") == "600519.SS"
    assert O.benchmark_for("600519") == "000300.SS"


def test_us_stock_lookup_resolves_hynix_aliases():
    from Data.providers import us_stock

    assert us_stock.input2Symbol("海力士") == "SKHY"
    assert us_stock.input2Symbol("000660") == "SKHY"
    assert us_stock.input2Symbol("SK海力士") == "SKHY"


def test_info_service_resolves_hynix_to_us():
    from Data.service import InfoService

    s = object.__new__(InfoService)
    assert s._resolve("海力士") == ("US", "SKHY")
    assert s._resolve("000660") == ("US", "SKHY")
