"""
Tests for Data/cache.py — CacheManager.
"""

import time
import pytest
from unittest.mock import patch
from Data.cache import CacheManager
from Data.models import DataType, UnifiedStockData


class TestCacheKey:
    def test_key_deterministic(self, cache_manager):
        """Same symbol + same types → same key regardless of order."""
        k1 = cache_manager._make_key("AAPL", ("price_data", "basic_info"))
        k2 = cache_manager._make_key("AAPL", ("basic_info", "price_data"))
        assert k1 == k2

    def test_key_different_symbol(self, cache_manager):
        k1 = cache_manager._make_key("AAPL", ("price_data",))
        k2 = cache_manager._make_key("MSFT", ("price_data",))
        assert k1 != k2

    def test_key_different_types(self, cache_manager):
        k1 = cache_manager._make_key("AAPL", ("price_data",))
        k2 = cache_manager._make_key("AAPL", ("basic_info",))
        assert k1 != k2

    def test_key_length(self, cache_manager):
        k = cache_manager._make_key("AAPL", ("price_data",))
        assert len(k) == 16  # MD5 truncated to 16 hex chars


class TestCacheSetGet:
    def test_set_and_get(self, cache_manager):
        data = {"price": 150.0}
        cache_manager.set("AAPL", [DataType.PRICE_DATA], data)
        result = cache_manager.get("AAPL", [DataType.PRICE_DATA])
        assert result == data

    def test_get_miss(self, cache_manager):
        result = cache_manager.get("AAPL", [DataType.PRICE_DATA])
        assert result is None

    def test_get_expired(self, cache_manager):
        """Entry past its TTL should return None."""
        data = {"price": 150.0}
        cache_manager.set("AAPL", [DataType.PRICE_DATA], data, ttl=10)

        # Simulate time passing
        with patch("Data.cache.time") as mock_time:
            mock_time.time.return_value = time.time() + 11
            result = cache_manager.get("AAPL", [DataType.PRICE_DATA])
        assert result is None

    def test_get_within_ttl(self, cache_manager):
        data = {"price": 150.0}
        cache_manager.set("AAPL", [DataType.PRICE_DATA], data, ttl=300)

        with patch("Data.cache.time") as mock_time:
            mock_time.time.return_value = time.time() + 100
            result = cache_manager.get("AAPL", [DataType.PRICE_DATA])
        assert result == data

    def test_custom_ttl(self, cache_manager):
        data = {"price": 150.0}
        cache_manager.set("AAPL", [DataType.PRICE_DATA], data, ttl=1)
        # Should still be available immediately
        assert cache_manager.get("AAPL", [DataType.PRICE_DATA]) == data

    def test_overwrite(self, cache_manager):
        cache_manager.set("AAPL", [DataType.PRICE_DATA], {"v": 1})
        cache_manager.set("AAPL", [DataType.PRICE_DATA], {"v": 2})
        result = cache_manager.get("AAPL", [DataType.PRICE_DATA])
        assert result == {"v": 2}


class TestCacheInvalidate:
    def test_invalidate_all(self, cache_manager):
        cache_manager.set("AAPL", [DataType.PRICE_DATA], {"v": 1})
        cache_manager.set("MSFT", [DataType.PRICE_DATA], {"v": 2})
        cache_manager.invalidate()
        assert cache_manager.get("AAPL", [DataType.PRICE_DATA]) is None
        assert cache_manager.get("MSFT", [DataType.PRICE_DATA]) is None

    def test_invalidate_by_symbol(self, cache_manager):
        """invalidate(symbol=) only removes entries whose data has that symbol."""
        obj_aapl = UnifiedStockData(symbol="AAPL", market_type="US")
        obj_msft = UnifiedStockData(symbol="MSFT", market_type="US")
        cache_manager.set("AAPL", [DataType.PRICE_DATA], obj_aapl)
        cache_manager.set("MSFT", [DataType.PRICE_DATA], obj_msft)

        cache_manager.invalidate(symbol="AAPL")
        assert cache_manager.get("AAPL", [DataType.PRICE_DATA]) is None
        assert cache_manager.get("MSFT", [DataType.PRICE_DATA]) is not None

    def test_invalidate_missing_symbol(self, cache_manager):
        cache_manager.set("AAPL", [DataType.PRICE_DATA], {"v": 1})
        cache_manager.invalidate(symbol="NOPE")
        # Should not delete anything
        assert cache_manager.get("AAPL", [DataType.PRICE_DATA]) is not None


class TestCacheStats:
    def test_empty_stats(self, cache_manager):
        stats = cache_manager.get_stats()
        assert stats["total_entries"] == 0
        assert stats["memory_size"] == 0

    def test_stats_after_set(self, cache_manager):
        cache_manager.set("AAPL", [DataType.PRICE_DATA], {"price": 150})
        cache_manager.set("MSFT", [DataType.PRICE_DATA], {"price": 300})
        stats = cache_manager.get_stats()
        assert stats["total_entries"] == 2
        assert stats["memory_size"] > 0
