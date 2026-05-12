"""
Cache Manager - 数据缓存管理
"""

import time
import hashlib
from dataclasses import dataclass
from typing import Any, Optional, Dict, Tuple


@dataclass
class CacheEntry:
    data: Any
    timestamp: float
    ttl: int  # 秒
    key_hash: str


class CacheManager:
    """
    内存缓存管理器
    两级缓存策略：
    1. L1: 内存缓存（同一次分析内使用）
    2. 可扩展 L2: 磁盘缓存（跨进程/跨会话）
    """

    def __init__(self, default_ttl: int = 300):  # 默认5分钟
        self._memory: Dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl

    def _make_key(self, symbol: str, data_types: Tuple[str, ...]) -> str:
        """生成缓存键"""
        key_data = f"{symbol}:{','.join(sorted(data_types))}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]

    def get(self, symbol: str, data_types: list) -> Optional[Any]:
        """获取缓存"""
        key = self._make_key(symbol, tuple(dt.value for dt in data_types))
        entry = self._memory.get(key)

        if not entry:
            return None

        # 检查过期
        if time.time() - entry.timestamp > entry.ttl:
            del self._memory[key]
            return None

        return entry.data

    def set(self, symbol: str, data_types: list, data: Any, ttl: Optional[int] = None):
        """设置缓存"""
        key = self._make_key(symbol, tuple(dt.value for dt in data_types))
        self._memory[key] = CacheEntry(
            data=data, timestamp=time.time(), ttl=ttl or self._default_ttl, key_hash=key
        )

    def invalidate(self, symbol: str = None):
        """清除缓存"""
        if symbol:
            keys_to_del = [
                k
                for k, v in self._memory.items()
                if hasattr(v.data, "symbol") and v.data.symbol == symbol
            ]
            for k in keys_to_del:
                del self._memory[k]
        else:
            self._memory.clear()

    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        return {
            "total_entries": len(self._memory),
            "memory_size": sum(len(str(v.data)) for v in self._memory.values()),
        }
