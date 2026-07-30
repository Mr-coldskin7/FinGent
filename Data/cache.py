"""
Cache Manager - 数据缓存管理
两级缓存策略：
  L1: 内存缓存（同一次分析内使用，TTL 默认 5 分钟）
  L2: Redis 缓存（跨进程/跨会话，TTL 默认 10 分钟）
"""

import time
import json
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Optional, Dict, Tuple

logger = logging.getLogger(__name__)

# 尝试导入 Redis（仅同步客户端）
try:
    from redis import StrictRedis as _StrictRedis
    from redis import ConnectionError as _RedisConnectionError

    _HAS_REDIS = True
except ImportError:
    _StrictRedis = None
    _RedisConnectionError = Exception
    _HAS_REDIS = False


@dataclass
class CacheEntry:
    data: Any
    timestamp: float
    ttl: int  # 秒
    key_hash: str


class CacheManager:
    """
    两级缓存管理器：
    L1: 内存（dict），快速但进程内
    L2: Redis（可选），跨进程共享
    """

    def __init__(
        self,
        default_ttl: int = 300,
        redis_url: Optional[str] = None,
        redis_ttl: int = 600,
    ):
        self._memory: Dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl
        self._redis_ttl = redis_ttl
        self._redis: Optional[Any] = None

        # 尝试连接 Redis L2（StrictRedis 同步客户端）
        if redis_url and _HAS_REDIS:
            try:
                self._redis = _StrictRedis.from_url(
                    redis_url, decode_responses=True, socket_connect_timeout=2
                )
                self._redis.ping()
                logger.info(f"CacheManager L2 Redis connected: {redis_url}")
            except Exception as e:
                logger.warning(f"CacheManager Redis 连接失败，仅使用 L1: {e}")
                self._redis = None

    def _make_key(self, symbol: str, data_types: Tuple[str, ...]) -> str:
        """生成缓存键"""
        key_data = f"{symbol}:{','.join(sorted(data_types))}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]

    def _safe_call(self, method, *args, **kwargs):
        """安全调用 Redis 方法，处理可能返回协程的情况"""
        try:
            result = method(*args, **kwargs)
            # 如果返回了协程（不应发生），丢弃并返回 None
            import asyncio
            if asyncio.iscoroutine(result) or asyncio.iscoroutinefunction(method):
                logger.warning("Redis 方法返回了协程，已丢弃")
                result.close() if hasattr(result, 'close') else None
                return None
            return result
        except _RedisConnectionError:
            logger.warning("Redis 连接断开，降级到 L1 缓存")
            self._redis = None
            return None
        except Exception as e:
            logger.debug(f"Redis 调用失败: {e}")
            return None

    def get(self, symbol: str, data_types: list) -> Optional[Any]:
        """获取缓存（L1 → L2）"""
        key = self._make_key(symbol, tuple(dt.value for dt in data_types))

        # L1 查找
        entry = self._memory.get(key)
        if entry:
            if time.time() - entry.timestamp <= entry.ttl:
                return entry.data
            del self._memory[key]  # 过期删除

        # L2 查找（同步 fallback）
        if self._redis:
            raw = self._safe_call(self._redis.get, f"fingent_cache:{key}")
            if raw:
                try:
                    data = json.loads(raw)
                    self._memory[key] = CacheEntry(
                        data=data,
                        timestamp=time.time(),
                        ttl=self._default_ttl,
                        key_hash=key,
                    )
                    return data
                except (json.JSONDecodeError, TypeError):
                    pass

        return None

    def set(
        self, symbol: str, data_types: list, data: Any, ttl: Optional[int] = None
    ):
        """设置缓存（L1 + L2）"""
        key = self._make_key(symbol, tuple(dt.value for dt in data_types))
        effective_ttl = ttl or self._default_ttl

        # L1 写入
        self._memory[key] = CacheEntry(
            data=data, timestamp=time.time(), ttl=effective_ttl, key_hash=key
        )

        # L2 写入（可序列化的数据才写 Redis）
        if self._redis:
            try:
                serialized = json.dumps(data, ensure_ascii=False, default=str)
                self._safe_call(
                    self._redis.setex,
                    f"fingent_cache:{key}", self._redis_ttl, serialized
                )
            except (TypeError, ValueError):
                pass  # 不可序列化，跳过 L2

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
            "redis_connected": self._redis is not None,
        }
