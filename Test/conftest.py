"""
Shared test fixtures for FinGent test suite.
"""

import os
import sys
import pytest
import sqlite3
import tempfile
from unittest.mock import MagicMock

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Mock unavailable imports ─────────────────────────────────────────
# 1. langchain.agents.create_agent doesn't exist in the installed version.
if "langchain.agents" in sys.modules:
    _agents_mod = sys.modules["langchain.agents"]
    if not hasattr(_agents_mod, "create_agent"):
        _agents_mod.create_agent = MagicMock()
else:
    import langchain.agents as _agents_mod

    if not hasattr(_agents_mod, "create_agent"):
        _agents_mod.create_agent = MagicMock()
        sys.modules["langchain.agents"] = _agents_mod

# 2. langgraph.checkpoint.redis may not be installed.
if "langgraph.checkpoint.redis" not in sys.modules:
    _mock_redis_mod = MagicMock()
    sys.modules["langgraph.checkpoint.redis"] = _mock_redis_mod
    sys.modules["langgraph.checkpoint.redis.aio"] = _mock_redis_mod


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite database path for MemoryManager tests."""
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def memory_manager(tmp_db):
    """Create a MemoryManager instance with a temporary database."""
    from Data.memory import MemoryManager

    return MemoryManager(db_path=tmp_db)


@pytest.fixture
def cache_manager():
    """Create a fresh CacheManager instance."""
    from Data.cache import CacheManager

    return CacheManager(default_ttl=300)


@pytest.fixture
def router():
    """Create a Router instance."""
    from LLM.router import Router

    return Router()
