"""
Regression tests for the bugs that were fixed.

These tests verify that:
1. RAG/build_db.py uses `ids` (variable) not `id` (builtin)
2. SentimentAnalyzer passes memory_manager to super().__init__()
3. bare `except:` has been replaced with `except Exception:`
"""

import ast
import inspect
import pytest


class TestBuildDbBugFix:
    """Regression: RAG/build_db.py line 64 was `ids=id` (builtin), should be `ids=ids[...]`."""

    def test_ids_variable_used_not_builtin_id(self):
        """Verify the source file uses `ids` variable, not `id` builtin."""
        import os

        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "RAG",
            "build_db.py",
        )
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()

        # The fix: should NOT contain `ids=id,` (with comma after id)
        # The old bug was `ids=id,` which passes the builtin `id` function
        assert "ids=id," not in source, (
            "Bug regression: `ids=id,` found in build_db.py — "
            "should be `ids=ids[i : i + chunk_size],`"
        )
        # Should contain the fixed version
        assert "ids=ids[i : i + chunk_size]" in source

    def test_no_bare_except_in_build_db(self):
        """Verify no bare `except:` in build_db.py."""
        import os

        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "RAG",
            "build_db.py",
        )
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                pytest.fail(
                    f"Found bare `except:` at line {node.lineno} in build_db.py"
                )


class TestSentimentAnalyzerMemoryFix:
    """Regression: SentimentAnalyzer.__init__() didn't pass memory_manager to super()."""

    def test_super_init_receives_memory_manager_kwarg(self):
        """Verify the source passes memory_manager= to super().__init__()."""
        import os

        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "LLM",
            "agent.py",
        )
        with open(path, "r", encoding="utf-8-sig") as f:
            source = f.read()

        tree = ast.parse(source)

        # Find the SentimentAnalyzer class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "SentimentAnalyzer":
                # Find its __init__
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        # Find super().__init__() call
                        for stmt in ast.walk(item):
                            if isinstance(stmt, ast.Call):
                                func = stmt.func
                                if (
                                    isinstance(func, ast.Attribute)
                                    and func.attr == "__init__"
                                    and isinstance(func.value, ast.Call)
                                    and isinstance(func.value.func, ast.Name)
                                    and func.value.func.id == "super"
                                ):
                                    # Check that memory_manager is in keywords
                                    kwarg_names = [
                                        kw.arg for kw in stmt.keywords
                                    ]
                                    assert "memory_manager" in kwarg_names, (
                                        "SentimentAnalyzer.__init__ does not pass "
                                        "memory_manager to super().__init__()"
                                    )
                                    return
                pytest.fail("Could not find super().__init__() call in SentimentAnalyzer")
        pytest.fail("Could not find SentimentAnalyzer class")

    def test_sentiment_analyzer_accepts_memory_manager(self):
        """Verify SentimentAnalyzer constructor accepts memory_manager param."""
        import os
        import sys

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)

        from LLM.agent import SentimentAnalyzer

        sig = inspect.signature(SentimentAnalyzer.__init__)
        param_names = list(sig.parameters.keys())
        assert "memory_manager" in param_names, (
            "SentimentAnalyzer.__init__ missing memory_manager parameter"
        )


class TestApiServerNoBareExcept:
    """Verify api_server.py has no bare `except:` clauses."""

    def test_no_bare_except(self):
        import os

        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "api_server.py",
        )
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        bare_excepts = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                bare_excepts.append(node.lineno)

        assert bare_excepts == [], (
            f"Found bare `except:` at lines {bare_excepts} in api_server.py"
        )


class TestMainPyUsesRedis:
    """Verify main.py uses Redis, not MySQL."""

    def test_no_mysql_import(self):
        import os

        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main.py",
        )
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()

        assert "PyMySQLSaver" not in source, (
            "main.py still imports PyMySQLSaver — should use AsyncRedisSaver"
        )
        assert "AsyncRedisSaver" in source

    def test_uses_config_for_redis(self):
        """Verify main.py uses config module for Redis URL, not raw os.getenv."""
        import os

        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main.py",
        )
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()

        # Should use config module
        assert "from config import get_settings" in source
        assert "settings.redis_url" in source
        # Should NOT use raw os.getenv for redis
        assert 'os.getenv("REDIS_URL"' not in source
        assert "MYSQL_URL" not in source
