"""
Tests for LLM/registry.py — Agent 注册表（skills 自动发现 / 工具解析 / 非法名校验）。
"""

import pytest

from LLM.registry import RegistryError, all_agents, register_agent, resolve
from LLM.registry import AgentInfo


class TestDiscoverSkills:
    def test_builtin_agents_registered(self):
        ags = all_agents()
        for name in [
            "technical-nerd",
            "morefit",
            "risk-manager",
            "sentiment-analyzer",
        ]:
            assert name in ags, f"内置 agent 应自动注册: {name}"

    def test_diy_skill_auto_registered(self):
        """丢一个 skills 目录 = 自动注册（tiktok-auntie 是演示用 DIY agent）。"""
        ags = all_agents()
        assert "tiktok-auntie" in ags

    def test_tools_resolved_from_allowed_tools(self):
        info = all_agents()["tiktok-auntie"]
        names = {t.name for t in info.tools}
        assert names == {"get_stock_price", "get_stock_company_info", "bocha_search"}


class TestResolve:
    def test_resolve_valid(self):
        infos = resolve(["technical-nerd", "risk-manager"])
        assert [i.name for i in infos] == ["technical-nerd", "risk-manager"]

    def test_resolve_none_returns_all(self):
        assert len(resolve(None)) == len(all_agents())

    def test_resolve_unknown_raises(self):
        with pytest.raises(RegistryError):
            resolve(["not-an-agent"])


class TestRegisterAgent:
    def test_register_and_instantiate_shape(self):
        info = AgentInfo(
            name="test-diy",
            description="测试用 DIY agent",
            skill_name="test-diy",
            allowed_tools=["get_stock_price"],
        )
        register_agent(info)
        assert all_agents()["test-diy"].tools[0].name == "get_stock_price"

    def test_duplicate_register_raises(self):
        dup = AgentInfo(name="technical-nerd", description="重复")
        with pytest.raises(RegistryError):
            register_agent(dup)