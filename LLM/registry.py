"""
Agent 注册表 — 系统里 agent 的唯一来源。

设计意图（替代 graph.py 里硬编码的 AGENT_CONFIG / 路由表）：
- **agent 不再是图的拓扑**，而是可枚举的注册实体；
- **DIY agent = 丢一个 skill 目录**（LLM/skills/<name>/SKILL.md）即自动注册；
  身份/分析协议/JSON 输出格式都在 SKILL.md 里，可用 allowed-tools 声明工具
  （不声明 = 使用全部现有金融工具）；
- 一次运行选哪几个 agent：显式（API/CLI/用户语言）> 语义（preprocess LLM
  依据各 AgentInfo.description 输出 agent_selection）> 旧 Router 兜底。

用法：
    from LLM.registry import discover_skills, register_agent, resolve, instantiate
    discover_skills()                      # 启动时扫描 skills 目录
    agents = resolve(["technical-nerd"])   # 校验名字
    inst = instantiate("technical-nerd", model, checkpointer)   # 装配实例
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

SKILLS_DIR = Path(__file__).parent / "skills"

# 语义路由可用的 agent 描述（注册后填充）
_REGISTRY: Dict[str, "AgentInfo"] = {}


class RegistryError(ValueError):
    """注册表异常：未知 agent / 重复注册 / 技能文件无效"""


# ───────────────────────── 工具目录 ─────────────────────────
def _tool_catalog() -> Dict[str, Any]:
    """allowed-tools 名字 → 真实工具对象（langchain BaseTool）。集中维护，新增工具在这里挂。"""
    from LLM.unified_stock_tools import (
        bocha_search,
        get_financial_indicator_explanation,
        get_full_stock_analysis,
        get_stock_basic_info,
        get_stock_company_info,
        get_stock_financial_report_links,
        get_stock_financial_statements,
        get_stock_price,
        search_financial_knowledge,
    )
    from LLM.risk_tools import RISK_TOOLS
    from LLM.sentiment_tools import SENTIMENT_TOOLS

    tools: Dict[str, Any] = {
        t.name: t
        for t in [
            get_stock_price,
            get_stock_basic_info,
            get_stock_company_info,
            get_stock_financial_report_links,
            get_stock_financial_statements,
            get_full_stock_analysis,
            search_financial_knowledge,
            get_financial_indicator_explanation,
            bocha_search,
        ]
    }
    for t in RISK_TOOLS + SENTIMENT_TOOLS:
        tools[t.name] = t
    return tools


TOOL_CATALOG = _tool_catalog()

@dataclass
class AgentInfo:
    """一个 agent 的注册元数据。tools 在 instantiate 时装配。"""

    name: str
    description: str
    skill_name: Optional[str] = None
    allowed_tools: List[str] = field(default_factory=list)
    output_parser: str = "json"

    @property
    def tools(self) -> List[Any]:
        """解析 allowed_tools；空列表 = 使用全部现有金融工具（DIY 便利）。"""
        if not self.allowed_tools:
            return list(TOOL_CATALOG.values())
        resolved = [TOOL_CATALOG[n] for n in self.allowed_tools if n in TOOL_CATALOG]
        missing = [n for n in self.allowed_tools if n not in TOOL_CATALOG]
        if missing:
            raise RegistryError(f"agent {self.name} 声明了未知工具: {missing}")
        return resolved


# ───────────────────────── 注册 / 发现 ─────────────────────────
def register_agent(info: AgentInfo) -> None:
    """编程式注册一个 agent（不落盘的 DIY agent 用这个）。"""
    if info.name in _REGISTRY:
        raise RegistryError(f"agent 已存在: {info.name}")
    _REGISTRY[info.name] = info


def discover_skills() -> Dict[str, AgentInfo]:
    """扫描 LLM/skills/*/SKILL.md，解析 frontmatter 自动注册。"""
    discoveries: Dict[str, AgentInfo] = {}
    if not SKILLS_DIR.exists():
        return discoveries
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        md = skill_dir / "SKILL.md"
        if not md.is_file():
            continue
        text = md.read_text(encoding="utf-8")
        try:
            front = yaml.safe_load(text.split("---", 2)[1]) or {}
        except (IndexError, yaml.YAMLError) as e:
            print(f"⚠️ 跳过非法技能 {skill_dir.name}: {e}")
            continue
        name = front.get("name") or skill_dir.name
        desc = front.get("description") or f"{name} 自定义分析 agent"
        allowed = str(front.get("allowed-tools", "") or "").split()
        _REGISTRY[name] = AgentInfo(
            name=name,
            description=desc,
            skill_name=skill_dir.name,
            allowed_tools=allowed,
        )
        discoveries[name] = _REGISTRY[name]
    return discoveries


# ───────────────────────── 查询 / 装配 ─────────────────────────
def all_agents() -> Dict[str, AgentInfo]:
    return dict(_REGISTRY)


def resolve(names: Optional[List[str]]) -> List[AgentInfo]:
    """按名字取 AgentInfo；非法名抛 RegistryError。None → 返回全部。"""
    if names is None:
        return list(_REGISTRY.values())
    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        raise RegistryError(f"未知 agent: {unknown}（可选: {list(_REGISTRY)}）")
    return [_REGISTRY[n] for n in names]


def instantiate(
    name: str,
    model,
    checkpointer=None,
    memory_manager=None,
):
    """按注册表装配 agent 实例（惰性导入 agent 类，避免循环依赖）。"""
    info = _REGISTRY.get(name)
    if info is None:
        raise RegistryError(f"未知 agent: {name}")

    from LLM.base import Agent
    from LLM.agent import Morefit, RiskManager, SentimentAnalyzer, TECHNICAL_NERD

    # 内置 agent 用各自子类（保留 format_prompt 行为）；自定义 skill 用通用 Agent
    BUILTIN = {
        "technical-nerd": TECHNICAL_NERD,
        "morefit": Morefit,
        "risk-manager": RiskManager,
        "sentiment-analyzer": SentimentAnalyzer,
    }
    cls = BUILTIN.get(info.skill_name or "", Agent)

    if cls is Agent:
        # 通用 DIY agent：身份来自 SKILL.md，工具 = allowed_tools（缺省全部）
        return Agent(
            model=model,
            tools=info.tools,
            system_prompt="",
            checkpointer=checkpointer,
            memory_manager=memory_manager,
            skill_name=info.skill_name,
        )

    return cls(
        model=model,
        tools=info.tools,
        checkpointer=checkpointer,
        memory_manager=memory_manager,
    )


# 启动即扫描；显式 register_agent 可在之后叠加
discover_skills()