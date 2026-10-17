try:
    from LLM.base import Agent
except ImportError:
    from base import Agent
from datetime import datetime, timedelta
from dotenv import load_dotenv

from typing import Optional
import os
import json
import re

try:
    from LLM.tools import fuzzy_search_us_symbols
except ImportError:
    from tools import fuzzy_search_us_symbols

load_dotenv()
qianwen_api_key = os.getenv("QIANWEN_API_KEY")
datetime_format = "%Y-%m-%d"
current_date = datetime.now().strftime(datetime_format)

json_format = {
    "status": "ready | clarification_needed",
    "intent": {
        "type": "MARKET_DATA | COMPANY_INFO | REPORT_ANALYSIS | NEWS_SENTIMENT | TECHNICAL_ANALYSIS | RISK_MANAGEMENT | SUGGESTIONS"
    },
    "entities": {"symbols": ["AAPL"], "names": ["苹果公司"], "code": ["320193"]},
    "time_range": {"start": "2024-01-01", "end": "2024-12-31"},
    "original_input": "<用户原始输入>",
    "clarification": "None or {issue_type: str, message: str, options: List[str]}",
}


class Preprocessor(Agent):
    def __init__(self, model, checkpointer=None):
        date_30d_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # 注册表 agent 列表 → 注入提示词，供语义选择参考（LLM/registry.py）
        try:
            from LLM.registry import all_agents as _registry_all_agents

            _agent_catalog = list(_registry_all_agents().items())
        except Exception:
            _agent_catalog = []
        agents_section = "\n".join(
            f'- "{name}": {info.description}' for name, info in _agent_catalog
        ) or "- （无可用 agent）"

        system_prompt = (
            "You are a financial query parser. Convert user questions to structured JSON.\n\n"
            "## Output Schema\n"
            "{\n"
            '  "status": "ready|clarification_needed",\n'
            '  "intent": {\n'
            '    "type": "MARKET_DATA|COMPANY_INFO|REPORT_ANALYSIS|NEWS_SENTIMENT|TECHNICAL_ANALYSIS|RISK_MANAGEMENT|SUGGESTIONS|PORTFOLIO_RISK|SENTIMENT_CHECK|CHAT"\n'
            "  },\n"
            '  "entities": {\n'
            '    "symbols": ["英伟达"],\n'
            '    "names": ["英伟达"],\n'
            '    "code": ["NVDA"]\n'
            "  },\n"
            '  "agent_selection": ["technical-nerd"],\n'
            '  "agent_selection_reason": "一句话说明为何选这些 agent",\n'
            '  "time_range": {\n'
            '    "start": "YYYY-MM-DD",\n'
            '    "end": "YYYY-MM-DD"\n'
            "  },\n"
            '  "original_input": "string",\n'
            '  "clarification": null | {\n'
            '    "issue_type": "AMBIGUOUS_NAME|MISSING_SYMBOL",\n'
            '    "message": "string",\n'
            '    "options": ["option1", "option2"]\n'
            "  }\n"
            "}\n\n"
            "## Rules\n"
            f'1. Today is {current_date}. Infer time: "最近"->30d, "今年"->YTD, "历史"->1y\n'
            '2. Map slang: "达子"->NVDA, "茅王"->600519, "水果公司"->AAPL\n'
            "3. Entity Fields (CRITICAL):\n"
            '   - "symbols": MUST be company NAME (e.g., "寒武纪", "英伟达", "茅台") - NOT code!\n'
            '   - "names": Company name in Chinese or English\n'
            '   - "code": Stock ticker/code (e.g., "688256", "NVDA", "600519")\n'
            '4. status="clarification_needed" when ambiguous (e.g., "金龙" matches multiple)\n'
            '5. "agent_selection": pick 1-2 most relevant agents from the list below based on the user intent. Empty [] if not applicable. MUST use exact names from the list. Always also output "agent_selection_reason": one short Chinese sentence (<=40 chars) explaining the choice; use "" when agent_selection is [].\n'
            '6. intent="CHAT" for greetings/thanks/capability questions/general conversation NOT about analyzing a specific stock or the market. entities empty, clarification null.\n'
            "7. Return ONLY valid JSON, no markdown, no explanations\n\n"
            "## Available Agents (semantic selection)\n"
            + agents_section
            + "\n\n"
            "## Examples\n"
            'Input: "达子今天股价多少"\n'
            f'Output: {{"status":"ready","intent":{{"type":"MARKET_DATA"}},"entities":{{"symbols":["NVDA"],"names":["英伟达"]}},"time_range":{{"start":"{current_date}","end":"{current_date}"}},"original_input":"达子今天股价多少","clarification":null}}\n\n'
            'Input: "金龙最近怎么样"\n'
            f'Output: {{"status":"clarification_needed","intent":{{"type":"MARKET_DATA"}},"entities":{{"symbols":[],"names":["金龙"]}},"time_range":{{"start":"{date_30d_ago}","end":"{current_date}"}},"original_input":"金龙最近怎么样","clarification":{{"issue_type":"AMBIGUOUS_NAME","message":"请指定具体股票","options":["金龙汽车(600686)","金龙羽(002882)","金龙鱼(300999)"]}}}}\n\n'
            'Input: "英伟达风险大吗，该买多少"\n'
            f'Output: {{"status":"ready","intent":{{"type":"RISK_MANAGEMENT"}},"entities":{{"symbols":["NVDA"],"names":["英伟达"],"code":["NVDA"]}},"agent_selection":["risk-manager"],"agent_selection_reason":"问题聚焦风险评估与仓位，交给风险管理 Agent","time_range":{{"start":"{date_30d_ago}","end":"{current_date}"}},"original_input":"英伟达风险大吗，该买多少","clarification":null}}\n\n'
            'Input: "最近市场情绪怎么样"\n'
            f'Output: {{"status":"ready","intent":{{"type":"SENTIMENT_CHECK"}},"entities":{{"symbols":[],"names":[]}},"time_range":{{"start":"{date_30d_ago}","end":"{current_date}"}},"original_input":"最近市场情绪怎么样","clarification":null}}\n\n'
            'Input: "茅台舆情如何"\n'
            f'Output: {{"status":"ready","intent":{{"type":"NEWS_SENTIMENT"}},"entities":{{"symbols":["贵州茅台"],"names":["贵州茅台"],"code":["600519"]}},"agent_selection":["sentiment-analyzer"],"agent_selection_reason":"问题问舆情，交给舆情分析 Agent","time_range":{{"start":"{date_30d_ago}","end":"{current_date}"}},"original_input":"茅台舆情如何","clarification":null}}\n\n'
            'Input: "你好"\n'
            f'Output: {{"status":"ready","intent":{{"type":"CHAT"}},"entities":{{"symbols":[],"names":[]}},"agent_selection":[],"agent_selection_reason":"","time_range":{{"start":"","end":""}},"original_input":"你好","clarification":null}}\n\n'
            'Input: "你能做什么"\n'
            f'Output: {{"status":"ready","intent":{{"type":"CHAT"}},"entities":{{"symbols":[],"names":[]}},"agent_selection":[],"agent_selection_reason":"","time_range":{{"start":"","end":""}},"original_input":"你能做什么","clarification":null}}'
        )
        tools = [fuzzy_search_us_symbols]
        super().__init__(model, tools, system_prompt, checkpointer)

    def _extract_json(self, raw_output: str) -> dict:
        """从 LLM 输出中提取 JSON"""
        cleaned = raw_output.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as err:
            json_match = re.search(
                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned, re.DOTALL
            )
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Invalid JSON output: {raw_output[:200]}") from err

    def _parse_backtest_format(self, user_input: str) -> Optional[dict]:
        """
        直接解析回测格式输入 (symbol=XXX\ndate=YYY...)
        避免走LLM，解决checkpointer上下文限制问题
        """
        lines = user_input.strip().split("\n")
        data = {}

        for line in lines:
            if "=" in line:
                key, value = line.split("=", 1)
                data[key.strip()] = value.strip()

        # 检查是否是回测格式
        if "symbol" in data and "date" in data and "close_price" in data:
            symbol = data["symbol"]
            date = data["date"]

            return {
                "status": "ready",
                "intent": {"type": "MARKET_DATA"},
                "entities": {"symbols": [symbol], "names": [symbol], "code": [symbol]},
                "time_range": {"start": date, "end": date},
                "original_input": user_input,
                "clarification": None,
            }
        return None

    def parse(self, user_input: str, thread_id: Optional[str] = None) -> dict:
        """
        解析用户输入，返回结构化 dict（同步版本）
        """
        # 1. 先尝试直接解析回测格式
        backtest_result = self._parse_backtest_format(user_input)
        if backtest_result:
            return backtest_result

        # 2. 普通对话走LLM解析
        response = self.chat(user_input, thread_id)
        result = self._extract_json(response.final_answer)
        if "original_input" not in result:
            result["original_input"] = user_input
        return result

    async def async_parse(
        self, user_input: str, thread_id: Optional[str] = None
    ) -> dict:
        """
        解析用户输入，返回结构化 dict（异步版本，用于 async graph 节点）
        """
        # 1. 先尝试直接解析回测格式
        backtest_result = self._parse_backtest_format(user_input)
        if backtest_result:
            return backtest_result

        # 2. 普通对话走LLM解析（异步）
        response = await self.achat(user_input, thread_id)
        result = self._extract_json(response.final_answer)
        if "original_input" not in result:
            result["original_input"] = user_input
        return result


if __name__ == "__main__":
    print("本模块无独立入口：单元测试见 Test/test_preprocess.py，交互请使用 main.py")
