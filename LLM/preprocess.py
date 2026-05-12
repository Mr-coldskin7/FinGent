try:
    from LLM.base import AgentResponse, Agent
except ImportError:
    from base import AgentResponse, Agent
from datetime import datetime, timedelta
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from dotenv import load_dotenv
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Iterator
import os
import uuid
import json
import re
import unittest
from unittest.mock import Mock

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
        year_start = f"{current_date[:4]}-01-01"

        system_prompt = (
            "You are a financial query parser. Convert user questions to structured JSON.\n\n"
            "## Output Schema\n"
            "{\n"
            '  "status": "ready|clarification_needed",\n'
            '  "intent": {\n'
            '    "type": "MARKET_DATA|COMPANY_INFO|REPORT_ANALYSIS|NEWS_SENTIMENT|TECHNICAL_ANALYSIS|RISK_MANAGEMENT|SUGGESTIONS|PORTFOLIO_RISK|SENTIMENT_CHECK"\n'
            "  },\n"
            '  "entities": {\n'
            '    "symbols": ["英伟达"],\n'
            '    "names": ["英伟达"],\n'
            '    "code": ["NVDA"]\n'
            "  },\n"
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
            "5. Return ONLY valid JSON, no markdown, no explanations\n\n"
            "## Examples\n"
            'Input: "达子今天股价多少"\n'
            f'Output: {{"status":"ready","intent":{{"type":"MARKET_DATA"}},"entities":{{"symbols":["NVDA"],"names":["英伟达"]}},"time_range":{{"start":"{current_date}","end":"{current_date}"}},"original_input":"达子今天股价多少","clarification":null}}\n\n'
            'Input: "金龙最近怎么样"\n'
            f'Output: {{"status":"clarification_needed","intent":{{"type":"MARKET_DATA"}},"entities":{{"symbols":[],"names":["金龙"]}},"time_range":{{"start":"{date_30d_ago}","end":"{current_date}"}},"original_input":"金龙最近怎么样","clarification":{{"issue_type":"AMBIGUOUS_NAME","message":"请指定具体股票","options":["金龙汽车(600686)","金龙羽(002882)","金龙鱼(300999)"]}}}}\n\n'
            'Input: "英伟达风险大吗，该买多少"\n'
            f'Output: {{"status":"ready","intent":{{"type":"RISK_MANAGEMENT"}},"entities":{{"symbols":["NVDA"],"names":["英伟达"],"code":["NVDA"]}},"time_range":{{"start":"{date_30d_ago}","end":"{current_date}"}},"original_input":"英伟达风险大吗，该买多少","clarification":null}}\n\n'
            'Input: "最近市场情绪怎么样"\n'
            f'Output: {{"status":"ready","intent":{{"type":"SENTIMENT_CHECK"}},"entities":{{"symbols":[],"names":[]}},"time_range":{{"start":"{date_30d_ago}","end":"{current_date}"}},"original_input":"最近市场情绪怎么样","clarification":null}}\n\n'
            'Input: "茅台舆情如何"\n'
            f'Output: {{"status":"ready","intent":{{"type":"NEWS_SENTIMENT"}},"entities":{{"symbols":["贵州茅台"],"names":["贵州茅台"],"code":["600519"]}},"time_range":{{"start":"{date_30d_ago}","end":"{current_date}"}},"original_input":"茅台舆情如何","clarification":null}}'
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
        except json.JSONDecodeError:
            json_match = re.search(
                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned, re.DOTALL
            )
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Invalid JSON output: {raw_output[:200]}")

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


class TestPreprocessorThread(unittest.TestCase):
    """简短测试：验证 thread_id 是否正确传递"""

    def test_thread_id_passed_to_chat(self):
        """测试 parse 方法是否正确传递 thread_id 到 chat"""
        from unittest.mock import patch

        _mysql_saver_ctx = PyMySQLSaver.from_conn_string(
            os.getenv(
                "MYSQL_URL", "mysql+pymysql://root:password@localhost:3306/fingent"
            )
        )
        checkpointer = _mysql_saver_ctx.__enter__()
        checkpointer.setup()
        mock_model = Mock()
        mock_model.bind_tools = Mock(return_value=mock_model)
        preprocessor = Preprocessor(mock_model, checkpointer)

        # Mock chat 方法
        mock_response = AgentResponse(
            final_answer='{"status":"ready","intent":{"type":"MARKET_DATA"},"entities":{"symbols":["NVDA"],"names":["英伟达"]},"time_range":{"start":"2024-01-01","end":"2024-01-01"},"original_input":"测试","clarification":null}',
            messages=[],
            thread_id="test-thread-123",
            step_count=2,
            tool_calls=0,
        )

        with patch("base.Agent.chat", return_value=mock_response) as mock_chat:
            # 测试1: 不传 thread_id
            preprocessor.parse("测试")
            mock_chat.assert_called_with("测试", None)

            # 测试2: 传入指定 thread_id
            preprocessor.parse("测试", thread_id="my-thread")
            mock_chat.assert_called_with("测试", "my-thread")

            # 测试3: 相同 thread_id 多次调用
            preprocessor.parse("第二次", thread_id="my-thread")
            mock_chat.assert_called_with("第二次", "my-thread")

            print(f"✓ thread_id 传递测试通过，共调用 {mock_chat.call_count} 次")


# ==================== Usage Examples ====================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # python preprocess.py --test
        unittest.main(argv=[""], exit=False, verbosity=2)
    else:
        # 正常运行
        _mysql_saver_ctx = PyMySQLSaver.from_conn_string(
            os.getenv(
                "MYSQL_URL", "mysql+pymysql://root:password@localhost:3306/fingent"
            )
        )
        checkpointer = _mysql_saver_ctx.__enter__()
        checkpointer.setup()
        model = ChatOpenAI(
            api_key=qianwen_api_key,
            base_url=os.getenv(
                "MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            model="qwen-max",
            temperature=0.3,
        )
        preprocessor = Preprocessor(model, checkpointer)
        print(preprocessor.parse("达子今天股价多少"))
