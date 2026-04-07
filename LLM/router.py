try:
    from LLM.preprocess import Preprocessor
except ImportError:
    from preprocess import Preprocessor
import json
import re

json_format = {
    "status": "ready | clarification_needed",
    "intent": {
        "type": "MARKET_DATA | COMPANY_INFO | REPORT_ANALYSIS | NEWS_SENTIMENT | TECHNICAL_ANALYSIS | RISK_MANAGEMENT | SUGGESTIONS"
    },
    "entities": {
        "symbols": ["AAPL"],
        "names": ["苹果公司"],
        "code":["320193"]
    },
    "time_range": {
        "start": "2024-01-01",
        "end": "2024-12-31"
    },
    "original_input": "<用户原始输入>",
    "clarification": "None or {issue_type: str, message: str, options: List[str]}"
}

class Router:
    """纯路由决策 - 只返回目标节点名称"""
    
    # intent → agent 映射表
    ROUTES = {
        "MARKET_DATA": "ALL",                  # 市场数据 -> 双Agent投票
        "COMPANY_INFO": "Morefit",             # 公司信息 -> 基本面分析
        "REPORT_ANALYSIS": "Morefit",          # 财报分析 -> 基本面分析
        "NEWS_SENTIMENT": "ALL",               # 新闻情绪 -> 双Agent投票
        "TECHNICAL_ANALYSIS": "TECHNICAL_NERD", # 技术分析 -> 技术面分析Agent
        "RISK_MANAGEMENT": "ALL",              # 风险管理 -> 双Agent投票
        "SUGGESTIONS": "ALL"                   # 投资建议 -> 双Agent投票
    }
    
    def route(self, preprocess_result: dict) -> str:
        """
        返回下一个节点名称（LangGraph 使用）
        
        Returns:
            "clarify_node" - 需要澄清
            "market_agent" / "company_agent" ... - 具体 agent
            "unknown" - 未知 intent
        """
        if preprocess_result["status"] == "clarification_needed":
            return "clarify_node"
        
        intent = preprocess_result["intent"]["type"]
        return self.ROUTES.get(intent, "unknown")


if __name__ == "__main__":
    # 初始化
    from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
    from dotenv import load_dotenv
    import os
    load_dotenv()
    qianwen_api_key = os.getenv("QIANWEN_API_KEY")
    _mysql_saver_ctx = PyMySQLSaver.from_conn_string(os.getenv("MYSQL_URL", "mysql+pymysql://root:password@localhost:3306/fingent"))
    checkpointer = _mysql_saver_ctx.__enter__()
    checkpointer.setup()
    from langchain_community.chat_models import ChatTongyi
    model = ChatTongyi(api_key=qianwen_api_key, temperature=0.3)
    
    preprocessor = Preprocessor(model, checkpointer)
    reponse = preprocessor.parse("达子今天股价多少")
    router = Router()
    next_node = router.route(reponse)
