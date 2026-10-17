"""没有个股实体时的路由：板块/通用回答，而不是澄清死胡同（不走网络、不调模型）。"""
import asyncio
from types import SimpleNamespace


from Data import market_context
from Eval.reliability import classify_route
from LLM.graph import FinGraph


def _graph(model=None):
    g = object.__new__(FinGraph)  # 不走 __init__，避免初始化模型/检查点
    g.preprocessor = SimpleNamespace(model=model)
    return g


def _no_entity_parse():
    return {
        "status": "ready",
        "intent": {"type": "SUGGESTIONS"},
        "entities": {"symbols": [], "names": [], "code": []},
    }


def test_route_no_entity_goes_to_sector_not_clarify():
    g = _graph()
    out = g._route({"input": "科技股还能买吗", "parse_result": _no_entity_parse()})
    assert out["next_agent"] == "sector_node"
    assert out["route_info"]["mode"] == "sector"
    assert out["route_info"]["source"] == "sector"
    assert out["next_agent"] != "clarify_node"


def test_llm_explicit_clarification_still_clarifies():
    """LLM 明确判定需要澄清（如指代不清）时，仍走澄清，不被板块分支吞掉。"""
    g = _graph()
    parse = {"status": "clarification_needed", "intent": {"type": "MARKET_DATA"},
             "entities": {}, "clarification": {"message": "请说明是哪只股票"}}
    out = g._route({"input": "这个怎么样", "parse_result": parse})
    assert out["next_agent"] == "clarify_node"


def test_fallback_parse_without_ticker_is_not_a_dead_end():
    g = _graph()
    parse = g._fallback_parse("科技股还能买吗")
    assert parse["status"] == "ready"
    assert parse["clarification"] is None
    route = g._route({"input": "科技股还能买吗", "parse_result": parse})
    assert route["next_agent"] == "sector_node"


def test_sector_node_without_model_gives_guidance_not_error():
    g = _graph(model=None)
    out = asyncio.run(g._sector({"input": "科技股还能买吗"}))
    assert out["status"] == "completed"
    assert "股票名称或代码" in out["result"]


def test_sector_node_uses_grounded_context_and_no_invented_numbers(monkeypatch):
    seen = {}

    class FakeModel:
        async def ainvoke(self, messages):
            seen["system"] = messages[0].content
            return SimpleNamespace(content="框架回答")

    async def fake_ctx():
        return ("美股科技篮子（14 只）：上涨 9 只。", {"count": 14})

    monkeypatch.setattr(market_context, "tech_market_context", fake_ctx)
    g = _graph(model=FakeModel())
    out = asyncio.run(g._sector({"input": "科技股还能买吗"}))
    assert out["result"] == "框架回答"
    assert "上涨 9 只" in seen["system"]
    assert "不要编造" not in seen["system"]


def test_sector_node_without_data_tells_model_not_to_invent(monkeypatch):
    seen = {}

    class FakeModel:
        async def ainvoke(self, messages):
            seen["system"] = messages[0].content
            return SimpleNamespace(content="ok")

    async def no_ctx():
        return ("", None)

    monkeypatch.setattr(market_context, "tech_market_context", no_ctx)
    asyncio.run(_graph(model=FakeModel())._sector({"input": "科技股还能买吗"}))
    assert "不要编造" in seen["system"]


def test_classify_route_sector():
    assert classify_route([{"stage": "preprocess"}, {"stage": "sector"}]) == "sector"


def test_market_context_format_and_stats():
    quotes = [
        {"symbol": "NVDA", "name": "NVIDIA", "price": 100.0, "changePercent": 2.0},
        {"symbol": "AAPL", "name": "Apple", "price": 50.0, "changePercent": -1.0},
        {"symbol": "MSFT", "name": "Microsoft", "price": 80.0, "changePercent": 0.5},
    ]
    text, stats = market_context._format(quotes)
    assert stats["up"] == 2 and stats["down"] == 1
    assert stats["top"][0] == "NVDA" and stats["bottom"][-1] == "AAPL"
    assert "NVDA（NVIDIA）" in text and "+2.00%" in text


def test_market_context_returns_empty_when_source_down(monkeypatch):
    async def down(symbols, **kw):
        return []
    monkeypatch.setattr(market_context, "fetch_us_quotes", down)
    market_context._cache.update(ts=0.0, text=None, stats=None)
    assert asyncio.run(market_context.tech_market_context()) == ("", None)
