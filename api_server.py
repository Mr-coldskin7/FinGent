from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse
import asyncio
from typing import Optional
import queue
import pandas as pd
load_dotenv()

# ========== 初始化 Graph ==========
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from langchain_community.chat_models import ChatTongyi
from LLM.preprocess import Preprocessor
from LLM.router import Router
from LLM.agent import TECHNICAL_NERD, Morefit
from LLM.graph import FinGraph
from LLM.unified_stock_tools import *

# MySQL 持久化 - 使用上下文管理器
_mysql_saver_ctx = PyMySQLSaver.from_conn_string(os.getenv("MYSQL_URL", "mysql+pymysql://root:password@localhost:3306/fingent"))
_mysql_saver = _mysql_saver_ctx.__enter__()
_mysql_saver.setup()
checkpointer = _mysql_saver

model = ChatTongyi(api_key=os.getenv("QIANWEN_API_KEY"), temperature=0.5)

fin_graph = FinGraph(
    preprocessor=Preprocessor(model, checkpointer),
    router=Router(),
    agent={
        'TECHNICAL_NERD': TECHNICAL_NERD(model, [get_stock_price, get_stock_basic_info], checkpointer),
        'Morefit': Morefit(model, [get_stock_company_info, get_stock_financial_report_links, 
                                   get_stock_financial_statements, get_stock_price, get_stock_basic_info], 
                          checkpointer)
    },
    checkpointer=checkpointer
)

# 注册关闭钩子（程序退出时清理）
import atexit
@atexit.register
def cleanup():
    _mysql_saver_ctx.__exit__(None, None, None)

# ========== API ==========
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    user_input: str
    thread_id: Optional[str] = None

@app.post("/api/v1/chat")
async def chat(req: ChatRequest):
    """直接透传给 Graph.run()"""
    thread_id = req.thread_id or f"chat_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    
    result = await fin_graph.run(req.user_input, thread_id)
    
    print(f"\n{'='*60}")
    print(f"DEBUG: Graph result keys: {list(result.keys())}")
    print(f"{'='*60}\n")
    
    response = {
        "success": True,
        "thread_id": thread_id,
        "input": result.get("input"),
        "parse_result": result.get("parse_result"),
        "result": result.get("result"),
        "status": result.get("status"),
    }
    
    if result.get("tool_chain"):
        response["tool_chain"] = result["tool_chain"]
    if result.get("agent_name"):
        response["agent_name"] = result["agent_name"]
    if result.get("stock"):
        response["stock"] = result["stock"]
    if result.get("all_tool_chains"):
        response["all_tool_chains"] = result["all_tool_chains"]
    if result.get("final_decision"):
        response["final_decision"] = result["final_decision"]
    if result.get("detailed_analysis"):
        response["detailed_analysis"] = result["detailed_analysis"]
    
    return response


class BacktestRequest(BaseModel):
    symbol: str = "GOOGL"
    start: str = "2024-01-01"
    end: Optional[str] = None
    initial_cash: float = 10000.0
    commission: float = 0.001
    slippage: float = 0.0005
    min_confidence: float = 0.0
    rebalance_threshold: float = 0.02
    quiet: bool = False
    temperature: float = 0.0
    audit_path: str = "Trade/backtest_audit.jsonl"


@app.post("/api/v1/backtest")
async def backtest(req: BacktestRequest):
    """执行回测任务的API接口"""
    try:
        from Trade.runner import run_backtest_from_symbol
        import asyncio
        from functools import partial
        
        os.environ["FINGENT_BACKTEST_STRICT"] = "1"
        
        temp_model = ChatTongyi(api_key=os.getenv("QIANWEN_API_KEY"), temperature=req.temperature)
        _temp_mysql_ctx = PyMySQLSaver.from_conn_string(os.getenv("MYSQL_URL", "mysql+pymysql://root:password@localhost:3306/fingent"))
        temp_checkpointer = _temp_mysql_ctx.__enter__()
        temp_checkpointer.setup()
        
        temp_graph = FinGraph(
            preprocessor=Preprocessor(temp_model, temp_checkpointer),
            router=Router(),
            agent={
                'TECHNICAL_NERD': TECHNICAL_NERD(temp_model, [get_stock_price, get_stock_basic_info], temp_checkpointer),
                'Morefit': Morefit(temp_model, [get_stock_company_info, get_stock_financial_report_links, 
                               get_stock_financial_statements, get_stock_price, get_stock_basic_info], 
                      temp_checkpointer)
            },
            checkpointer=temp_checkpointer
        )
        
        loop = asyncio.get_event_loop()
        func = partial(run_backtest_from_symbol,
                       fin_graph=temp_graph,
                       symbol=req.symbol,
                       start=req.start,
                       end=req.end,
                       initial_cash=req.initial_cash,
                       commission=req.commission,
                       slippage_perc=req.slippage,
                       min_confidence=req.min_confidence,
                       rebalance_threshold=req.rebalance_threshold,
                       printlog=(not req.quiet),
                       audit_path=req.audit_path)
        result = await loop.run_in_executor(None, func)
        
        return {
            "success": True,
            "result": result,
            "symbol": req.symbol,
            "period": {"start": req.start, "end": req.end}
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# 全局变量用于流式回测
backtest_queue = None
backtest_done = False
backtest_error = None
backtest_cancelled = False  # 取消标志


def reset_backtest_state():
    """重置回测状态"""
    global backtest_queue, backtest_done, backtest_error, backtest_cancelled
    backtest_queue = None
    backtest_done = False
    backtest_error = None
    backtest_cancelled = False


def is_backtest_cancelled():
    """检查回测是否被取消"""
    global backtest_cancelled
    return backtest_cancelled


def create_strategy_with_callback(StrategyClass, data_queue, cancel_check_fn=None, **kwargs):
    """创建一个带有回调函数的策略类"""
    class CallbackStrategy(StrategyClass):
        def __init__(self):
            super().__init__()
            self.data_queue = data_queue
            self.cancel_check_fn = cancel_check_fn
        
        def on_daily_update(self, state):
            """回调函数：将每日状态推送到队列"""
            try:
                self.data_queue.put(state, block=False)
            except queue.Full:
                pass  # 队列满了，丢弃旧数据
        
        def next(self):
            # 检查是否被取消
            if self.cancel_check_fn and self.cancel_check_fn():
                self.log("回测已取消，正在停止...")
                self.env.runstop()
                return
            # 调用父类的 next
            super().next()
    
    # 复制原策略的参数定义
    CallbackStrategy.params = StrategyClass.params
    return CallbackStrategy


@app.post("/api/v1/backtest-stream")
async def backtest_stream(req: BacktestRequest):
    """流式回测API - 逐日返回结果
    
    SSE 事件格式:
    - event: start - 回测开始
    - event: daily_update - 每日更新
    - event: final_result - 最终结果
    - event: error - 错误信息
    - event: cancelled - 回测被取消
    - event: ping - 保持连接
    """
    import json
    import threading
    from Trade.runner import GraphSignalStrategy, load_price_dataframe
    import backtrader as bt
    
    # 重置取消状态
    reset_backtest_state()
    
    # 创建队列用于线程间通信
    global backtest_queue
    backtest_queue = queue.Queue(maxsize=1000)
    result_holder = {'strategy': None, 'start_value': 0, 'end_value': 0, 
                     'analyzers': {}, 'error': None, 'done': False}
    
    def run_backtest():
        """在后台线程中运行回测"""
        global backtest_cancelled, backtest_done, backtest_error
        try:
            os.environ["FINGENT_BACKTEST_STRICT"] = "1"
            
            print(f"[StreamBacktest] 加载 {req.symbol} 数据...")
            data = load_price_dataframe(symbol=req.symbol, start=req.start, end=req.end)
            print(f"[StreamBacktest] 加载完成: {len(data)} 条")
            
            # 检查是否已取消
            if backtest_cancelled:
                print(f"[StreamBacktest] 回测在启动前被取消")
                backtest_done = True
                return
            
            temp_model = ChatTongyi(api_key=os.getenv("QIANWEN_API_KEY"), temperature=req.temperature)
            _temp_mysql_ctx = PyMySQLSaver.from_conn_string(os.getenv("MYSQL_URL", "mysql+pymysql://root:password@localhost:3306/fingent"))
            temp_checkpointer = _temp_mysql_ctx.__enter__()
            temp_checkpointer.setup()
            
            temp_graph = FinGraph(
                preprocessor=Preprocessor(temp_model, temp_checkpointer),
                router=Router(),
                agent={
                    'TECHNICAL_NERD': TECHNICAL_NERD(temp_model, [get_stock_price, get_stock_basic_info], temp_checkpointer),
                    'Morefit': Morefit(temp_model, [get_stock_company_info, get_stock_financial_report_links, 
                                   get_stock_financial_statements, get_stock_price, get_stock_basic_info], 
                          temp_checkpointer)
                },
                checkpointer=temp_checkpointer
            )
            
            cerebro = bt.Cerebro()
            cerebro.broker.setcash(req.initial_cash)
            cerebro.broker.setcommission(commission=req.commission)
            if req.slippage > 0:
                cerebro.broker.set_slippage_perc(req.slippage)
            
            feed = bt.feeds.PandasData(dataname=data)
            cerebro.adddata(feed, name=req.symbol)
            
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
            cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
            
            # 创建带回调的策略类，传入取消检查函数
            CallbackStrategy = create_strategy_with_callback(
                GraphSignalStrategy, backtest_queue, cancel_check_fn=is_backtest_cancelled
            )
            
            cerebro.addstrategy(
                CallbackStrategy,
                graph=temp_graph,
                symbol=req.symbol,
                min_confidence=req.min_confidence,
                rebalance_threshold=req.rebalance_threshold,
                printlog=True,
                audit_path=req.audit_path
            )
            
            result_holder['start_value'] = float(cerebro.broker.getvalue())
            print(f"[StreamBacktest] 开始运行回测...")
            
            # 检查是否已取消
            if backtest_cancelled:
                print(f"[StreamBacktest] 回测在运行前被取消")
                backtest_done = True
                return
            
            results = cerebro.run()
            
            if backtest_cancelled:
                print(f"[StreamBacktest] 回测被取消")
            else:
                print(f"[StreamBacktest] 回测完成!")
            
            if results:
                strategy = results[0]
                result_holder['strategy'] = strategy
                result_holder['end_value'] = float(cerebro.broker.getvalue())
                try:
                    result_holder['analyzers'] = {
                        "drawdown": strategy.analyzers.drawdown.get_analysis(),
                        "returns": strategy.analyzers.returns.get_analysis(),
                        "sharpe": strategy.analyzers.sharpe.get_analysis(),
                        "trades": strategy.analyzers.trades.get_analysis(),
                    }
                except:
                    pass
            
            result_holder['done'] = True
            
        except Exception as e:
            print(f"[StreamBacktest] 错误: {e}")
            import traceback
            traceback.print_exc()
            result_holder['error'] = str(e)
            result_holder['done'] = True


    async def event_generator():
        """生成器：从队列读取数据并yield"""
        global backtest_cancelled
        
        # 启动后台线程
        thread = threading.Thread(target=run_backtest)
        thread.start()
        
        # 发送开始事件
        # 计算总天数
        total_days_estimate = 252  # 默认252个交易日
        if req.end:
            try:
                start_date = pd.to_datetime(req.start)
                end_date = pd.to_datetime(req.end)
                total_days_estimate = len(pd.bdate_range(start_date, end_date))
            except:
                pass
        
        yield {
            "event": "start",
            "data": json.dumps({
                "symbol": req.symbol,
                "message": "回测开始",
                "period": {"start": req.start, "end": req.end},
                "total_days": total_days_estimate
            }, ensure_ascii=False)
        }
        
        total_days = 0
        max_wait = 7200  # 120分钟超时（回测可能需要很长时间）
        waited = 0
        last_data_time = asyncio.get_event_loop().time()  # 记录最后收到数据的时间
        
        # 循环读取队列数据
        while not result_holder['done'] or not backtest_queue.empty():
            # 检查是否被取消
            if backtest_cancelled:
                print("[StreamBacktest] 检测到取消信号，停止事件生成")
                yield {
                    "event": "cancelled",
                    "data": json.dumps({
                        "message": "回测已取消",
                        "total_days": total_days
                    }, ensure_ascii=False)
                }
                return
            
            try:
                # 非阻塞获取数据
                state = backtest_queue.get(block=False)
                total_days += 1
                
                # 构建每日结果 - 包含OHLCV和信号信息
                daily_result = {
                    "date": str(state["date"]),
                    "cash": float(state["cash"]),
                    "portfolio_value": float(state["portfolio_value"]),
                    "position_size": int(state["position_size"]) if state["position_size"] is not None else 0,
                    "avg_cost": float(state["avg_cost"]) if state["avg_cost"] is not None else 0,
                    # OHLCV 数据
                    "open_price": float(state.get("open_price", state["close_price"])),
                    "high_price": float(state.get("high_price", state["close_price"])),
                    "low_price": float(state.get("low_price", state["close_price"])),
                    "close_price": float(state["close_price"]),
                    "volume": float(state.get("volume", 0)),
                    "day_number": total_days
                }
                
                # 如果有信号信息，添加到结果中
                print(f"[StreamBacktest] Day {total_days}: signal in state={'signal' in state}, state['signal']={state.get('signal')}")
                if "signal" in state and state["signal"]:
                    daily_result["signal"] = {
                        "vote": state["signal"].get("vote", ""),
                        "confidence": float(state["signal"].get("confidence", 0)),
                        "target_position_pct": float(state["signal"].get("target_position_pct", 0)),
                        "reason": state["signal"].get("reason", "")
                    }
                    print(f"[StreamBacktest] Day {total_days}: signal={daily_result['signal']}")
                
                yield {
                    "event": "daily_update",
                    "data": json.dumps(daily_result, ensure_ascii=False, default=str)
                }
                
                # 更新最后收到数据的时间
                last_data_time = asyncio.get_event_loop().time()
                
                # 让出控制权，确保数据及时发送
                await asyncio.sleep(0.01)
                
            except queue.Empty:
                # 队列为空，等待一下
                if not result_holder['done']:
                    await asyncio.sleep(0.1)
                    waited += 0.1
                    
                    # 检查是否长时间没有收到数据（可能是卡住了）
                    time_since_last_data = asyncio.get_event_loop().time() - last_data_time
                    if time_since_last_data > 300:  # 5分钟没有数据
                        print(f"[StreamBacktest] 警告: {time_since_last_data:.0f}秒没有收到数据")
                        # 发送心跳保持连接
                        yield {
                            "event": "ping",
                            "data": json.dumps({"status": "waiting", "elapsed": time_since_last_data}, ensure_ascii=False)
                        }
                        last_data_time = asyncio.get_event_loop().time()  # 重置时间
                    
                    if waited > max_wait:
                        yield {
                            "event": "error",
                            "data": json.dumps({"error": f"回测执行超时（{max_wait/60:.0f}分钟）"}, ensure_ascii=False)
                        }
                        return
        
        # 等待线程结束
        thread.join(timeout=5)
        
        # 发送最终结果
        if backtest_cancelled:
            # 回测被取消，不发送最终结果
            print("[StreamBacktest] 回测已被取消，跳过最终结果")
        elif result_holder['error']:
            yield {
                "event": "error",
                "data": json.dumps({"error": result_holder['error']}, ensure_ascii=False)
            }
        elif result_holder['strategy']:
            final_result = {
                "symbol": req.symbol,
                "start_value": result_holder['start_value'],
                "end_value": result_holder['end_value'],
                "pnl": result_holder['end_value'] - result_holder['start_value'],
                "return_pct": ((result_holder['end_value'] / result_holder['start_value']) - 1.0) * 100.0 if result_holder['start_value'] else 0.0,
                "last_signal": result_holder['strategy'].last_signal,
                "total_days": total_days,
                "analyzers": result_holder['analyzers']
            }
            
            yield {
                "event": "final_result",
                "data": json.dumps(final_result, ensure_ascii=False, default=str)
            }
    
    return EventSourceResponse(
        event_generator(),
        ping=15,  # 每15秒发送一次ping保持连接
        ping_message_factory=lambda: {"event": "ping", "data": "{}"}
    )


@app.post("/api/v1/backtest-cancel")
async def cancel_backtest():
    """
    取消正在进行的回测
    """
    global backtest_cancelled
    backtest_cancelled = True
    print("[Backtest] 收到取消请求")
    return {
        "success": True,
        "message": "回测取消信号已发送"
    }


@app.get("/api/v1/backtest-chart")
async def get_backtest_chart(audit_path: str = "Trade/backtest_audit.jsonl", symbol: str = "STOCK"):
    """
    获取回测图表数据，用于前端可视化
    
    参数:
        audit_path: 回测审计文件路径
        symbol: 股票代码
    
    返回:
        {
            "symbol": "NVDA",
            "candles": [{"time": "2024-01-01", "open": 100, "high": 105, "low": 99, "close": 102, "volume": 10000}],
            "trade_markers": [{"time": "2024-01-05", "type": "buy", "price": 102.0, "size": 10}],
            "statistics": {...}
        }
    """
    import os
    print(f"[ChartAPI] 请求图表数据: symbol={symbol}, audit_path={audit_path}")
    
    try:
        from Trade.visualizer import parse_backtest_audit, extract_trades_and_signals, generate_chart_data
        
        # 检查文件是否存在
        if not os.path.exists(audit_path):
            # 尝试其他路径
            alt_paths = [
                f"Trade/{audit_path}",
                audit_path.replace("Trade/", ""),
                os.path.join("Trade", "backtest_audit.jsonl"),
                "backtest_audit.jsonl"
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    audit_path = alt_path
                    print(f"[ChartAPI] 使用替代路径: {audit_path}")
                    break
        
        print(f"[ChartAPI] 最终审计文件路径: {audit_path}, 存在: {os.path.exists(audit_path)}")
        
        # 解析回测数据
        records = parse_backtest_audit(audit_path)
        print(f"[ChartAPI] 解析到 {len(records)} 条记录")
        
        if not records:
            return {
                "success": False,
                "error": f"没有找到回测数据，请先运行回测 (路径: {audit_path})"
            }
        
        # 提取交易信息
        daily_records, trades = extract_trades_and_signals(records)
        print(f"[ChartAPI] 提取到 {len(daily_records)} 日记录, {len(trades)} 笔交易")
        
        # 生成图表数据
        chart_data = generate_chart_data(daily_records, trades, symbol)
        print(f"[ChartAPI] 生成图表数据: {len(chart_data.get('candles', []))} 根K线")
        
        return {
            "success": True,
            "data": chart_data
        }
        
    except Exception as e:
        import traceback
        print(f"[ChartAPI] 错误: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/health")
async def health(): 
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
