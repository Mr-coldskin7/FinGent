#!/usr/bin/env python3
"""测试图表 API"""
import requests
import json

def test_chart_api():
    url = "http://localhost:8000/api/v1/backtest-chart"
    params = {
        "symbol": "AAPL",
        "audit_path": "Trade/backtest_audit.jsonl"
    }
    
    print(f"Testing: {url}")
    print(f"Params: {params}")
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        print(f"Status: {resp.status_code}")
        
        data = resp.json()
        print(f"\nResponse:")
        print(json.dumps(data, indent=2, ensure_ascii=False)[:2000])
        
        if data.get("success") and data.get("data"):
            candles = data["data"].get("candles", [])
            markers = data["data"].get("trade_markers", [])
            print(f"\n✅ 成功! {len(candles)} 根K线, {len(markers)} 个交易标记")
            
            if candles:
                print(f"\n第一根K线: {candles[0]}")
                print(f"最后一根K线: {candles[-1]}")
        else:
            print(f"\n❌ 失败: {data.get('error')}")
            
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    test_chart_api()
