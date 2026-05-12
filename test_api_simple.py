#!/usr/bin/env python3
"""简单测试 API"""
import requests
import json


def test_chart_api():
    url = "http://localhost:8000/api/v1/backtest-chart"
    params = {"symbol": "GOOGL"}

    print(f"Testing: {url}")
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        if data.get("success"):
            print(f"✅ API 正常!")
            print(f"   K线数量: {len(data['data'].get('candles', []))}")
            print(f"   交易标记: {len(data['data'].get('trade_markers', []))}")
            if data["data"].get("candles"):
                print(f"   示例K线: {data['data']['candles'][0]}")
        else:
            print(f"❌ API 错误: {data.get('error')}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")


if __name__ == "__main__":
    test_chart_api()
