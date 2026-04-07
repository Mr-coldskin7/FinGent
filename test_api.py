#!/usr/bin/env python3
"""测试图表 API"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Trade.visualizer import parse_backtest_audit, extract_trades_and_signals, generate_chart_data

print("=" * 60)
print("Test Chart Data Generation")
print("=" * 60)

# 解析审计文件
audit_path = "Trade/backtest_audit.jsonl"
print(f"\n1. Parsing audit file: {audit_path}")
records = parse_backtest_audit(audit_path)
print(f"   Found {len(records)} records")

if records:
    print(f"\n2. First record:")
    first = records[0]
    print(f"   Date: {first.get('date')}")
    print(f"   State keys: {list(first.get('state', {}).keys())}")
    
    print(f"\n3. Extracting trade data...")
    daily_records, trades = extract_trades_and_signals(records)
    print(f"   Daily records: {len(daily_records)}")
    print(f"   Trades: {len(trades)}")
    
    if daily_records:
        print(f"\n4. First daily record:")
        dr = daily_records[0]
        print(f"   Date: {dr.date}")
        print(f"   OHLC: {dr.open_price}, {dr.high_price}, {dr.low_price}, {dr.close_price}")
        print(f"   Volume: {dr.volume}")
    
    print(f"\n5. Generating chart data...")
    chart_data = generate_chart_data(daily_records, trades, "TEST")
    print(f"   Candles: {len(chart_data.get('candles', []))}")
    print(f"   Trade markers: {len(chart_data.get('trade_markers', []))}")
    
    if chart_data.get('candles'):
        print(f"\n6. First candle:")
        print(f"   {chart_data['candles'][0]}")
    
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
else:
    print("\nFAILED: No records found")
