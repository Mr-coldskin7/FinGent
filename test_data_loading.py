"""
测试修复后的数据加载功能
"""

import sys

sys.path.insert(0, r"E:\FinGent")

from Trade.runner import load_price_dataframe

print("测试加载 GOOGL 2023 年的价格数据...")
try:
    df = load_price_dataframe(symbol="GOOGL", start="2023-01-01", end="2023-12-31")
    print(f"✓ 成功加载数据！")
    print(f"  数据条数: {len(df)}")
    print(f"  日期范围: {df.index.min()} 到 {df.index.max()}")
    print(f"  列: {list(df.columns)}")
    print(f"  前5行:")
    print(df.head())
    print(f"  后5行:")
    print(df.tail())
except Exception as e:
    print(f"✗ 加载失败: {e}")
    import traceback

    traceback.print_exc()

print("\n测试加载 AAPL 2024 年的价格数据...")
try:
    df = load_price_dataframe(symbol="AAPL", start="2024-01-01", end="2024-12-31")
    print(f"✓ 成功加载数据！")
    print(f"  数据条数: {len(df)}")
    print(f"  日期范围: {df.index.min()} 到 {df.index.max()}")
except Exception as e:
    print(f"✗ 加载失败: {e}")
    import traceback

    traceback.print_exc()
