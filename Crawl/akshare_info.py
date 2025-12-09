import akshare as ak
from datetime import datetime, timedelta

end_date = datetime.now().strftime("%Y%m%d")
start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

df = ak.stock_zh_a_hist(
    symbol="600519", period="daily", start_date=start_date, end_date=end_date
)

df = df.sort_values("日期", ascending=False)

print(df.head())
