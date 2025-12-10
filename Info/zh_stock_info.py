import pandas as pd
import akshare as ak
from datetime import datetime

#========================================================================================================================
df = pd.read_csv("..\Data\zh_stock_map.csv")
mapping = dict(zip(df['code'].astype(str).str.strip(), df['name'].str.strip()))
#=========================================================================================================================
def input2zh(input_str: str) -> str:
    """
    input_normalize 的 Docstring
    将“600519”和“sh600519”转化为化为“贵州茅台"
    :param input_str: 数字类型或者其他类型的字符串
    :type input_str: str
    :return: 中文股票名称
    :rtype: str
    """
    if input_str.startswith(('sh', 'sz', 'bj')):
        input_str = input_str[2:]
    return mapping.get(input_str, "")

def input2code(input_str: str) -> str:
    """
    input_normalize 的 Docstring
    将“贵州茅台”转化为化为“sh600519"
    :param input_str: 输入的字符串
    :type input_str: str
    :return: 股票代码
    :rtype: str
    """
    if input_str.isdigit():
        return "sh"+input_str
    return "sh"+str(df[df['name'] == input_str]['code'].values[0])

def input2number(input_str: str) -> str:
    """
    input_normalize 的 Docstring
    将“贵州茅台”转化为化为“600519"
    :param input_str: 输入的股票名称
    :type input_str: str
    :return: 输入的股票代码
    :rtype: str
    """
    if input_str.startswith(('sh', 'sz', 'bj')):
        return input_str[2:] 
    return str(df[df['name'] == input_str]['code'].values[0])

def get_see_summary():
    """
    获取上交所概况数据
    """
    stock_sse_summary_df = ak.stock_sse_summary()
    print(stock_sse_summary_df)
    return stock_sse_summary_df

def get_sz_stock_info():
    """
    获取深交所概况数据
    """
    stock_szse_summary_df = ak.stock_szse_summary()
    print(stock_szse_summary_df)
    return stock_szse_summary_df

def get_stock_info(data: str):
    """
    param symbol: choice of {"主板A股", "主板B股", "科创板"}
    获取股票信息
    """
    stock_info_df = ak.stock_info_sh_name_code(symbol=data)
    print(stock_info_df)
    return stock_info_df

def stock_individual_info(symbol: str):
    """
    获取个股信息
    :param symbol: 股票代码，如 '600519'
    :return: 个股信息 DataFrame
    """
    if not symbol.isdigit():
        symbol = input2code(symbol)
    stock_individual_info_df = ak.stock_individual_basic_info_xq(symbol=symbol)
    print(stock_individual_info_df)
    return stock_individual_info_df

def stock_individual_price_info_recent(symbol: str):
    """
    获取个股近期信息
    :param symbol: 股票
    :return: 个股近期信息 DataFrame
    """
    symbol = input2number(symbol)
    stock_individual_price_info_recent_df = ak.stock_bid_ask_em(symbol=symbol)
    print(stock_individual_price_info_recent_df)
    return stock_individual_price_info_recent_df

def recent_stock_list():
    """
    实时行情数据
    :return:DataFrame
    """
    stock_zh_a_spot = ak.stock_zh_a_spot()
    print(stock_zh_a_spot)
    return stock_zh_a_spot
def get_financial_report(stock_num:str): 
    """
    获取财务报告数据
    :return:DataFrame
    """
    stock_financial_report_sina_df = ak.stock_financial_abstract(symbol=stock_num)
    print(stock_financial_report_sina_df.head())
    return stock_financial_report_sina_df

if __name__ == "__main__":
    stock_individual_price_info_recent("贵州茅台")