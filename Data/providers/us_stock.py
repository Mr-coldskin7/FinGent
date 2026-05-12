import warnings
import os
import re
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import pandas as pd
import httpx
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List
from dotenv import load_dotenv

# ===============================================================================
# use FMP to get US stock info
# ===============================================================================
load_dotenv()
fmp_api_key = os.getenv("FMP_API_KEY")
tiingo_api_key = os.getenv("TIINGO_API_KEY")
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
_HEADERS = {"User-Agent": "hw2258@bath.ac.uk"}
# 文件级异步 HTTP 客户端（复用连接池）
_async_client = httpx.AsyncClient(headers=_HEADERS)


# ===============================================================================
def _cik10(cik: str) -> str:
    """Left-zero-pad CIK to 10 digits."""
    return cik.zfill(10)


from typing import Optional


def input2Symbol(name: str = "Apple Inc.") -> Optional[str]:
    """
    get company symbol by name or symbol itself
    :param name: stock name (e.g., 'Apple Inc.') or symbol (e.g., 'AAPL')
    :return: symbol string or None
    """
    import os

    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    csv_path = os.path.join(base_dir, "Data", "us_stock_cik_map.csv")
    df = pd.read_csv(csv_path)
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["name"] = df["name"].astype(str).str.strip()

    input_upper = str(name).strip().upper()
    row_by_ticker = df[df["ticker"] == input_upper]
    if not row_by_ticker.empty:
        return str(row_by_ticker["ticker"].values[0])

    input_name = str(name).strip()
    row_by_name = df[df["name"].str.lower() == input_name.lower()]
    if not row_by_name.empty:
        return str(row_by_name["ticker"].values[0])

    return None


def input2Name(symbol: str = "AAPL") -> Optional[str]:
    """
    get company name by symbol
    :param symbol: stock symbol e.g., 'AAPL'
    :return: company name string or None
    """
    import os

    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    csv_path = os.path.join(base_dir, "Data", "us_stock_cik_map.csv")
    df = pd.read_csv(csv_path)
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["name"] = df["name"].astype(str).str.strip()

    input_upper = str(symbol).strip().upper()
    row = df[df["ticker"] == input_upper]
    return str(row["name"].values[0]) if not row.empty else None


async def get_basic_info_by_symbol(symbol: str):
    """
    get company basic info by symbol
    :param symbol: stock name exp: 'AAPL'
    :return: basic_info DataFrame
    'symbol', 'name', 'currency', 'exchangeFullName', 'exchange'
    """
    symbol = input2Symbol(symbol)
    url = f"https://financialmodelingprep.com/stable/search-symbol?query={symbol}&apikey={fmp_api_key}"
    response = await _async_client.get(url)
    data = response.json()
    return data


async def get_basic_info_by_name(name: str):
    """
    get company basic info by name
    :param name: stock name exp: 'AAPL'
    :return: basic_info DataFrame
    'symbol', 'name', 'currency', 'exchangeFullName', 'exchange'
    """
    name = input2Name(name)
    url = f"https://financialmodelingprep.com/stable/search-name?query={name}&apikey={fmp_api_key}"
    response = await _async_client.get(url)
    data = response.json()
    return data


async def get_historical_stock_price_by_symbol(
    symbol: str = "AAPL",
    startDate: str = None,
    endDate: str = None,
    resampleFreq: str = "daily",
):
    """
    get historical stock price by symbol
    :param symbol: stock name exp: 'AAPL'
    :param startDate: start date exp: '2023-01-01'
    :param endDate: end date exp: '2023-12-31'
    :param resampleFreq: resample frequency exp: 'daily','weekly','monthly','annually'
    :return: stock_price DataFrame
    'symbol', 'name', 'currency', 'exchangeFullName', 'exchange'
    """
    symbol = input2Symbol(symbol)
    datetime_format = "%Y-%m-%d"
    if startDate is None:
        startDate = (datetime.now() - timedelta(days=365)).strftime(datetime_format)
    if endDate is None:
        endDate = datetime.now().strftime(datetime_format)
    if startDate > endDate:
        raise ValueError("startDate must be before endDate")
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate={startDate}&endDate={endDate}&token={tiingo_api_key}&resampleFreq={resampleFreq}"
    response = await _async_client.get(url)
    return response.json()


async def get_description(symbol: str = "AAPL"):
    """
    get company description by symbol
    :param symbol: stock name exp: 'AAPL'
    :return: description DataFrame
    """
    symbol = input2Symbol(symbol)
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}?token={tiingo_api_key}"
    response = await _async_client.get(url)
    return response.json()


# def get_relative_news_by_stock(symbol: str="AAPL"):
#     """
#     get company relative news by symbol
#     :param symbol: stock name exp: 'AAPL'
#     :return: news DataFrame
#     """
#     requestResponse = requests.get(f"https://api.tiingo.com/tiingo/news?tickers={symbol}&token={tiingo_api_key}", headers=_HEADERS)
#     return pd.DataFrame([requestResponse.json()])

# def get_relative_news_by_tag(symbol: str="election"):
#     """
#     get company relative news by symbol
#     :param symbol: stock tag exp: 'election'
#     :return: news DataFrame
#     """
#     requestResponse = requests.get(f"https://api.tiingo.com/tiingo/news?tag={symbol}&token={tiingo_api_key}", headers=_HEADERS)
#     return pd.DataFrame([requestResponse.json()])


def get_cik_code_by_symbol0rName(symbol: str = "AAPL"):
    """
    get company cik code by symbol or name
    :param symbol: stock name exp: 'AAPL'or full name 'Apple Inc.'
    :return: cik code DataFrame
    """
    symbol = input2Symbol(symbol)
    import os

    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    csv_path = os.path.join(base_dir, "Data", "us_stock_cik_map.csv")
    df = pd.read_csv(csv_path)
    df["ticker"] = df["ticker"].str.strip()
    df["name"] = df["name"].str.strip()
    row = df[df["ticker"].eq(symbol) | df["name"].eq(symbol)]
    return str(row["cik"].values[0]) if not row.empty else None


async def get_10Q_financial_report(symbol: str = "AAPL", num: int = 1):
    """
    get company's 10Q financial report by symbol
    :param symbol: stock name exp: 'AAPL'
    :param num: number of reports to retrieve
    :return: financial report DataFrame
    """
    symbol = input2Symbol(symbol)
    cik_num = get_cik_code_by_symbol0rName(symbol)
    cik10 = _cik10(cik_num)
    response = await _async_client.get(
        f"https://data.sec.gov/submissions/CIK{cik10}.json"
    )
    sub = response.json()
    recent_df = pd.DataFrame(sub["filings"]["recent"])
    q_dfs = recent_df[recent_df["form"] == "10-Q"].head(num)
    file_names = q_dfs["primaryDocument"].tolist()
    accessionNumbers = q_dfs["accessionNumber"].tolist()
    urls = []
    for i in range(len(file_names)):
        file_name = file_names[i]
        accessionNumber = accessionNumbers[i].replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{accessionNumber}/{file_name}"
        urls.append(url)
    return urls


async def fuzzy_search_symbols(symbol: str = "AAPL"):
    """
    Tiingo's search feature lets you find specific assets in our database by the ticker or the name of the asset.
    This endpoint lets you segment by active, delisted, tickers across asset classes.
    The endpoint first searches for ticker matches and then expands to matches in the name of the asset.
    This endpoint is useful for looking up existing assets
    """
    url = f"https://api.tiingo.com/tiingo/utilities/search?query={symbol}&token={tiingo_api_key}"
    response = await _async_client.get(url)
    data = response.json()
    return data


async def get_10K_financial_report(symbol: str = "AAPL", num: int = 1):
    """
    get company's 10k financial report by symbol
    :param symbol: stock name exp: 'AAPL'
    :param num: number of reports to retrieve
    :return: financial report DataFrame
    """
    symbol = input2Symbol(symbol)
    cik_num = get_cik_code_by_symbol0rName(symbol)
    cik10 = _cik10(cik_num)
    response = await _async_client.get(
        f"https://data.sec.gov/submissions/CIK{cik10}.json"
    )
    sub = response.json()
    recent_df = pd.DataFrame(sub["filings"]["recent"])
    q_dfs = recent_df[recent_df["form"] == "10-K"].head(num)
    file_names = q_dfs["primaryDocument"].tolist()
    accessionNumbers = q_dfs["accessionNumber"].tolist()
    urls = []
    for i in range(len(file_names)):
        file_name = file_names[i]
        accessionNumber = accessionNumbers[i].replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{accessionNumber}/{file_name}"
        urls.append(url)
    return urls


async def get_financial_report(symbol: str):
    """
    Extracts financial tables from the latest 10-Q filing for a given symbol.
    This function attempts to identify and categorize common financial statements.

    Args:
        symbol (str): The stock symbol (e.g., "AAPL", "NVDA").

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary of pandas DataFrames for identified financial statements.
            - dict: A dictionary of *all* tables found in the document (for debugging/backup).
    """

    async def fetch_html_content(url, headers):
        """Fetches the HTML content of the given URL."""
        try:
            response = await _async_client.get(url)
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as e:
            print(f"Error fetching HTML content from {url}: {e}")
            return None

    def parse_all_tables(soup):
        """Parses all tables in the soup object."""
        all_tables = soup.find_all("table")
        parsed_tables = []

        for i, table in enumerate(all_tables):
            rows = table.find_all("tr")
            if not rows:
                continue

            # Attempt to parse headers and data
            # This is a simplified approach, might need refinement for complex tables
            header_rows = []
            data_rows = []

            # Find header rows (usually the first few rows with <th>)
            for row in rows:
                cells = row.find_all(["td", "th"])
                if any(cell.name == "th" for cell in cells):
                    header_rows.append([cell.get_text(strip=True) for cell in cells])
                else:
                    break  # Stop when first data row is found

            # The remaining rows are data rows
            data_start_index = len(header_rows)
            data_rows = []
            for row in rows[data_start_index:]:
                cells = row.find_all(["td", "th"])  # Treat remaining <th> as <td>
                row_data = [cell.get_text(strip=True) for cell in cells]
                data_rows.append(row_data)

            if header_rows or data_rows:  # If we found some structure
                # Flatten multi-row headers if needed, for simplicity, just take the last header row if multiple exist
                # Or concatenate them. Here, we'll take the last one as the primary header.
                if header_rows:
                    final_headers = header_rows[
                        -1
                    ]  # Use the last header row as column names
                else:
                    # If no <th> found, use generic names or first data row as header
                    if data_rows:
                        final_headers = [
                            f"Column_{j}" for j in range(len(data_rows[0]))
                        ]
                        data_rows = data_rows[
                            1:
                        ]  # Use first row as header, remove it from data
                    else:
                        continue  # Skip if no structure found

                # Ensure data rows match header length
                processed_data_rows = []
                for row_data in data_rows:
                    if len(row_data) < len(final_headers):
                        row_data.extend([None] * (len(final_headers) - len(row_data)))
                    elif len(row_data) > len(final_headers):
                        row_data = row_data[: len(final_headers)]
                    processed_data_rows.append(row_data)

                if processed_data_rows:  # Only create DataFrame if there's data
                    df = pd.DataFrame(processed_data_rows, columns=final_headers)
                    parsed_tables.append(df)

        return parsed_tables

    def identify_financial_statement(df):
        """Attempts to identify the type of financial statement based on common keywords."""
        if df.empty:
            return "Unknown_Table"

        # Get a string representation of the table content for keyword searching
        table_text = df.to_string(index=False, header=True).lower()

        # Define keywords for different financial statements
        income_keywords = [
            "revenue",
            "net sales",
            "gross profit",
            "operating income",
            "operating expenses",
            "research and development",
            "selling general and administrative",
            "sg&a",
            "net income",
            "earnings",
            "eps",
            "cost of goods sold",
            "cogs",
            "gross margin",
        ]
        balance_keywords = [
            "assets",
            "liabilities",
            "equity",
            "current assets",
            "total assets",
            "current liabilities",
            "total liabilities",
            "shareholders equity",
            "retained earnings",
            "inventory",
            "accounts receivable",
            "cash",
            "debt",
        ]
        cash_flow_keywords = [
            "operating activities",
            "investing activities",
            "financing activities",
            "net cash",
            "depreciation",
            "capital expenditures",
            "capex",
            "cash flow",
            "proceeds",
            "payments",
        ]
        # Add more keywords as needed

        # Count occurrences of keywords in the table text
        income_score = sum(1 for keyword in income_keywords if keyword in table_text)
        balance_score = sum(1 for keyword in balance_keywords if keyword in table_text)
        cash_flow_score = sum(
            1 for keyword in cash_flow_keywords if keyword in table_text
        )

        # Determine the type based on the highest score
        max_score = max(income_score, balance_score, cash_flow_score)
        if max_score == 0:
            return "Other_Financial_Table"  # Or "Unknown_Table"

        if income_score == max_score:
            return "Income_Statement"
        elif balance_score == max_score:
            return "Balance_Sheet"
        elif cash_flow_score == max_score:
            return "Cash_Flow_Statement"

        # If scores are tied, return a combined type or the first max found
        # For simplicity, returning the first one found with max score
        # This logic can be made more sophisticated
        if income_score == max_score:
            return "Income_Statement"
        elif balance_score == max_score:
            return "Balance_Sheet"
        else:  # cash_flow_score == max_score
            return "Cash_Flow_Statement"

    # --- Main Execution ---
    print(f"--- Starting extraction for symbol: {symbol} ---")

    # 1. Get the URL for the latest 10-Q (this function needs your implementation)
    symbol = input2Symbol(symbol)
    urls = await get_10Q_financial_report(symbol)
    if urls is None or not urls:
        print("Failed to get the latest 10-Q URL.")
        return {}, {}

    url = urls[0]  # Assuming the first URL is the one needed

    # 2. Fetch the HTML content
    html_text = await fetch_html_content(url, _HEADERS)
    if html_text is None:
        print("Failed to fetch HTML content.")
        return {}, {}

    # 3. Parse with BeautifulSoup
    soup = BeautifulSoup(html_text, "html.parser")

    # 4. Extract *ALL* Tables
    all_parsed_tables = parse_all_tables(soup)

    if not all_parsed_tables:
        print("No tables found in the document.")
        return {}, {}

    print(f"Found {len(all_parsed_tables)} tables in total.")

    # 5. Identify and Categorize Tables
    categorized_tables = {}
    all_tables_dict = {}

    for idx, table_df in enumerate(all_parsed_tables):
        table_type = identify_financial_statement(table_df)

        # Store in a dictionary with a unique key
        key_base = table_type
        counter = 1
        unique_key = key_base
        while unique_key in categorized_tables:
            unique_key = f"{key_base}_{counter}"
            counter += 1

        categorized_tables[unique_key] = table_df

        # Also store all tables with a generic key for backup/debugging
        all_tables_dict[f"Table_{idx}"] = table_df
    # Note: This returns categorized tables based on keyword matching.
    # The accuracy depends on the keywords and the structure of the tables in the 10-Q.
    # It might not perfectly identify tables like "Segment Performance" or "Products and Services"
    # which are specific to certain companies.
    return categorized_tables, all_tables_dict


if __name__ == "__main__":
    import asyncio

    asyncio.run(get_financial_report("INTC"))
