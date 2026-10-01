import pandas as pd
import yfinance as yf
import pandas_datareader.data as web

def get_asset_returns(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches adjusted close prices and calculates daily percentage returns."""
    raw = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    # Handle yfinance multi-index format variations
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.levels[0]:
            prices = raw["Adj Close"]
        elif "Adj Close" in raw.columns.levels[1]:
            prices = raw.xs("Adj Close", axis=1, level=1)
        elif "Close" in raw.columns.levels[0]:
            prices = raw["Close"]
        else:
            prices = raw.xs("Close", axis=1, level=1)
    else:
        prices = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]

    returns = prices[tickers].pct_change().dropna(how="all")
    return returns

def get_fama_french_factors(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches Fama-French 5 Factors (Daily) from Kenneth French Data Library."""
    ds = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start_date, end_date)
    ff_daily = ds[0] / 100.0
    
    # Convert PeriodIndex to DatetimeIndex safely
    if isinstance(ff_daily.index, pd.PeriodIndex):
        ff_daily.index = ff_daily.index.to_timestamp()
    else:
        ff_daily.index = pd.to_datetime(ff_daily.index.astype(str))
        
    return ff_daily

if __name__ == "__main__":
    universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "XOM", "JNJ", "PG", "BRK-B"]
    start = "2021-01-01"
    end = "2026-09-01"
    
    asset_ret = get_asset_returns(universe, start, end)
    ff_factors = get_fama_french_factors(start, end)
    
    aligned = asset_ret.join(ff_factors, how="inner").dropna()
    aligned.to_csv("aligned_factor_returns.csv")
    print(f"Data cached: {aligned.shape[0]} trading days across {len(universe)} assets.")