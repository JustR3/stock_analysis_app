import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

def get_stock_data(symbol: str, period: str = "1y") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fetch stock data and basic information.
    
    Args:
        symbol: Stock symbol
        period: Time period for historical data
        
    Returns:
        Tuple of (historical data DataFrame, stock info dictionary)
    """
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        info = stock.info
        return hist, info
    except Exception as e:
        raise Exception(f"Error fetching data for {symbol}: {str(e)}")

def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate various return metrics.
    
    Args:
        data: DataFrame with 'Close' prices
        
    Returns:
        DataFrame with return calculations
    """
    returns = pd.DataFrame()
    returns['Daily_Return'] = data['Close'].pct_change()
    returns['Cumulative_Return'] = (1 + returns['Daily_Return']).cumprod() - 1
    returns['Rolling_30d_Return'] = returns['Daily_Return'].rolling(window=30).mean()
    return returns

def calculate_volatility(data: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Calculate rolling volatility.
    
    Args:
        data: DataFrame with 'Close' prices
        window: Rolling window size
        
    Returns:
        Series with volatility calculations
    """
    returns = data['Close'].pct_change()
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized volatility 