import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockDataCollector:
    def __init__(self, rate_limit_delay: int = 2):
        """
        Initialize the stock data collector.
        
        Args:
            rate_limit_delay (int): Delay between API calls in seconds
        """
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0

    def _respect_rate_limit(self):
        """Ensure we don't exceed yfinance rate limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last_request)
        self.last_request_time = time.time()

    def get_stock_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch stock data for a given ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            interval (str): Data interval (1d, 1wk, 1mo, etc.)
            
        Returns:
            pd.DataFrame: Stock data
        """
        try:
            self._respect_rate_limit()
            
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
            
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
                
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def get_stock_info(self, ticker: str) -> Dict:
        """
        Fetch basic stock information.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Dict: Stock information
        """
        try:
            self._respect_rate_limit()
            stock = yf.Ticker(ticker)
            info = stock.info
            return info
        except Exception as e:
            logger.error(f"Error fetching info for {ticker}: {str(e)}")
            return {}

    def get_multiple_stocks(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks.
        
        Args:
            tickers (List[str]): List of stock ticker symbols
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            interval (str): Data interval
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of stock dataframes
        """
        results = {}
        for ticker in tickers:
            data = self.get_stock_data(ticker, start_date, end_date, interval)
            if not data.empty:
                results[ticker] = data
        return results 