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
    def __init__(self, rate_limit_delay: int = 5, max_retries: int = 3):
        """
        Initialize the stock data collector.

        Args:
            rate_limit_delay (int): Base delay between API calls in seconds
            max_retries (int): Maximum number of retry attempts for rate-limited requests
        """
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.last_request_time = 0

    def _respect_rate_limit(self):
        """Ensure we don't exceed yfinance rate limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last_request)
        self.last_request_time = time.time()

    def _handle_rate_limit_error(self, ticker: str, attempt: int) -> bool:
        """
        Handle rate limit errors with exponential backoff.

        Args:
            ticker: Stock ticker symbol
            attempt: Current attempt number

        Returns:
            bool: True if should retry, False otherwise
        """
        if attempt < self.max_retries:
            # Exponential backoff: 5, 10, 20 seconds
            backoff_time = self.rate_limit_delay * (2 ** attempt)
            logger.warning(f"Rate limited for {ticker}. Retrying in {backoff_time} seconds (attempt {attempt + 1}/{self.max_retries})")
            time.sleep(backoff_time)
            return True
        else:
            logger.error(f"Rate limit exceeded for {ticker} after {self.max_retries} attempts")
            return False

    def get_stock_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch stock data for a given ticker with retry mechanism.

        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            interval (str): Data interval (1d, 1wk, 1mo, etc.)

        Returns:
            pd.DataFrame: Stock data
        """
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")

        for attempt in range(self.max_retries + 1):
            try:
                self._respect_rate_limit()

                stock = yf.Ticker(ticker)
                data = stock.history(start=start_date, end=end_date, interval=interval)

                if data.empty:
                    logger.warning(f"No data found for {ticker}")
                    return pd.DataFrame()

                logger.info(f"Successfully fetched data for {ticker}")
                return data

            except Exception as e:
                error_msg = str(e).lower()
                if "too many requests" in error_msg or "rate limit" in error_msg:
                    if self._handle_rate_limit_error(ticker, attempt):
                        continue
                    else:
                        # Max retries exceeded
                        raise Exception(f"Rate limit exceeded for {ticker} after {self.max_retries} attempts")
                else:
                    # Non-rate-limit error, don't retry
                    logger.error(f"Error fetching data for {ticker}: {str(e)}")
                    raise Exception(f"Error fetching data for {ticker}: {str(e)}")

        # This should never be reached, but just in case
        return pd.DataFrame()

    def get_stock_info(self, ticker: str) -> Dict:
        """
        Fetch basic stock information with retry mechanism.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Dict: Stock information
        """
        logger.info(f"Fetching stock info for {ticker}")

        for attempt in range(self.max_retries + 1):
            try:
                self._respect_rate_limit()

                stock = yf.Ticker(ticker)
                info = stock.info

                logger.info(f"Successfully fetched stock info for {ticker}")
                return info

            except Exception as e:
                error_msg = str(e).lower()
                if "too many requests" in error_msg or "rate limit" in error_msg:
                    if self._handle_rate_limit_error(ticker, attempt):
                        continue
                    else:
                        # Max retries exceeded
                        logger.error(f"Failed to fetch stock info for {ticker} after {self.max_retries} attempts")
                        return {}
                else:
                    # Non-rate-limit error, don't retry
                    logger.error(f"Error fetching info for {ticker}: {str(e)}")
                    return {}

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