import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .data_collector import StockDataCollector
from .stock_analyzer import StockAnalyzer

logger = logging.getLogger(__name__)


class StockDataManager:
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize the stock data manager.

        Args:
            cache_dir (str): Directory to store cached data
        """
        self.cache_dir = cache_dir
        self.collector = StockDataCollector()
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, ticker: str) -> str:
        """Get the cache file path for a ticker"""
        return os.path.join(self.cache_dir, f"{ticker}_data.parquet")

    def _get_metadata_path(self, ticker: str) -> str:
        """Get the metadata file path for a ticker"""
        return os.path.join(self.cache_dir, f"{ticker}_metadata.json")

    def _get_stock_info_path(self, ticker: str) -> str:
        """Get the stock info cache file path for a ticker"""
        return os.path.join(self.cache_dir, f"{ticker}_info.json")

    def _save_metadata(self, ticker: str, metadata: Dict):
        """Save metadata for a ticker"""
        with open(self._get_metadata_path(ticker), "w") as f:
            json.dump(metadata, f)

    def _load_metadata(self, ticker: str) -> Optional[Dict]:
        """Load metadata for a ticker"""
        try:
            with open(self._get_metadata_path(ticker), "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def _save_stock_info(self, ticker: str, info: Dict):
        """Save stock info for a ticker"""
        with open(self._get_stock_info_path(ticker), "w") as f:
            json.dump(info, f)

    def _load_stock_info(self, ticker: str) -> Optional[Dict]:
        """Load cached stock info for a ticker"""
        try:
            with open(self._get_stock_info_path(ticker), "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def _is_stock_info_cache_valid(self, ticker: str) -> bool:
        """Check if cached stock info is valid (smart caching based on market hours)"""
        try:
            cache_file = self._get_stock_info_path(ticker)
            if not os.path.exists(cache_file):
                return False

            # Check file modification time
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            now = datetime.now()

            # Check if it's currently market hours
            is_market_hours = (
                now.weekday() < 5
                and (  # Monday-Friday
                    now.hour > 9 or (now.hour == 9 and now.minute >= 30)
                )
                and (now.hour < 16)
            )

            if is_market_hours:
                # During market hours, use 6-hour cache for stock info
                cache_duration = timedelta(hours=6)
            else:
                # Outside market hours, use 24-hour cache
                cache_duration = timedelta(days=1)

            return (now - file_mod_time) < cache_duration
        except Exception:
            return False

    def _is_cache_valid(self, ticker: str) -> bool:
        """Check if cached data is valid (less than 1 day old, but during market hours use shorter cache)"""
        metadata = self._load_metadata(ticker)
        if not metadata:
            return False

        last_update = datetime.fromisoformat(metadata["last_update"])
        now = datetime.now()

        # Check if it's currently market hours (9:30 AM - 4:00 PM ET, Monday-Friday)
        # Note: This is a simple check and doesn't account for holidays
        is_market_hours = (
            now.weekday() < 5
            and (now.hour > 9 or (now.hour == 9 and now.minute >= 30))  # Monday-Friday
            and (now.hour < 16)
        )

        if is_market_hours:
            # During market hours, use 4-hour cache
            cache_duration = timedelta(hours=4)
        else:
            # Outside market hours, use 24-hour cache
            cache_duration = timedelta(days=1)

        return (now - last_update) < cache_duration

    def _enrich_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add analysis parameters to the dataframe"""
        analyzer = StockAnalyzer(data)

        # Calculate returns
        data["daily_return"] = analyzer.calculate_returns(period="daily")
        data["monthly_return"] = analyzer.calculate_returns(period="monthly")

        # Calculate volatility
        data["volatility"] = analyzer.calculate_volatility()

        # Calculate moving averages
        ma_dict = analyzer.calculate_moving_averages()
        for ma_name, ma_series in ma_dict.items():
            data[ma_name] = ma_series

        return data

    def get_stock_data(self, ticker: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get stock data, either from cache or by fetching new data.

        Args:
            ticker (str): Stock ticker symbol
            force_refresh (bool): Force refresh of data even if cache is valid

        Returns:
            pd.DataFrame: Stock data with analysis parameters
        """
        cache_path = self._get_cache_path(ticker)

        # Check if we can use cached data
        if (
            not force_refresh
            and os.path.exists(cache_path)
            and self._is_cache_valid(ticker)
        ):
            logger.info(f"Using cached data for {ticker} from {cache_path}")
            logger.info("Cache is valid and less than 1 day old")
            return pd.read_parquet(cache_path)

        # Fetch new data
        logger.info(f"Cache not found or invalid for {ticker}. Fetching new data...")
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

        data = self.collector.get_stock_data(ticker, start_date, end_date)
        if data.empty:
            raise ValueError(f"No data found for {ticker}")

        # Enrich data with analysis parameters
        logger.info("Enriching data with analysis parameters...")
        data = self._enrich_data(data)

        # Save to cache
        logger.info(f"Saving data to cache at {cache_path}")
        data.to_parquet(cache_path)
        self._save_metadata(
            ticker,
            {
                "last_update": datetime.now().isoformat(),
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
            },
        )

        return data

    def get_stock_info(self, ticker: str, force_refresh: bool = False) -> Dict:
        """
        Get stock info, either from cache or by fetching new info.

        Args:
            ticker (str): Stock ticker symbol
            force_refresh (bool): Force refresh of info even if cache is valid

        Returns:
            Dict: Stock information
        """
        # Check if we can use cached info
        if not force_refresh and self._is_stock_info_cache_valid(ticker):
            cached_info = self._load_stock_info(ticker)
            if cached_info:
                logger.info(f"Using cached stock info for {ticker}")
                return cached_info

        # Fetch new info
        logger.info(f"Fetching new stock info for {ticker}")
        info = self.collector.get_stock_info(ticker)

        if info:
            # Save to cache
            logger.info(f"Saving stock info to cache for {ticker}")
            self._save_stock_info(ticker, info)

        return info if info else {}
