import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Dict, Optional
from data_collector import StockDataCollector
from stock_analyzer import StockAnalyzer
import logging

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
        
    def _save_metadata(self, ticker: str, metadata: Dict):
        """Save metadata for a ticker"""
        with open(self._get_metadata_path(ticker), 'w') as f:
            json.dump(metadata, f)
            
    def _load_metadata(self, ticker: str) -> Optional[Dict]:
        """Load metadata for a ticker"""
        try:
            with open(self._get_metadata_path(ticker), 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
            
    def _is_cache_valid(self, ticker: str) -> bool:
        """Check if cached data is valid (less than 1 day old)"""
        metadata = self._load_metadata(ticker)
        if not metadata:
            return False
            
        last_update = datetime.fromisoformat(metadata['last_update'])
        return (datetime.now() - last_update) < timedelta(days=1)
        
    def _enrich_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add analysis parameters to the dataframe"""
        analyzer = StockAnalyzer(data)
        
        # Calculate returns
        data['daily_return'] = analyzer.calculate_returns()
        data['weekly_return'] = analyzer.calculate_returns(period='weekly')
        data['monthly_return'] = analyzer.calculate_returns(period='monthly')
        
        # Calculate volatility
        data['volatility'] = analyzer.calculate_volatility()
        
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
        if not force_refresh and os.path.exists(cache_path) and self._is_cache_valid(ticker):
            logger.info(f"Loading cached data for {ticker}")
            return pd.read_parquet(cache_path)
            
        # Fetch new data
        logger.info(f"Fetching new data for {ticker}")
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        data = self.collector.get_stock_data(ticker, start_date, end_date)
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
            
        # Enrich data with analysis parameters
        data = self._enrich_data(data)
        
        # Save to cache
        data.to_parquet(cache_path)
        self._save_metadata(ticker, {
            'last_update': datetime.now().isoformat(),
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date
        })
        
        return data 