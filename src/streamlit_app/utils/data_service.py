import sys
from pathlib import Path
import pandas as pd
from typing import Tuple, Dict, Any

# Add the src directory to the Python path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from single_stock_analysis.data_manager import StockDataManager
from .config_manager import ConfigManager

class StreamlitDataService:
    def __init__(self):
        """Initialize the Streamlit data service."""
        self.config = ConfigManager()

        # Get API configuration for rate limiting
        rate_limit_delay = self.config.get("api.yfinance.rate_limit_delay", 5)
        max_retries = self.config.get("api.yfinance.max_retries", 3)

        # Initialize data manager with configured collector
        from single_stock_analysis.data_collector import StockDataCollector
        collector = StockDataCollector(
            rate_limit_delay=rate_limit_delay,
            max_retries=max_retries
        )

        # Create data manager with the configured collector
        self.data_manager = StockDataManager(
            cache_dir=self.config.get("data.cache_dir", "data/cache")
        )
        self.data_manager.collector = collector
        
    def get_stock_data(self, symbol: str, start_date: str = None, end_date: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Get stock data with caching.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            Tuple of (stock data DataFrame, stock info dictionary)
        """
        try:
            # Get data from cache or fetch new
            data = self.data_manager.get_stock_data(symbol)
            
            # Filter by date range if provided
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
                
            # Get stock info (with caching)
            stock = self.data_manager.get_stock_info(symbol)
            
            return data, stock
            
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
            
    def get_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the data.
        
        Args:
            data: Stock data DataFrame
            
        Returns:
            DataFrame with technical indicators
        """
        # Get indicator parameters from config
        rsi_period = self.config.get("technical_analysis.indicators.rsi.period", 14)
        macd_fast = self.config.get("technical_analysis.indicators.macd.fast_period", 12)
        macd_slow = self.config.get("technical_analysis.indicators.macd.slow_period", 26)
        macd_signal = self.config.get("technical_analysis.indicators.macd.signal_period", 9)
        bb_period = self.config.get("technical_analysis.indicators.bollinger_bands.period", 20)
        bb_std = self.config.get("technical_analysis.indicators.bollinger_bands.std_dev", 2)
        
        # Calculate indicators
        indicators = pd.DataFrame(index=data.index)
        
        # RSI
        indicators['RSI'] = data['Close'].ewm(span=rsi_period).mean()
        
        # MACD
        exp1 = data['Close'].ewm(span=macd_fast).mean()
        exp2 = data['Close'].ewm(span=macd_slow).mean()
        indicators['MACD'] = exp1 - exp2
        indicators['MACD_Signal'] = indicators['MACD'].ewm(span=macd_signal).mean()
        
        # Bollinger Bands
        indicators['BB_Middle'] = data['Close'].rolling(window=bb_period).mean()
        indicators['BB_Upper'] = indicators['BB_Middle'] + (data['Close'].rolling(window=bb_period).std() * bb_std)
        indicators['BB_Lower'] = indicators['BB_Middle'] - (data['Close'].rolling(window=bb_period).std() * bb_std)
        
        return indicators 