import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class StockAnalyzer:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the stock analyzer with historical data.
        
        Args:
            data (pd.DataFrame): Historical stock data
        """
        self.data = data
        self._validate_data()
    
    def _validate_data(self):
        """Validate that the data has required columns"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def calculate_returns(self, period: str = 'daily') -> pd.Series:
        """
        Calculate returns for the specified period.
        
        Args:
            period (str): 'daily', 'weekly', or 'monthly'
            
        Returns:
            pd.Series: Returns series
        """
        if period == 'daily':
            return self.data['Close'].pct_change()
        elif period == 'weekly':
            return self.data['Close'].resample('W').last().pct_change()
        elif period == 'monthly':
            return self.data['Close'].resample('M').last().pct_change()
        else:
            raise ValueError("Period must be 'daily', 'weekly', or 'monthly'")
    
    def calculate_volatility(self, window: int = 252) -> pd.Series:
        """
        Calculate rolling volatility.
        
        Args:
            window (int): Rolling window size in days
            
        Returns:
            pd.Series: Volatility series
        """
        returns = self.calculate_returns()
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    def calculate_moving_averages(self, windows: List[int] = [20, 50, 200]) -> Dict[str, pd.Series]:
        """
        Calculate moving averages for specified windows.
        
        Args:
            windows (List[int]): List of window sizes
            
        Returns:
            Dict[str, pd.Series]: Dictionary of moving averages
        """
        return {f'MA_{window}': self.data['Close'].rolling(window=window).mean() 
                for window in windows}
    
    def get_summary_statistics(self) -> Dict:
        """
        Calculate summary statistics for the stock.
        
        Returns:
            Dict: Dictionary of summary statistics
        """
        returns = self.calculate_returns()
        return {
            'mean_return': returns.mean(),
            'std_return': returns.std(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': (self.data['Close'] / self.data['Close'].cummax() - 1).min(),
            'volatility': self.calculate_volatility().iloc[-1],
            'current_price': self.data['Close'].iloc[-1],
            'price_change_1d': self.data['Close'].pct_change().iloc[-1],
            'volume_avg': self.data['Volume'].mean()
        } 