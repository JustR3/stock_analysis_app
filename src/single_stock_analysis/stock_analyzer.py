import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats as stats

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
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [
            col for col in required_columns if col not in self.data.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def calculate_returns(self, period: str = "daily") -> pd.Series:
        """
        Calculate returns for the specified period.

        Args:
            period (str): 'daily' or 'monthly'

        Returns:
            pd.Series: Returns series
        """
        # Ensure index is datetime
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)

        if period == "daily":
            return self.data["Close"].pct_change()
        elif period == "monthly":
            # Resample to monthly frequency and calculate returns
            monthly_data = self.data["Close"].resample("ME").last()
            return monthly_data.pct_change()
        else:
            raise ValueError("Period must be 'daily' or 'monthly'")

    def calculate_volatility(self, window: int = 252) -> pd.Series:
        """
        Calculate rolling volatility.

        Args:
            window (int): Rolling window size in days

        Returns:
            pd.Series: Volatility series
        """
        returns = self.calculate_returns(period="daily")
        return returns.rolling(window=window).std() * np.sqrt(252)

    def calculate_moving_averages(
        self, windows: List[int] = [20, 50, 200]
    ) -> Dict[str, pd.Series]:
        """
        Calculate moving averages for specified windows.

        Args:
            windows (List[int]): List of window sizes

        Returns:
            Dict[str, pd.Series]: Dictionary of moving averages
        """
        return {
            f"MA_{window}": self.data["Close"].rolling(window=window).mean()
            for window in windows
        }

    def get_summary_statistics(self) -> Dict:
        """
        Calculate summary statistics for the stock.

        Returns:
            Dict: Dictionary of summary statistics
        """
        returns = self.calculate_returns(period="daily")
        return {
            "mean_return": returns.mean(),
            "std_return": returns.std(),
            "sharpe_ratio": returns.mean() / returns.std() * np.sqrt(252),
            "max_drawdown": (
                self.data["Close"] / self.data["Close"].cummax() - 1
            ).min(),
            "volatility": self.calculate_volatility().iloc[-1],
            "current_price": self.data["Close"].iloc[-1],
            "price_change_1d": self.data["Close"].pct_change().iloc[-1],
            "volume_avg": self.data["Volume"].mean(),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
        }

    def get_returns_confidence_intervals(
        self, period: str = "daily", confidence: float = 0.95
    ) -> Dict:
        """
        Calculate confidence intervals for returns.

        Args:
            period (str): 'daily' or 'monthly'
            confidence (float): Confidence level (0-1)

        Returns:
            Dict: Dictionary containing confidence intervals
        """
        returns = self.calculate_returns(period)
        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf((1 + confidence) / 2)

        return {
            "lower_bound": mean - z_score * std,
            "upper_bound": mean + z_score * std,
            "confidence_level": confidence,
        }

    def calculate_var(
        self, confidence_level: float = 0.95, method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            confidence_level (float): Confidence level (0-1)
            method (str): 'historical' or 'parametric'

        Returns:
            float: Value at Risk
        """
        returns = self.calculate_returns(period="daily")
        if method == "historical":
            return np.percentile(returns, (1 - confidence_level) * 100)
        elif method == "parametric":
            mean = returns.mean()
            std = returns.std()
            return mean + stats.norm.ppf(1 - confidence_level) * std
        else:
            raise ValueError("Method must be 'historical' or 'parametric'")

    def calculate_cvar(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).

        Args:
            confidence_level (float): Confidence level (0-1)

        Returns:
            float: Conditional Value at Risk
        """
        returns = self.calculate_returns(period="daily")
        var = self.calculate_var(confidence_level)
        return returns[returns <= var].mean()

    def calculate_beta(self, market_returns: pd.Series) -> float:
        """
        Calculate beta coefficient relative to market returns.

        Args:
            market_returns (pd.Series): Market returns series

        Returns:
            float: Beta coefficient
        """
        stock_returns = self.calculate_returns(period="daily")
        covariance = np.cov(stock_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        return covariance / market_variance

    def calculate_information_ratio(self, benchmark_returns: pd.Series) -> float:
        """
        Calculate Information Ratio.

        Args:
            benchmark_returns (pd.Series): Benchmark returns series

        Returns:
            float: Information Ratio
        """
        excess_returns = self.calculate_returns(period="daily") - benchmark_returns
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def get_advanced_statistics(
        self, market_returns: Optional[pd.Series] = None
    ) -> Dict:
        """
        Calculate advanced statistical measures.

        Args:
            market_returns (Optional[pd.Series]): Market returns for beta calculation

        Returns:
            Dict: Dictionary of advanced statistics
        """
        stats_dict = {
            "var_95": self.calculate_var(0.95),
            "cvar_95": self.calculate_cvar(0.95),
            "var_99": self.calculate_var(0.99),
            "cvar_99": self.calculate_cvar(0.99),
            "sortino_ratio": self._calculate_sortino_ratio(),
            "calmar_ratio": self._calculate_calmar_ratio(),
            "omega_ratio": self._calculate_omega_ratio(),
        }

        if market_returns is not None:
            stats_dict.update(
                {
                    "beta": self.calculate_beta(market_returns),
                    "information_ratio": self.calculate_information_ratio(
                        market_returns
                    ),
                }
            )

        return stats_dict

    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino Ratio."""
        returns = self.calculate_returns(period="daily")
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns**2))
        return np.sqrt(252) * excess_returns.mean() / downside_std

    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar Ratio."""
        returns = self.calculate_returns(period="daily")
        max_drawdown = (self.data["Close"] / self.data["Close"].cummax() - 1).min()
        return np.sqrt(252) * returns.mean() / abs(max_drawdown)

    def _calculate_omega_ratio(self, threshold: float = 0.0) -> float:
        """Calculate Omega Ratio."""
        returns = self.calculate_returns()
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        return gains / losses if losses != 0 else float("inf")
