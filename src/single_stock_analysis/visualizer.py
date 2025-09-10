import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional
import os
import logging

logger = logging.getLogger(__name__)

class StockVisualizer:
    def __init__(self, output_dir: str = "data/plots"):
        """
        Initialize the stock visualizer.
        
        Args:
            output_dir (str): Directory to store generated plots
        """
        self.output_dir = output_dir
        self._ensure_output_dir()
        
        # Set seaborn style
        sns.set_theme(style="darkgrid")
        sns.set_context("notebook", font_scale=1.2)
        
    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _get_plot_path(self, ticker: str, plot_type: str) -> str:
        """Get the plot file path"""
        return os.path.join(self.output_dir, f"{ticker}_{plot_type}.png")
        
    def plot_price_and_moving_averages(
        self,
        data: pd.DataFrame,
        ticker: str,
        ma_windows: List[int] = [20, 50, 200],
        save: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot stock price and moving averages.
        
        Args:
            data (pd.DataFrame): Stock data
            ticker (str): Stock ticker symbol
            ma_windows (List[int]): List of moving average windows
            save (bool): Whether to save the plot
            
        Returns:
            Optional[plt.Figure]: The plot figure if save is False
        """
        plt.figure(figsize=(15, 8))
        
        # Calculate statistics
        avg_daily_return = data['daily_return'].mean()
        avg_volatility = data['volatility'].mean()
        max_drawdown = (data['Close'] / data['Close'].cummax() - 1).min()
        
        # Plot closing price
        sns.lineplot(data=data, x=data.index, y='Close', label='Price', color='black', alpha=0.7)
        
        # Plot moving averages
        colors = ['blue', 'green', 'red']
        for window, color in zip(ma_windows, colors):
            ma_col = f'MA_{window}'
            if ma_col in data.columns:
                sns.lineplot(
                    data=data,
                    x=data.index,
                    y=ma_col,
                    label=f'{window}-day MA',
                    color=color,
                    alpha=0.7
                )
        
        # Add statistics to legend
        plt.plot([], [], ' ', label=f'Avg Daily Return: {avg_daily_return:.2%}')
        plt.plot([], [], ' ', label=f'Avg Volatility: {avg_volatility:.2%}')
        plt.plot([], [], ' ', label=f'Max Drawdown: {max_drawdown:.2%}')
        
        # Customize plot
        plt.title(f'{ticker} Stock Price and Moving Averages', fontsize=16, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to accommodate the legend
        plt.tight_layout()
        
        if save:
            plot_path = self._get_plot_path(ticker, 'price_ma')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Plot saved to {plot_path}")
            return None
        else:
            return plt.gcf()
            
    def plot_technical_indicators(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ticker: str,
        save: bool = False
    ) -> Optional[plt.Figure]:
        """
        Plot price chart with Bollinger Bands.

        Args:
            data (pd.DataFrame): Stock data
            indicators (pd.DataFrame): Technical indicators data
            ticker (str): Stock ticker symbol
            save (bool): Whether to save the plot

        Returns:
            Optional[plt.Figure]: The plot figure if save is False
        """
        plt.figure(figsize=(14, 8))

        # Plot price and Bollinger Bands
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['Close'], label='Close Price', color='blue', linewidth=2)

        if 'BB_Upper' in indicators.columns and 'BB_Middle' in indicators.columns and 'BB_Lower' in indicators.columns:
            plt.plot(data.index, indicators['BB_Upper'], label='Upper BB', color='red', linestyle='--', alpha=0.7)
            plt.plot(data.index, indicators['BB_Middle'], label='Middle BB', color='orange', linestyle='-', alpha=0.7)
            plt.plot(data.index, indicators['BB_Lower'], label='Lower BB', color='green', linestyle='--', alpha=0.7)

            # Fill between bands
            plt.fill_between(data.index, indicators['BB_Upper'], indicators['BB_Lower'],
                           alpha=0.1, color='gray', label='BB Range')

        plt.title(f'{ticker} Price and Bollinger Bands', fontsize=16, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(fontsize=10, loc='upper left')
        plt.xticks(rotation=45)

        # Plot RSI
        plt.subplot(2, 2, 3)
        if 'RSI' in indicators.columns:
            plt.plot(data.index, indicators['RSI'], label='RSI', color='purple', linewidth=1.5)

            # Add RSI levels
            plt.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
            plt.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
            plt.axhline(y=50, color='gray', linestyle='-', alpha=0.3, label='Neutral (50)')

            plt.ylim(0, 100)
            plt.title('RSI (Relative Strength Index)', fontsize=12)
            plt.xlabel('Date', fontsize=10)
            plt.ylabel('RSI', fontsize=10)
            plt.xticks(rotation=45)

        # Plot MACD
        plt.subplot(2, 2, 4)
        if 'MACD' in indicators.columns and 'MACD_Signal' in indicators.columns:
            plt.plot(data.index, indicators['MACD'], label='MACD', color='blue', linewidth=1.5)
            plt.plot(data.index, indicators['MACD_Signal'], label='Signal', color='red', linewidth=1.5)

            # Add zero line
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            plt.title('MACD (Moving Average Convergence Divergence)', fontsize=12)
            plt.xlabel('Date', fontsize=10)
            plt.ylabel('MACD', fontsize=10)
            plt.legend(fontsize=8)
            plt.xticks(rotation=45)

        plt.tight_layout()

        if save:
            plot_path = self._get_plot_path(ticker, 'technical_analysis')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Technical analysis plot saved to {plot_path}")
            return None
        else:
            return plt.gcf()

    def plot_volume_chart(
        self,
        data: pd.DataFrame,
        ticker: str,
        save: bool = False
    ) -> Optional[plt.Figure]:
        """
        Plot trading volume chart.

        Args:
            data (pd.DataFrame): Stock data
            ticker (str): Stock ticker symbol
            save (bool): Whether to save the plot

        Returns:
            Optional[plt.Figure]: The plot figure if save is False
        """
        plt.figure(figsize=(14, 6))

        # Create volume bars with color based on price movement
        colors = ['green' if close > open else 'red'
                 for close, open in zip(data['Close'], data['Open'])]

        plt.bar(data.index, data['Volume'], color=colors, alpha=0.7, width=1)

        plt.title(f'{ticker} Trading Volume', fontsize=16, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Volume', fontsize=12)
        plt.xticks(rotation=45)

        # Format y-axis labels
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))

        plt.tight_layout()

        if save:
            plot_path = self._get_plot_path(ticker, 'volume')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Volume plot saved to {plot_path}")
            return None
        else:
            return plt.gcf()

    def plot_returns_distribution(
        self,
        data: pd.DataFrame,
        ticker: str,
        period: str = 'daily',
        save: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot returns distribution.

        Args:
            data (pd.DataFrame): Stock data
            ticker (str): Stock ticker symbol
            period (str): 'daily', 'weekly', or 'monthly'
            save (bool): Whether to save the plot

        Returns:
            Optional[plt.Figure]: The plot figure if save is False
        """
        plt.figure(figsize=(12, 6))
        
        # Get returns column
        returns_col = f'{period}_return'
        if returns_col not in data.columns:
            raise ValueError(f"Returns column {returns_col} not found in data")
            
        # Plot histogram with KDE
        sns.histplot(
            data=data,
            x=returns_col,
            kde=True,
            stat='density',
            color='blue',
            alpha=0.6
        )
        
        # Add normal distribution curve
        from scipy import stats
        returns = data[returns_col].dropna()
        mu, std = returns.mean(), returns.std()
        x = np.linspace(mu - 4*std, mu + 4*std, 100)
        p = stats.norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
        
        # Customize plot
        plt.title(f'{ticker} {period.capitalize()} Returns Distribution', fontsize=16, pad=20)
        plt.xlabel('Returns', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(fontsize=10)
        
        # Add statistics text
        stats_text = f'Mean: {mu:.2%}\nStd: {std:.2%}'
        plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        if save:
            plot_path = self._get_plot_path(ticker, f'returns_{period}')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Plot saved to {plot_path}")
            return None
        else:
            return plt.gcf() 