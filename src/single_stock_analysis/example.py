import logging

import pandas as pd
from data_manager import StockDataManager
from visualizer import StockVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_stock(ticker: str, force_refresh: bool = False):
    """
    Analyze a stock using the data manager and visualizer.

    Args:
        ticker (str): Stock ticker symbol
        force_refresh (bool): Force refresh of data even if cache is valid
    """
    # Initialize managers
    data_manager = StockDataManager()
    visualizer = StockVisualizer()

    try:
        # Get data (from cache or by fetching)
        data = data_manager.get_stock_data(ticker, force_refresh)

        # Display some basic information
        logger.info(f"\nData Summary for {ticker}:")
        logger.info(f"Date Range: {data.index[0]} to {data.index[-1]}")
        logger.info(f"Number of observations: {len(data)}")

        # Display latest values
        latest = data.iloc[-1]
        logger.info("\nLatest Values:")
        logger.info(f"Price: ${latest['Close']:.2f}")
        logger.info(f"Daily Return: {latest['daily_return']:.2%}")
        logger.info(f"Volatility: {latest['volatility']:.2%}")
        logger.info(f"MA_20: ${latest['MA_20']:.2f}")
        logger.info(f"MA_50: ${latest['MA_50']:.2f}")
        logger.info(f"MA_200: ${latest['MA_200']:.2f}")

        # Display some statistics
        logger.info("\nStatistics:")
        logger.info(f"Average Daily Return: {data['daily_return'].mean():.2%}")
        logger.info(f"Average Volatility: {data['volatility'].mean():.2%}")
        logger.info(
            f"Max Drawdown: {(data['Close'] / data['Close'].cummax() - 1).min():.2%}"
        )

        # Generate plots
        logger.info("\nGenerating plots...")
        visualizer.plot_price_and_moving_averages(data, ticker)
        visualizer.plot_returns_distribution(data, ticker, period="daily")
        visualizer.plot_returns_distribution(data, ticker, period="weekly")
        visualizer.plot_returns_distribution(data, ticker, period="monthly")

    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {str(e)}")


if __name__ == "__main__":
    # Example usage
    analyze_stock("AAPL")
