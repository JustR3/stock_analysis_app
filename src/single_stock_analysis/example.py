from data_collector import StockDataCollector
from stock_analyzer import StockAnalyzer
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_stock(ticker: str, start_date: str = None, end_date: str = None):
    """
    Analyze a stock using the data collector and analyzer.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    """
    # Collect data
    collector = StockDataCollector()
    data = collector.get_stock_data(ticker, start_date, end_date)
    
    if data.empty:
        logger.error(f"No data found for {ticker}")
        return
    
    # Analyze data
    analyzer = StockAnalyzer(data)
    
    # Get summary statistics
    stats = analyzer.get_summary_statistics()
    logger.info(f"\nSummary Statistics for {ticker}:")
    for key, value in stats.items():
        logger.info(f"{key}: {value:.4f}")
    
    # Calculate moving averages
    ma_dict = analyzer.calculate_moving_averages()
    logger.info("\nMoving Averages:")
    for ma_name, ma_series in ma_dict.items():
        logger.info(f"{ma_name}: {ma_series.iloc[-1]:.2f}")
    
    # Calculate volatility
    volatility = analyzer.calculate_volatility()
    logger.info(f"\nCurrent Volatility: {volatility.iloc[-1]:.4f}")

if __name__ == "__main__":
    # Example usage
    analyze_stock("AAPL", start_date="2023-01-01") 