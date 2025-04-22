from data_collector import StockDataCollector
import pandas as pd
from datetime import datetime, timedelta

def main():
    # Initialize the data collector
    collector = StockDataCollector(rate_limit_delay=2)
    
    # Example ticker
    ticker = "AAPL"
    
    # Get stock data for the last year
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Fetch data
    data = collector.get_stock_data(ticker, start_date, end_date)
    
    if not data.empty:
        print(f"\nData for {ticker}:")
        print(data.head())
        
        # Get basic stock information
        info = collector.get_stock_info(ticker)
        if info:
            print(f"\nBasic information for {ticker}:")
            print(f"Company Name: {info.get('longName', 'N/A')}")
            print(f"Current Price: ${info.get('currentPrice', 'N/A')}")
            print(f"Market Cap: ${info.get('marketCap', 'N/A'):,.2f}")
            print(f"52 Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}")
            print(f"52 Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}")
    else:
        print(f"No data found for {ticker}")

if __name__ == "__main__":
    main() 