import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Stock Analysis App",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Stock Analysis Dashboard")
st.markdown("""
    This dashboard provides comprehensive stock analysis tools and visualizations.
    Select a stock symbol to begin your analysis.
""")

# Sidebar
st.sidebar.header("Stock Selection")
symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL").upper()

# Date range selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        datetime.now() - timedelta(days=365)
    )
with col2:
    end_date = st.date_input(
        "End Date",
        datetime.now()
    )

# Main content
if symbol:
    try:
        # Fetch stock data
        stock = yf.Ticker(symbol)
        hist = stock.history(start=start_date, end=end_date)
        
        if not hist.empty:
            # Display basic info
            st.subheader(f"Basic Information for {symbol}")
            info = stock.info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
            with col2:
                st.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,.0f}")
            with col3:
                st.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
            
            # Price chart
            st.subheader("Price History")
            st.line_chart(hist['Close'])
            
            # Volume chart
            st.subheader("Trading Volume")
            st.bar_chart(hist['Volume'])
            
        else:
            st.error(f"No data found for symbol {symbol}")
            
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
else:
    st.info("Please enter a stock symbol to begin analysis") 