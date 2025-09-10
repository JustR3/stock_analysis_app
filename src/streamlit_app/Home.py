import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from .utils.config_manager import ConfigManager
from .utils.data_service import StreamlitDataService

# Import the visualizer
from single_stock_analysis.visualizer import StockVisualizer

# Initialize configuration, data service, and visualizer
config = ConfigManager()
data_service = StreamlitDataService()
visualizer = StockVisualizer()

# Page config
st.set_page_config(
    page_title=config.get("app.name"),
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title(config.get("app.name"))
st.markdown(config.get("app.description"))

# Sidebar
st.sidebar.header("Stock Selection")
symbol = st.sidebar.text_input("Enter Stock Symbol").upper()

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
        # Fetch stock data using the data service
        data, info = data_service.get_stock_data(
            symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if not data.empty:
            # Display basic info
            st.subheader(f"Basic Information for {symbol}")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
            with col2:
                st.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,.0f}")
            with col3:
                st.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
            
            # Price chart with moving averages
            st.subheader("Price History with Moving Averages")
            price_fig = visualizer.plot_price_and_moving_averages(
                data, symbol, ma_windows=[20, 50, 200], save=False
            )
            if price_fig:
                st.pyplot(price_fig)

            # Volume chart
            st.subheader("Trading Volume")
            volume_fig = visualizer.plot_volume_chart(data, symbol, save=False)
            if volume_fig:
                st.pyplot(volume_fig)
            
            # Additional metrics
            st.subheader("Performance Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Daily Return", f"{data['daily_return'].iloc[-1]:.2%}")
            with col2:
                st.metric("Volatility", f"{data['volatility'].iloc[-1]:.2%}")
            with col3:
                st.metric("20-day MA", f"${data['MA_20'].iloc[-1]:.2f}")
            
        else:
            st.error(f"No data found for symbol {symbol}")
            
    except Exception as e:
        error_msg = str(e).lower()

        # Provide specific error messages for different scenarios
        if "rate limit" in error_msg or "too many requests" in error_msg:
            st.error("🔄 **Rate Limit Exceeded**")
            st.warning("""
            Yahoo Finance API rate limit has been reached. This usually happens when:
            - Too many requests are made in a short time
            - Multiple users are accessing the service simultaneously

            **What you can do:**
            - Wait 5-10 minutes before trying again
            - Try a different stock symbol
            - Use cached data if available (try refreshing the page)

            The application will automatically retry with exponential backoff, but if the limit persists,
            please wait before making additional requests.
            """)
        elif "no data found" in error_msg:
            st.warning(f"📊 **No Data Found** for symbol '{symbol}'")
            st.info("""
            This could mean:
            - The stock symbol is incorrect or doesn't exist
            - The stock is delisted or not available
            - The market is closed

            Please verify the stock symbol and try again.
            """)
        elif "network" in error_msg or "connection" in error_msg:
            st.error("🌐 **Network Error**")
            st.warning("""
            Unable to connect to Yahoo Finance. Please check:
            - Your internet connection
            - Try again in a few minutes
            - The service might be temporarily unavailable
            """)
        else:
            st.error(f"❌ **Error**: {str(e)}")
            st.info("If this error persists, please try again later or contact support.")
else:
    st.info("Please enter a stock symbol to begin analysis") 