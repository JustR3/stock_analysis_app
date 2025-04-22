import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from utils.config_manager import ConfigManager
from utils.data_service import StreamlitDataService

# Initialize configuration and data service
config = ConfigManager()
data_service = StreamlitDataService()

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
default_symbol = config.get("data.default_symbol", "AAPL")
symbol = st.sidebar.text_input("Enter Stock Symbol", default_symbol).upper()

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
            
            # Price chart with configured height
            st.subheader("Price History")
            st.line_chart(
                data['Close'],
                height=config.get("visualization.chart_height", 400)
            )
            
            # Volume chart
            st.subheader("Trading Volume")
            st.bar_chart(
                data['Volume'],
                height=config.get("visualization.chart_height", 400)
            )
            
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
        st.error(f"Error fetching data: {str(e)}")
else:
    st.info("Please enter a stock symbol to begin analysis") 