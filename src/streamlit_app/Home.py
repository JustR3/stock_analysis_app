import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent  # Go up to src directory
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import the visualizer
from single_stock_analysis.visualizer import StockVisualizer

# Import utilities with absolute paths
from streamlit_app.utils.config_manager import ConfigManager
from streamlit_app.utils.data_service import StreamlitDataService

# Initialize configuration, data service, and visualizer
config = ConfigManager()
data_service = StreamlitDataService()
visualizer = StockVisualizer()

# Page config
st.set_page_config(page_title=config.get("app.name"), page_icon="📈", layout="wide")

# Title and description
st.title(config.get("app.name"))
st.markdown(config.get("app.description"))

# Sidebar
st.sidebar.header("Stock Selection")
symbol = st.sidebar.text_input("Enter Stock Symbol").upper()

# Date range selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
with col2:
    end_date = st.date_input("End Date", datetime.now())

# Main content
if symbol:
    try:
        # Fetch stock data using the data service
        data, info = data_service.get_stock_data(
            symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
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

            # Technical Indicators (RSI)
            st.subheader("Technical Indicators")
            try:
                # Calculate technical indicators
                indicators = data_service.get_technical_indicators(data)

                # RSI Chart
                if "RSI" in indicators.columns:
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        fig, ax = plt.subplots(figsize=(12, 4))
                        ax.plot(
                            data.index,
                            indicators["RSI"],
                            color="purple",
                            linewidth=1.5,
                            label="RSI",
                        )

                        # Add RSI reference lines
                        ax.axhline(
                            y=70,
                            color="red",
                            linestyle="--",
                            alpha=0.7,
                            label="Overbought (70)",
                        )
                        ax.axhline(
                            y=30,
                            color="green",
                            linestyle="--",
                            alpha=0.7,
                            label="Oversold (30)",
                        )
                        ax.axhline(
                            y=50,
                            color="gray",
                            linestyle="-",
                            alpha=0.5,
                            label="Neutral (50)",
                        )

                        ax.set_ylim(0, 100)
                        ax.set_title(
                            f"{symbol} RSI (Relative Strength Index)",
                            fontsize=14,
                            pad=20,
                        )
                        ax.set_xlabel("Date", fontsize=12)
                        ax.set_ylabel("RSI Value", fontsize=12)
                        ax.legend(fontsize=10)
                        ax.tick_params(axis="x", rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)

                    with col2:
                        rsi_value = indicators["RSI"].iloc[-1]
                        st.metric("Current RSI", f"{rsi_value:.2f}")

                        # RSI interpretation
                        if rsi_value > 70:
                            st.warning("📈 **Overbought** - Consider selling")
                        elif rsi_value < 30:
                            st.success("📉 **Oversold** - Consider buying")
                        else:
                            st.info("➖ **Neutral** - Hold position")
            except Exception as e:
                st.warning(f"Could not calculate technical indicators: {str(e)}")

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
            st.warning(
                """
            Yahoo Finance API rate limit has been reached. This usually happens when:
            - Too many requests are made in a short time
            - Multiple users are accessing the service simultaneously

            **What you can do:**
            - Wait 5-10 minutes before trying again
            - Try a different stock symbol
            - Use cached data if available (try refreshing the page)

            The application will automatically retry with exponential backoff, but if the limit persists,
            please wait before making additional requests.
            """
            )
        elif "no data found" in error_msg:
            st.warning(f"📊 **No Data Found** for symbol '{symbol}'")
            st.info(
                """
            This could mean:
            - The stock symbol is incorrect or doesn't exist
            - The stock is delisted or not available
            - The market is closed

            Please verify the stock symbol and try again.
            """
            )
        elif "network" in error_msg or "connection" in error_msg:
            st.error("🌐 **Network Error**")
            st.warning(
                """
            Unable to connect to Yahoo Finance. Please check:
            - Your internet connection
            - Try again in a few minutes
            - The service might be temporarily unavailable
            """
            )
        else:
            st.error(f"❌ **Error**: {str(e)}")
            st.info(
                "If this error persists, please try again later or contact support."
            )
else:
    st.info("Please enter a stock symbol to begin analysis")
