import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = str(Path(__file__).parent.parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from streamlit_app.utils.config_manager import ConfigManager
from streamlit_app.utils.data_service import StreamlitDataService

# Import the visualizer
from single_stock_analysis.visualizer import StockVisualizer

# Initialize configuration, data service, and visualizer
config = ConfigManager()
data_service = StreamlitDataService()
visualizer = StockVisualizer()

st.set_page_config(
    page_title="Technical Analysis",
    page_icon="📊"
)

st.title("Technical Analysis")

# Sidebar controls
st.sidebar.header("Analysis Parameters")
symbol = st.sidebar.text_input("Stock Symbol").upper()
period = st.sidebar.selectbox(
    "Time Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
    index=2  # Default to 1y
)

if symbol:
    try:
        # Calculate start date based on period
        end_date = datetime.now()
        if period == "1mo":
            start_date = end_date - timedelta(days=30)
        elif period == "3mo":
            start_date = end_date - timedelta(days=90)
        elif period == "6mo":
            start_date = end_date - timedelta(days=180)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "2y":
            start_date = end_date - timedelta(days=730)
        elif period == "5y":
            start_date = end_date - timedelta(days=1825)
        else:  # max
            start_date = end_date - timedelta(days=3650)
            
        # Fetch data using the data service
        data, _ = data_service.get_stock_data(
            symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if not data.empty:
            # Calculate technical indicators
            indicators = data_service.get_technical_indicators(data)
            
            # Display comprehensive technical analysis chart
            st.subheader("Comprehensive Technical Analysis")
            tech_fig = visualizer.plot_technical_indicators(
                data, indicators, symbol, save=False
            )
            if tech_fig:
                st.pyplot(tech_fig)

            # Technical indicators summary with color coding
            st.subheader("Technical Indicators Summary")
            col1, col2, col3 = st.columns(3)

            with col1:
                if 'RSI' in indicators.columns:
                    rsi_value = indicators['RSI'].iloc[-1]
                    # Use color coding for RSI levels
                    rsi_color = "normal" if rsi_value > config.get("technical_analysis.indicators.rsi.overbought", 70) else \
                               "inverse" if rsi_value < config.get("technical_analysis.indicators.rsi.oversold", 30) else "off"
                    st.metric("Current RSI", f"{rsi_value:.2f}", delta_color=rsi_color)

            with col2:
                if 'MACD' in indicators.columns:
                    macd_value = indicators['MACD'].iloc[-1]
                    st.metric("Current MACD", f"{macd_value:.4f}")

            with col3:
                if 'MACD_Signal' in indicators.columns:
                    signal_value = indicators['MACD_Signal'].iloc[-1]
                    st.metric("MACD Signal", f"{signal_value:.4f}")
                
        else:
            st.error(f"No data found for symbol {symbol}")
            
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
else:
    st.info("Please enter a stock symbol to begin analysis") 