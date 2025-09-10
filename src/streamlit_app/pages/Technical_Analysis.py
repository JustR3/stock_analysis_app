import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from utils.config_manager import ConfigManager
from utils.data_service import StreamlitDataService

# Initialize configuration and data service
config = ConfigManager()
data_service = StreamlitDataService()

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
            
            # Display charts with configured height
            chart_height = config.get("visualization.chart_height", 400)
            
            st.subheader("Price and Bollinger Bands")
            st.line_chart(
                pd.concat([
                    data['Close'],
                    indicators[['BB_Upper', 'BB_Middle', 'BB_Lower']]
                ], axis=1),
                height=chart_height
            )
            
            st.subheader("RSI (Relative Strength Index)")
            st.line_chart(
                indicators['RSI'],
                height=chart_height
            )
            
            st.subheader("MACD (Moving Average Convergence Divergence)")
            st.line_chart(
                indicators[['MACD', 'MACD_Signal']],
                height=chart_height
            )
            
            # Display current values with overbought/oversold levels
            col1, col2, col3 = st.columns(3)
            with col1:
                rsi_value = indicators['RSI'].iloc[-1]
                # Use 'normal' for overbought (red), 'inverse' for oversold (green), 'off' for neutral
                rsi_color = "normal" if rsi_value > config.get("technical_analysis.indicators.rsi.overbought", 70) else \
                           "inverse" if rsi_value < config.get("technical_analysis.indicators.rsi.oversold", 30) else "off"
                st.metric("Current RSI", f"{rsi_value:.2f}", delta_color=rsi_color)
            with col2:
                st.metric("MACD", f"{indicators['MACD'].iloc[-1]:.2f}")
            with col3:
                st.metric("MACD Signal", f"{indicators['MACD_Signal'].iloc[-1]:.2f}")
                
        else:
            st.error(f"No data found for symbol {symbol}")
            
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
else:
    st.info("Please enter a stock symbol to begin analysis") 