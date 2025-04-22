import streamlit as st
import pandas as pd
import yfinance as yf
import ta
from datetime import datetime, timedelta
from utils.config_manager import ConfigManager

# Initialize configuration
config = ConfigManager()

st.set_page_config(
    page_title="Technical Analysis",
    page_icon="📊"
)

st.title("Technical Analysis")

# Sidebar controls
st.sidebar.header("Analysis Parameters")
default_symbol = config.get("data.default_symbol", "AAPL")
symbol = st.sidebar.text_input("Stock Symbol", default_symbol).upper()
period = st.sidebar.selectbox(
    "Time Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
    index=2  # Default to 1y
)

if symbol:
    try:
        # Fetch data
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if not hist.empty:
            # Calculate technical indicators with configured parameters
            # RSI
            rsi_period = config.get("technical_analysis.indicators.rsi.period", 14)
            hist['RSI'] = ta.momentum.RSIIndicator(hist['Close'], window=rsi_period).rsi()
            
            # MACD
            macd_fast = config.get("technical_analysis.indicators.macd.fast_period", 12)
            macd_slow = config.get("technical_analysis.indicators.macd.slow_period", 26)
            macd_signal = config.get("technical_analysis.indicators.macd.signal_period", 9)
            macd = ta.trend.MACD(
                hist['Close'],
                window_fast=macd_fast,
                window_slow=macd_slow,
                window_sign=macd_signal
            )
            hist['MACD'] = macd.macd()
            hist['MACD_Signal'] = macd.macd_signal()
            
            # Bollinger Bands
            bb_period = config.get("technical_analysis.indicators.bollinger_bands.period", 20)
            bb_std = config.get("technical_analysis.indicators.bollinger_bands.std_dev", 2)
            bollinger = ta.volatility.BollingerBands(
                hist['Close'],
                window=bb_period,
                window_dev=bb_std
            )
            hist['BB_Upper'] = bollinger.bollinger_hband()
            hist['BB_Lower'] = bollinger.bollinger_lband()
            
            # Display charts with configured height
            chart_height = config.get("visualization.chart_height", 400)
            
            st.subheader("Price and Bollinger Bands")
            st.line_chart(
                hist[['Close', 'BB_Upper', 'BB_Lower']],
                height=chart_height
            )
            
            st.subheader("RSI (Relative Strength Index)")
            st.line_chart(
                hist['RSI'],
                height=chart_height
            )
            
            st.subheader("MACD (Moving Average Convergence Divergence)")
            st.line_chart(
                hist[['MACD', 'MACD_Signal']],
                height=chart_height
            )
            
            # Display current values with overbought/oversold levels
            col1, col2, col3 = st.columns(3)
            with col1:
                rsi_value = hist['RSI'].iloc[-1]
                # Use 'normal' for overbought (red), 'inverse' for oversold (green), 'off' for neutral
                rsi_color = "normal" if rsi_value > config.get("technical_analysis.indicators.rsi.overbought", 70) else \
                           "inverse" if rsi_value < config.get("technical_analysis.indicators.rsi.oversold", 30) else "off"
                st.metric("Current RSI", f"{rsi_value:.2f}", delta_color=rsi_color)
            with col2:
                st.metric("MACD", f"{hist['MACD'].iloc[-1]:.2f}")
            with col3:
                st.metric("MACD Signal", f"{hist['MACD_Signal'].iloc[-1]:.2f}")
                
        else:
            st.error(f"No data found for symbol {symbol}")
            
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
else:
    st.info("Please enter a stock symbol to begin analysis") 