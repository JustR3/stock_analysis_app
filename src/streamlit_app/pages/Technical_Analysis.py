import streamlit as st
import pandas as pd
import yfinance as yf
import ta
from datetime import datetime, timedelta

st.set_page_config(page_title="Technical Analysis", page_icon="📊")

st.title("Technical Analysis")

# Sidebar controls
st.sidebar.header("Analysis Parameters")
symbol = st.sidebar.text_input("Stock Symbol", "AAPL").upper()
period = st.sidebar.selectbox(
    "Time Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
)

if symbol:
    try:
        # Fetch data
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if not hist.empty:
            # Calculate technical indicators
            # RSI
            hist['RSI'] = ta.momentum.RSIIndicator(hist['Close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(hist['Close'])
            hist['MACD'] = macd.macd()
            hist['MACD_Signal'] = macd.macd_signal()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(hist['Close'])
            hist['BB_Upper'] = bollinger.bollinger_hband()
            hist['BB_Lower'] = bollinger.bollinger_lband()
            
            # Display charts
            st.subheader("Price and Bollinger Bands")
            st.line_chart(hist[['Close', 'BB_Upper', 'BB_Lower']])
            
            st.subheader("RSI (Relative Strength Index)")
            st.line_chart(hist['RSI'])
            
            st.subheader("MACD (Moving Average Convergence Divergence)")
            st.line_chart(hist[['MACD', 'MACD_Signal']])
            
            # Display current values
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current RSI", f"{hist['RSI'].iloc[-1]:.2f}")
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