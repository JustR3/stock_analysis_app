"""
Advanced Time Series Analysis Page
Provides sophisticated forecasting with stochastic calculus
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from single_stock_analysis.prediction.predictor import StockPredictor
from streamlit_app.utils.config_manager import ConfigManager
from streamlit_app.utils.data_service import StreamlitDataService

# Initialize services
config = ConfigManager()
data_service = StreamlitDataService()

st.set_page_config(
    page_title="Advanced Time Series Analysis", page_icon="🔬", layout="wide"
)

st.title("🔬 Advanced Time Series Analysis")
st.markdown(
    """
This page provides **sophisticated time series forecasting** using:
- **ARIMA/SARIMA** models for trend analysis
- **GARCH** models for volatility forecasting
- **Geometric Brownian Motion** for stochastic simulation
- **Black-Scholes** option pricing
- **Value at Risk (VaR)** and **CVaR** calculations
- **Monte Carlo** simulations
"""
)

# Sidebar controls
st.sidebar.header("📊 Analysis Parameters")

# Stock selection
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()

# Analysis parameters
col1, col2 = st.sidebar.columns(2)

with col1:
    forecast_horizon = st.slider(
        "Forecast Horizon (days)",
        min_value=5,
        max_value=90,
        value=30,
        help="Number of days to forecast ahead",
    )

with col2:
    confidence_level = st.slider(
        "Confidence Level (%)",
        min_value=80,
        max_value=99,
        value=95,
        help="Confidence level for predictions",
    )

# Advanced options
st.sidebar.subheader("⚙️ Advanced Options")

gbm_simulations = st.sidebar.slider(
    "GBM Simulations",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
    help="Number of Monte Carlo simulations",
)

# Analysis sections
analysis_sections = st.sidebar.multiselect(
    "Analysis Sections",
    ["Time Series Models", "Stochastic Simulations", "Risk Analysis", "Option Pricing"],
    default=["Time Series Models", "Stochastic Simulations", "Risk Analysis"],
)

# Main analysis
if symbol:
    try:
        # Fetch data
        with st.spinner(f"Fetching data for {symbol}..."):
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")

            data, info = data_service.get_stock_data(
                symbol=symbol, start_date=start_date, end_date=end_date
            )

        if data.empty:
            st.error(f"❌ No data found for symbol {symbol}")
        else:
            st.success(f"✅ Loaded {len(data)} days of data for {symbol}")

            # Basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
            with col2:
                st.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,.0f}")
            with col3:
                st.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
            with col4:
                st.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")

            # Initialize predictor
            predictor = StockPredictor(data)

            # Time Series Models Section
            if "Time Series Models" in analysis_sections:
                st.header("📈 Time Series Models")

                # Generate advanced predictions
                with st.spinner("Fitting ARIMA and GARCH models..."):
                    predictions = predictor.get_advanced_predictions(
                        horizon=forecast_horizon
                    )

                if predictions.get("status") == "success":
                    pred_data = predictions["predictions"]

                    # ARIMA Results
                    if "arima" in pred_data:
                        st.subheader("📊 ARIMA Forecast")

                        arima_data = pred_data["arima"]
                        if "forecast" in arima_data:
                            forecast_df = arima_data["forecast"]

                            # Create forecast plot
                            fig, ax = plt.subplots(figsize=(12, 6))

                            # Plot historical data
                            ax.plot(
                                data.index[-50:],
                                data["Close"].iloc[-50:],
                                label="Historical",
                                color="blue",
                                linewidth=2,
                            )

                            # Plot forecast
                            ax.plot(
                                forecast_df.index,
                                forecast_df["forecast"],
                                label="ARIMA Forecast",
                                color="red",
                                linewidth=2,
                                linestyle="--",
                            )

                            # Plot confidence intervals
                            ax.fill_between(
                                forecast_df.index,
                                forecast_df["lower_ci"],
                                forecast_df["upper_ci"],
                                alpha=0.3,
                                color="red",
                                label=f"{confidence_level}% Confidence Interval",
                            )

                            ax.set_title(
                                f"{symbol} ARIMA Price Forecast ({forecast_horizon} days)",
                                fontsize=14,
                            )
                            ax.set_xlabel("Date")
                            ax.set_ylabel("Price ($)")
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            plt.xticks(rotation=45)

                            st.pyplot(fig)

                            # Forecast metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Forecast Price",
                                    f"${arima_data.get('mean_forecast', 0):.2f}",
                                )
                            with col2:
                                st.metric(
                                    "Upper Bound",
                                    f"${forecast_df['upper_ci'].iloc[-1]:.2f}",
                                )
                            with col3:
                                st.metric(
                                    "Lower Bound",
                                    f"${forecast_df['lower_ci'].iloc[-1]:.2f}",
                                )

                    # GARCH Results
                    if "garch" in pred_data:
                        st.subheader("📈 GARCH Volatility Forecast")

                        garch_data = pred_data["garch"]
                        if "volatility_forecast" in garch_data:
                            vol_df = garch_data["volatility_forecast"]

                            fig, ax = plt.subplots(figsize=(12, 6))

                            # Plot historical volatility
                            returns = np.log(
                                data["Close"] / data["Close"].shift(1)
                            ).dropna()
                            hist_vol = returns.rolling(window=30).std() * np.sqrt(252)
                            ax.plot(
                                hist_vol.index[-50:],
                                hist_vol.iloc[-50:],
                                label="Historical Volatility",
                                color="blue",
                                linewidth=2,
                            )

                            # Plot forecast
                            ax.plot(
                                vol_df.index,
                                vol_df["volatility_forecast"],
                                label="GARCH Forecast",
                                color="orange",
                                linewidth=2,
                                linestyle="--",
                            )

                            ax.set_title(
                                f"{symbol} Volatility Forecast (GARCH)", fontsize=14
                            )
                            ax.set_xlabel("Date")
                            ax.set_ylabel("Annualized Volatility")
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            plt.xticks(rotation=45)

                            st.pyplot(fig)

                else:
                    st.error(f"❌ {predictions.get('error', 'Unknown error')}")

            # Stochastic Simulations Section
            if "Stochastic Simulations" in analysis_sections:
                st.header("🎲 Stochastic Simulations")

                with st.spinner("Running Monte Carlo simulations..."):
                    # Get GBM simulation results
                    if predictor.advanced_predictor:
                        gbm_results = predictor.advanced_predictor.stochastic.simulate_geometric_brownian_motion(
                            initial_price=data["Close"].iloc[-1],
                            time_horizon=forecast_horizon / 252,  # Convert to years
                            num_simulations=gbm_simulations,
                            num_steps=forecast_horizon,
                        )

                        if gbm_results.get("price_paths") is not None:
                            st.subheader("🎯 Geometric Brownian Motion Simulation")

                            price_paths = gbm_results["price_paths"]
                            final_prices = gbm_results["final_prices"]

                            # Plot simulation paths
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                            # Price paths
                            for i in range(
                                min(50, price_paths.shape[0])
                            ):  # Plot first 50 paths
                                ax1.plot(price_paths[i, :], alpha=0.3, color="blue")
                            ax1.axhline(
                                y=data["Close"].iloc[-1],
                                color="red",
                                linestyle="--",
                                label="Current Price",
                            )
                            ax1.set_title(f"{symbol} GBM Price Simulations")
                            ax1.set_xlabel("Days")
                            ax1.set_ylabel("Price ($)")
                            ax1.legend()
                            ax1.grid(True, alpha=0.3)

                            # Distribution of final prices
                            ax2.hist(
                                final_prices,
                                bins=50,
                                alpha=0.7,
                                color="skyblue",
                                density=True,
                            )
                            ax2.axvline(
                                x=np.mean(final_prices),
                                color="red",
                                linestyle="--",
                                label=f"Mean: ${np.mean(final_prices):.2f}",
                            )
                            ax2.axvline(
                                x=np.percentile(final_prices, 50),
                                color="green",
                                linestyle="--",
                                label=f"Median: ${np.percentile(final_prices, 50):.2f}",
                            )
                            ax2.set_title("Distribution of Final Prices")
                            ax2.set_xlabel("Price ($)")
                            ax2.set_ylabel("Density")
                            ax2.legend()
                            ax2.grid(True, alpha=0.3)

                            plt.tight_layout()
                            st.pyplot(fig)

                            # Statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(
                                    "Expected Price",
                                    f"${gbm_results['statistics']['mean']:.2f}",
                                )
                            with col2:
                                st.metric(
                                    "5% Quantile",
                                    f"${gbm_results['statistics']['quantile_5']:.2f}",
                                )
                            with col3:
                                st.metric(
                                    "95% Quantile",
                                    f"${gbm_results['statistics']['quantile_95']:.2f}",
                                )
                            with col4:
                                st.metric(
                                    "Std Deviation",
                                    f"${gbm_results['statistics']['std']:.2f}",
                                )

            # Risk Analysis Section
            if "Risk Analysis" in analysis_sections:
                st.header("⚠️ Risk Analysis")

                with st.spinner("Calculating risk metrics..."):
                    if predictor.advanced_predictor:
                        # Calculate VaR
                        returns = np.log(
                            data["Close"] / data["Close"].shift(1)
                        ).dropna()

                        # Historical VaR
                        hist_var = np.percentile(
                            returns, (1 - confidence_level / 100) * 100
                        )

                        # Parametric VaR (assuming normal distribution)
                        mean_return = returns.mean()
                        std_return = returns.std()
                        z_score = abs(
                            np.percentile(
                                np.random.normal(0, 1, 10000),
                                (1 - confidence_level / 100) * 100,
                            )
                        )
                        param_var = mean_return + z_score * std_return

                        # Maximum Drawdown
                        cumulative = (1 + returns).cumprod()
                        running_max = cumulative.expanding().max()
                        drawdown = (cumulative - running_max) / running_max
                        max_dd = drawdown.min()

                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric(
                                "Historical VaR",
                                f"{hist_var*100:.2f}%",
                                help=f"{confidence_level}% confidence level",
                            )

                        with col2:
                            st.metric(
                                "Parametric VaR",
                                f"{param_var*100:.2f}%",
                                help="Assuming normal distribution",
                            )

                        with col3:
                            st.metric(
                                "Maximum Drawdown",
                                f"{max_dd*100:.2f}%",
                                help="Worst peak-to-trough decline",
                            )

                        with col4:
                            st.metric(
                                "Sharpe Ratio",
                                f"{mean_return/std_return*np.sqrt(252):.2f}",
                                help="Risk-adjusted return",
                            )

                        # Risk visualization
                        st.subheader("📊 Risk Distribution")

                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                        # Returns distribution
                        ax1.hist(
                            returns, bins=50, alpha=0.7, color="skyblue", density=True
                        )
                        ax1.axvline(
                            x=hist_var,
                            color="red",
                            linestyle="--",
                            label=f"VaR {confidence_level}%: {hist_var*100:.2f}%",
                        )
                        ax1.set_title("Returns Distribution with VaR")
                        ax1.set_xlabel("Daily Return")
                        ax1.set_ylabel("Density")
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)

                        # Drawdown chart
                        ax2.fill_between(
                            drawdown.index, drawdown.values, 0, color="red", alpha=0.3
                        )
                        ax2.set_title("Drawdown Chart")
                        ax2.set_xlabel("Date")
                        ax2.set_ylabel("Drawdown")
                        ax2.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)

                        plt.tight_layout()
                        st.pyplot(fig)

            # Option Pricing Section
            if "Option Pricing" in analysis_sections:
                st.header("💰 Option Pricing")

                col1, col2 = st.columns(2)

                with col1:
                    strike_price = st.number_input(
                        "Strike Price ($)",
                        min_value=0.0,
                        value=float(data["Close"].iloc[-1] * 1.1),
                        step=1.0,
                    )

                with col2:
                    time_to_expiry = st.slider(
                        "Time to Expiry (years)",
                        min_value=0.1,
                        max_value=2.0,
                        value=1.0,
                        step=0.1,
                    )

                risk_free_rate = (
                    st.slider(
                        "Risk-Free Rate (%)",
                        min_value=0.0,
                        max_value=10.0,
                        value=5.0,
                        step=0.1,
                    )
                    / 100
                )

                if st.button("Calculate Option Prices"):
                    with st.spinner("Calculating Black-Scholes prices..."):
                        if predictor.advanced_predictor:
                            current_price = data["Close"].iloc[-1]

                            # Calculate call and put prices
                            call_price = predictor.advanced_predictor.stochastic.black_scholes_price(
                                S=current_price,
                                K=strike_price,
                                T=time_to_expiry,
                                r=risk_free_rate,
                                option_type="call",
                            )

                            put_price = predictor.advanced_predictor.stochastic.black_scholes_price(
                                S=current_price,
                                K=strike_price,
                                T=time_to_expiry,
                                r=risk_free_rate,
                                option_type="put",
                            )

                            # Greeks (simplified)
                            d1 = (
                                np.log(current_price / strike_price)
                                + (
                                    risk_free_rate
                                    + predictor.advanced_predictor.stochastic.volatility
                                    ** 2
                                    / 2
                                )
                                * time_to_expiry
                            ) / (
                                predictor.advanced_predictor.stochastic.volatility
                                * np.sqrt(time_to_expiry)
                            )

                            delta_call = stats.norm.cdf(d1)
                            delta_put = delta_call - 1

                            # Display results
                            st.subheader("📈 Black-Scholes Option Prices")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Call Option Price", f"${call_price:.2f}")
                                st.metric("Call Delta", f"{delta_call:.3f}")

                            with col2:
                                st.metric("Put Option Price", f"${put_price:.2f}")
                                st.metric("Put Delta", f"{delta_put:.3f}")

                            with col3:
                                st.metric(
                                    "Current Stock Price", f"${current_price:.2f}"
                                )
                                st.metric("Strike Price", f"${strike_price:.2f}")
                                st.metric(
                                    "Time to Expiry", f"{time_to_expiry:.1f} years"
                                )

                            # Option analysis
                            if strike_price > current_price:
                                st.info(
                                    "📊 **Out-of-the-Money Call** - Strike > Current Price"
                                )
                            elif strike_price < current_price:
                                st.info(
                                    "📊 **In-the-Money Call** - Strike < Current Price"
                                )
                            else:
                                st.info("📊 **At-the-Money** - Strike = Current Price")

    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.info("💡 **Troubleshooting Tips:**")
        st.info("• Check if the stock symbol is valid")
        st.info("• Ensure you have a stable internet connection")
        st.info("• Try with a different stock symbol")
        st.info("• Check the logs for more detailed error information")

else:
    st.info("👆 Please enter a stock symbol to begin advanced analysis")

# Footer
st.markdown("---")
st.markdown(
    """
**🔬 Advanced Analysis Features:**
- **ARIMA/SARIMA**: Statistical forecasting models
- **GARCH**: Volatility clustering analysis
- **GBM**: Stochastic price simulations
- **VaR/CVaR**: Risk quantification
- **Black-Scholes**: Option pricing models
- **Monte Carlo**: Probabilistic forecasting
"""
)
