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

# Import the StockPredictor
from single_stock_analysis.prediction.predictor import StockPredictor

# Initialize configuration and data service
config = ConfigManager()
data_service = StreamlitDataService()

st.set_page_config(
    page_title="Stock Price Predictions",
    page_icon="🔮"
)

st.title("Stock Price Predictions")
st.markdown("Use machine learning models to predict future stock prices based on historical data and technical indicators.")

# Sidebar controls
st.sidebar.header("Prediction Settings")
symbol = st.sidebar.text_input("Stock Symbol").upper()
days_ahead = st.sidebar.slider("Days Ahead to Predict", 1, 30, 5)
model_choice = st.sidebar.selectbox(
    "ML Model",
    ["random_forest", "linear_regression"],
    index=0
)

# Train/Test split
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100

if symbol:
    try:
        # Fetch stock data
        with st.spinner("Fetching stock data..."):
            data, info = data_service.get_stock_data(
                symbol,
                start_date=(datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d')
            )

        if not data.empty:
            # Display basic stock info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
            with col2:
                st.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,.0f}")
            with col3:
                st.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")

            # Initialize predictor
            predictor = StockPredictor(data)

            # Check available features
            st.subheader("📊 Data Analysis")
            X, y, feature_names = predictor.prepare_features(days_ahead)

            if len(X) > 0 and len(y) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"📈 **Available Features:** {len(feature_names)}")
                    with st.expander("Feature List"):
                        for feature in feature_names:
                            st.write(f"• {feature}")

                with col2:
                    st.info(f"📊 **Training Samples:** {len(X)}")
                    st.info(f"🎯 **Target:** {days_ahead}-day future price")

                # Model training section
                st.subheader("🤖 Model Training")

                if st.button("Train Model", type="primary"):
                    with st.spinner(f"Training {model_choice} model..."):
                        result = predictor.train_model(model_choice, test_size)

                        if 'error' not in result:
                            st.success("✅ Model trained successfully!")

                            # Display metrics
                            metrics = result['metrics']
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric("Test R²", f"{metrics['test_r2']:.4f}")
                            with col2:
                                st.metric("Test MSE", f"{metrics['test_mse']:.4f}")
                            with col3:
                                st.metric("Test MAE", f"{metrics['test_mae']:.4f}")
                            with col4:
                                st.metric("Training Samples", result['training_samples'])

                            # Model performance visualization
                            st.subheader("📈 Model Performance")

                            # Create a simple performance chart
                            performance_data = {
                                'Metric': ['R² Score', 'MSE', 'MAE'],
                                'Train': [metrics['train_r2'], metrics['train_mse'], metrics['train_mae']],
                                'Test': [metrics['test_r2'], metrics['test_mse'], metrics['test_mae']]
                            }

                            perf_df = pd.DataFrame(performance_data)
                            st.bar_chart(perf_df.set_index('Metric'))
                        else:
                            st.error(f"❌ Training failed: {result['error']}")

                # Prediction section
                st.subheader("🔮 Price Prediction")

                available_models = predictor.get_available_models()

                if available_models:
                    if st.button("Generate Prediction", type="secondary"):
                        with st.spinner("Generating prediction..."):
                            prediction = predictor.predict(days_ahead, model_choice)

                            if not prediction.empty:
                                predicted_price = prediction.iloc[0]
                                prediction_date = prediction.index[0]

                                # Display prediction
                                st.success(f"🎯 **Predicted Price:** ${predicted_price:.2f}")
                                st.info(f"📅 **Prediction Date:** {prediction_date.strftime('%Y-%m-%d')}")

                                # Compare with current price
                                current_price = data['Close'].iloc[-1]
                                price_change = ((predicted_price - current_price) / current_price) * 100

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Current Price", f"${current_price:.2f}")
                                with col2:
                                    color = "inverse" if price_change > 0 else "normal"
                                    st.metric("Predicted Change", f"{price_change:+.2f}%", delta_color=color)

                                # Show recent price history
                                st.subheader("📈 Recent Price History")
                                recent_data = data.tail(30)  # Last 30 days

                                # Create a simple line chart
                                chart_data = pd.DataFrame({
                                    'Close': recent_data['Close']
                                })
                                st.line_chart(chart_data)

                                # Add prediction point to chart
                                st.info("💡 **Note:** This prediction is based on historical patterns and technical indicators. Always consider multiple factors and consult with financial advisors before making investment decisions.")

                            else:
                                st.error("❌ Failed to generate prediction. Please ensure the model is trained.")
                else:
                    st.warning("⚠️ **No trained models available.** Please train a model first.")

                # Model comparison section
                st.subheader("⚖️ Model Comparison")

                if st.button("Compare All Models"):
                    with st.spinner("Evaluating all models..."):
                        results = predictor.evaluate_models()

                        # Create comparison table
                        comparison_data = []
                        for model_name, result in results.items():
                            if 'error' not in result:
                                metrics = result['metrics']
                                comparison_data.append({
                                    'Model': model_name.replace('_', ' ').title(),
                                    'Test R²': metrics['test_r2'],
                                    'Test MSE': metrics['test_mse'],
                                    'Test MAE': metrics['test_mae']
                                })

                        if comparison_data:
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Test R²']))

                            # Best model recommendation
                            best_model = max(comparison_data, key=lambda x: x['Test R²'])
                            st.success(f"🏆 **Recommended Model:** {best_model['Model']} (R²: {best_model['Test R²']:.4f})")
                        else:
                            st.error("❌ Model comparison failed. Check the logs for details.")

            else:
                st.warning("⚠️ **Insufficient data for prediction.** The stock needs more historical data or technical indicators.")

        else:
            st.error(f"❌ No data found for symbol {symbol}")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 **Troubleshooting tips:**")
        st.info("• Make sure the stock symbol is valid")
        st.info("• Check your internet connection")
        st.info("• Try a different stock symbol")

else:
    st.info("🎯 **Getting Started:**")
    st.markdown("""
    1. Enter a valid stock symbol (e.g., AAPL, GOOGL, MSFT, TSLA)
    2. Adjust prediction settings in the sidebar
    3. Train a machine learning model
    4. Generate price predictions
    5. Compare different models

    **Note:** Predictions are based on historical data and technical indicators.
    They should not be used as the sole basis for investment decisions.
    """)

    # Show some example stocks
    st.subheader("📋 Popular Stock Symbols to Try:")
    example_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META"]
    cols = st.columns(4)
    for i, stock in enumerate(example_stocks):
        with cols[i % 4]:
            if st.button(stock, key=f"example_{stock}"):
                st.session_state.symbol = stock
                st.rerun()
