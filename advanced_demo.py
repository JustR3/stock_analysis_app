#!/usr/bin/env python3
"""
Advanced Time Series Analysis Demo
Demonstrates sophisticated forecasting with stochastic calculus
"""
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def demo_advanced_analysis():
    """Demonstrate advanced time series analysis capabilities"""
    print("🚀 Advanced Time Series Analysis Demo")
    print("=" * 60)

    try:
        # Import required modules
        from stock_analysis_app.models.timeseries_models import (
            AdvancedPredictor,
            ARIMAModel,
            GARCHModel,
            RiskMetrics,
            StochasticCalculus,
            TimeSeriesAnalyzer,
        )
        from streamlit_app.utils.data_service import StreamlitDataService

        print("✅ Imported advanced analysis modules")

        # Initialize data service
        data_service = StreamlitDataService()

        # Demo with sample stock data
        symbol = "AAPL"
        print(f"\n📊 Analyzing {symbol} with Advanced Models")
        print("-" * 40)

        # Fetch data
        start_date = "2023-01-01"
        end_date = "2024-01-01"

        print(f"📡 Fetching data from {start_date} to {end_date}...")
        data, info = data_service.get_stock_data(
            symbol=symbol, start_date=start_date, end_date=end_date
        )

        if data.empty:
            print("❌ No data retrieved. Using synthetic data for demo...")
            # Generate synthetic data for demonstration
            dates = pd.date_range(start=start_date, end=end_date, freq="B")
            np.random.seed(42)

            # Generate synthetic price data with trend and volatility
            t = np.arange(len(dates))
            trend = 150 + 0.1 * t  # Upward trend
            seasonal = 10 * np.sin(2 * np.pi * t / 252)  # Annual seasonality
            noise = np.random.normal(0, 2, len(dates))  # Random noise

            prices = trend + seasonal + noise

            data = pd.DataFrame(
                {
                    "Open": prices * 0.99,
                    "High": prices * 1.02,
                    "Low": prices * 0.98,
                    "Close": prices,
                    "Volume": np.random.randint(1000000, 10000000, len(dates)),
                },
                index=dates,
            )

            info = {"currentPrice": prices[-1]}

        print(f"✅ Loaded {len(data)} data points")

        # Initialize advanced predictor
        print("\n🤖 Initializing Advanced Predictor...")
        predictor = AdvancedPredictor(data)

        # 1. Stationarity Analysis
        print("\n📈 Stationarity Analysis:")
        print("-" * 25)

        ts_analyzer = TimeSeriesAnalyzer(data)
        stationarity = ts_analyzer.test_stationarity(data["Close"])

        if stationarity:
            print(".4f")
            print(".4f")
            print(f"Conclusion: {stationarity['conclusion']}")

        # 2. Fit Models
        print("\n🎯 Fitting Advanced Models:")
        print("-" * 30)

        model_results = predictor.fit_models()

        for model_name, results in model_results.items():
            if results:
                print(".4f")
            else:
                print(f"❌ {model_name.upper()} model failed to fit")

        # 3. Generate Predictions
        print("\n🔮 Generating Predictions (30 days horizon):")
        print("-" * 45)

        predictions = predictor.generate_predictions(horizon=30)

        if predictions:
            print("✅ Predictions generated successfully")

            # ARIMA Results
            if "arima" in predictions and "mean_forecast" in predictions["arima"]:
                arima_price = predictions["arima"]["mean_forecast"]
                print(".2f")

            # GARCH Volatility
            if "garch" in predictions and "final_forecast" in predictions["garch"]:
                vol_forecast = predictions["garch"]["final_forecast"]
                print(".1%")

            # GBM Statistics
            if "gbm" in predictions and "statistics" in predictions["gbm"]:
                gbm_stats = predictions["gbm"]["statistics"]
                print(".2f")
                print(".2f")
                print(".2f")

            # Risk Metrics
            if "risk_metrics" in predictions:
                risk_data = predictions["risk_metrics"]
                if "VaR_95" in risk_data:
                    var_pct = risk_data["VaR_95"].get("VaR_percent", 0)
                    print(".2f")

        # 4. Stochastic Calculus Demo
        print("\n🎲 Stochastic Calculus Demo:")
        print("-" * 30)

        stochastic = StochasticCalculus(data["Close"])

        # Black-Scholes option pricing
        current_price = data["Close"].iloc[-1]
        strike_price = current_price * 1.05  # 5% above current price

        call_price = stochastic.black_scholes_price(
            S=current_price,
            K=strike_price,
            T=1.0,  # 1 year
            r=0.05,  # 5% risk-free rate
            option_type="call",
        )

        put_price = stochastic.black_scholes_price(
            S=current_price, K=strike_price, T=1.0, r=0.05, option_type="put"
        )

        print(".2f")
        print(".2f")
        print(".2f")

        # 5. Monte Carlo Simulation
        print("\n🎯 Monte Carlo Simulation (1,000 paths):")
        print("-" * 40)

        mc_results = stochastic.simulate_geometric_brownian_motion(
            initial_price=current_price,
            time_horizon=1.0,  # 1 year
            num_simulations=1000,
            num_steps=252,  # Daily steps
        )

        if mc_results.get("final_prices") is not None:
            final_prices = mc_results["final_prices"]
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")

        # 6. Risk Analysis
        print("\n⚠️ Risk Analysis:")
        print("-" * 20)

        risk_metrics = RiskMetrics(ts_analyzer.returns)

        # Value at Risk
        var_95 = risk_metrics.calculate_var(confidence_level=0.95)
        var_99 = risk_metrics.calculate_var(confidence_level=0.99)

        if var_95 and "VaR_percent" in var_95:
            print(".2f")
        if var_99 and "VaR_percent" in var_99:
            print(".2f")

        # Maximum Drawdown
        max_dd = risk_metrics.calculate_max_drawdown()
        if max_dd and "max_drawdown_percent" in max_dd:
            print(".2f")
            print(
                f"Recovery Period: {max_dd.get('drawdown_duration_days', 'N/A')} days"
            )

        # 7. Create Visualization
        print("\n📊 Creating Analysis Visualization...")
        print("-" * 35)

        try:
            fig = predictor.plot_predictions()
            if fig:
                fig.savefig("advanced_analysis_demo.png", dpi=300, bbox_inches="tight")
                print("✅ Analysis visualization saved as 'advanced_analysis_demo.png'")
        except Exception as e:
            print(f"⚠️ Visualization failed: {str(e)}")

        print("\n" + "=" * 60)
        print("✅ Advanced Time Series Analysis Demo Complete!")
        print("=" * 60)

        print("\n🎯 Key Insights:")
        print("• ARIMA/SARIMA models capture trend and seasonality patterns")
        print("• GARCH models handle volatility clustering in financial data")
        print("• Geometric Brownian Motion provides realistic price simulations")
        print("• Black-Scholes framework enables option pricing and risk analysis")
        print("• Monte Carlo methods quantify uncertainty in predictions")
        print("• VaR/CVaR metrics provide comprehensive risk assessment")

        print("\n📈 Applications:")
        print("• Portfolio optimization and risk management")
        print("• Option pricing and derivatives valuation")
        print("• Algorithmic trading strategy development")
        print("• Financial risk assessment and stress testing")
        print("• Investment decision support systems")

    except ImportError as e:
        print(f"❌ Import Error: {str(e)}")
        print("💡 Install required packages:")
        print("   pip install arch scipy")
        print("   pip install -e .[ml]")

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    demo_advanced_analysis()
