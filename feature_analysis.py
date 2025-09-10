#!/usr/bin/env python3
"""
Feature Engineering Analysis Script
Comprehensive EDA for stock prediction features
"""
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from single_stock_analysis.feature_eda import FeatureEDA
from streamlit_app.utils.data_service import StreamlitDataService


def run_feature_analysis(symbol: str = "AAPL", days: int = 365):
    """
    Run comprehensive feature analysis for a given stock

    Args:
        symbol (str): Stock symbol to analyze
        days (int): Number of days of historical data
    """
    print(f"🔍 Starting Feature Analysis for {symbol}")
    print("=" * 60)

    try:
        # Initialize services
        data_service = StreamlitDataService()

        # Fetch stock data
        print(f"📊 Fetching {days} days of {symbol} data...")
        start_date = (datetime.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        data, info = data_service.get_stock_data(
            symbol=symbol, start_date=start_date, end_date=end_date
        )

        if data.empty:
            print(f"❌ No data found for {symbol}")
            return

        print(f"✅ Loaded {len(data)} samples for {symbol}")
        print(f"📅 Date range: {data.index.min()} to {data.index.max()}")

        # Initialize EDA
        eda = FeatureEDA(data)

        # Prepare features and target
        print("\n🔧 Preparing features and target...")
        success = eda.prepare_features_and_target(target_col="Close", prediction_days=1)

        if not success:
            print("❌ Failed to prepare features")
            return

        print(f"✅ Prepared {len(eda.feature_names)} features")

        # Run correlation analysis
        print("\n📈 Running correlation analysis...")
        corr_results = eda.analyze_correlations(threshold=0.8)

        if corr_results:
            print(
                f"📊 Found {corr_results['summary']['highly_correlated_pairs']} highly correlated feature pairs"
            )

            if corr_results["high_correlation_pairs"]:
                print("\n🔗 Top correlated feature pairs:")
                for pair in corr_results["high_correlation_pairs"][:5]:
                    print(".3f")

        # Run feature importance analysis
        print("\n🌲 Running Random Forest feature importance analysis...")
        importance_results = eda.analyze_feature_importance(n_estimators=100)

        if importance_results:
            print(".4f")
            print("\n🎯 Top 10 most important features:")
            for i, feature in enumerate(importance_results["top_features"][:10], 1):
                print("2d")

        # Generate recommendations
        print("\n💡 Feature Engineering Recommendations:")
        recommendations = eda.generate_eda_report().get("recommendations", {})

        if recommendations.get("suggested_features"):
            print(
                f"✅ Suggested features ({len(recommendations['suggested_features'])}):"
            )
            for feature in recommendations["suggested_features"][:10]:
                print(f"   • {feature}")

        if recommendations.get("features_to_remove"):
            print(
                f"❌ Features to consider removing ({len(recommendations['features_to_remove'])}):"
            )
            for feature in recommendations["features_to_remove"][:5]:
                print(f"   • {feature}")

        if recommendations.get("correlation_issues"):
            print(
                f"⚠️  Correlation issues to address ({len(recommendations['correlation_issues'])}):"
            )
            for issue in recommendations["correlation_issues"][:3]:
                print(
                    f"   • {issue['features'][0]} ↔ {issue['features'][1]} (r={issue['correlation']:.3f})"
                )

        # Create visualizations
        print("\n📊 Creating visualizations...")

        # Correlation matrix
        corr_fig = eda.plot_correlation_matrix()
        if corr_fig:
            corr_fig.savefig(
                f"{symbol}_correlation_matrix.png", dpi=300, bbox_inches="tight"
            )
            print(f"✅ Saved correlation matrix to {symbol}_correlation_matrix.png")

        # Feature importance
        importance_fig = eda.plot_feature_importance(top_n=15)
        if importance_fig:
            importance_fig.savefig(
                f"{symbol}_feature_importance.png", dpi=300, bbox_inches="tight"
            )
            print(f"✅ Saved feature importance plot to {symbol}_feature_importance.png")

        # Feature selection analysis
        print("\n🎯 Optimal Feature Selection Analysis:")
        selection_results = eda.select_optimal_features(method="importance", k=10)

        if selection_results:
            print(f"📋 Recommended top 10 features using importance method:")
            for i, feature in enumerate(selection_results["selected_features"], 1):
                if eda.feature_importance is not None:
                    importance = eda.feature_importance[
                        eda.feature_importance["feature"] == feature
                    ]["importance"].iloc[0]
                    print("2d")

        print("\n" + "=" * 60)
        print("✅ Feature analysis completed successfully!")
        print(
            f"📁 Check {symbol}_correlation_matrix.png and {symbol}_feature_importance.png for visualizations"
        )

    except Exception as e:
        print(f"❌ Error during feature analysis: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run analysis for multiple stocks
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN"]

    for stock in stocks:
        try:
            run_feature_analysis(stock, days=365)
            print("\n" + "=" * 80 + "\n")
        except Exception as e:
            print(f"❌ Failed to analyze {stock}: {str(e)}")
            print("\n" + "=" * 80 + "\n")
