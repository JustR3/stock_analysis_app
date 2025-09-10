#!/usr/bin/env python3
"""
Simple Feature Analysis Script
Basic analysis of stock prediction features without complex dependencies
"""
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


# Simple data analysis without complex dependencies
def analyze_current_features():
    """Analyze the current feature set from saved models"""

    print("🔍 Simple Feature Analysis")
    print("=" * 50)

    # Check model directories
    model_dirs = ["models", "src/models"]

    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            print(f"\n📁 {model_dir}/ analysis:")

            # Look for feature files
            feature_files = [f for f in os.listdir(model_dir) if "features" in f]
            model_files = [f for f in os.listdir(model_dir) if "model" in f]

            print(f"  📊 Found {len(feature_files)} feature files")
            print(f"  🤖 Found {len(model_files)} model files")

            # Analyze feature files
            for feature_file in feature_files:
                try:
                    # Simple analysis without joblib dependency issues
                    file_path = os.path.join(model_dir, feature_file)
                    file_size = os.path.getsize(file_path)

                    model_name = feature_file.replace("_features.pkl", "")
                    print(f"  ✅ {model_name}: {file_size} bytes")

                except Exception as e:
                    print(f"  ❌ Error analyzing {feature_file}: {e}")


def analyze_feature_importance_conceptually():
    """Provide conceptual analysis of feature importance"""

    print("\n🧠 Conceptual Feature Importance Analysis")
    print("=" * 50)

    print("\n📈 PRICE FEATURES (Usually High Importance):")
    print("  • Close: Current price - MOST important for prediction")
    print("  • Open: Opening price - Good context")
    print("  • High/Low: Price range - Volatility indicators")

    print("\n⏰ LAG FEATURES (NEW - Should be High Importance):")
    print("  • Close_lag_1: Previous day price - Strong predictor")
    print("  • Close_lag_2: Two days ago price - Momentum indicator")

    print("\n📊 TECHNICAL INDICATORS (Variable Importance):")
    print("  • RSI: Momentum oscillator - Medium importance")
    print("  • MACD: Trend indicator - Medium importance")
    print("  • Bollinger Bands: Volatility measure - Low-Medium importance")
    print(
        "  • Moving Averages (MA_20, MA_50, MA_200): Trend following - High importance"
    )

    print("\n💰 VOLUME & RETURNS (Context Features):")
    print("  • Volume: Trading activity - Supporting feature")
    print("  • daily_return: Price change % - Momentum context")
    print("  • volatility: Price stability - Risk context")


def suggest_feature_optimization():
    """Suggest optimal feature selection"""

    print("\n🎯 Feature Optimization Recommendations")
    print("=" * 50)

    print("\n✅ TOP RECOMMENDED FEATURES (High Impact):")
    top_features = [
        "Close",  # Current price
        "Close_lag_1",  # Previous day price
        "Close_lag_2",  # Two days ago price
        "MA_20",  # Short-term trend
        "MA_50",  # Medium-term trend
        "daily_return",  # Recent momentum
        "Volume",  # Market participation
    ]

    for i, feature in enumerate(top_features, 1):
        print("2d")

    print("\n⚠️  FEATURES TO CONSIDER REMOVING (Low Impact):")
    low_importance = [
        "BB_Upper",  # Redundant with other volatility measures
        "BB_Lower",  # Redundant with other volatility measures
        "MACD_Signal",  # Complex signal, may overfit
        "MA_200",  # Long-term, less relevant for short-term prediction
    ]

    for feature in low_importance:
        print(f"   • {feature}")

    print("\n🔄 POTENTIAL NEW FEATURES TO ADD:")
    new_features = [
        "Close_lag_3",  # Three-day lag
        "Close_lag_5",  # Five-day lag
        "price_change_5d",  # 5-day price change
        "volume_ma_5",  # 5-day volume average
        "rsi_lag_1",  # Previous RSI
        "momentum_5d",  # 5-day momentum
    ]

    for feature in new_features:
        print(f"   • {feature}")


def analyze_correlation_patterns():
    """Analyze expected correlation patterns"""

    print("\n🔗 Expected Correlation Patterns")
    print("=" * 50)

    print("\n📈 HIGHLY CORRELATED (r > 0.8) - May cause multicollinearity:")
    high_corr = [
        ("Close", "MA_20", "Very high - moving average of close price"),
        ("Close", "Close_lag_1", "High - consecutive day prices"),
        ("MA_20", "MA_50", "High - overlapping moving averages"),
        ("High", "Low", "High - price range boundaries"),
        ("BB_Upper", "BB_Lower", "Very high - both based on same calculation"),
    ]

    for feature1, feature2, reason in high_corr:
        print(f"   • {feature1} ↔ {feature2}: {reason}")

    print("\n📊 MODERATELY CORRELATED (0.3 < r < 0.8) - Useful relationships:")
    mod_corr = [
        ("Close", "Volume", "Price and trading activity"),
        ("RSI", "daily_return", "Momentum indicators"),
        ("volatility", "BB_Upper", "Volatility measures"),
        ("MACD", "MA_50", "Trend indicators"),
    ]

    for feature1, feature2, reason in mod_corr:
        print(f"   • {feature1} ↔ {feature2}: {reason}")


def main():
    """Main analysis function"""
    print("🚀 STOCK PREDICTION FEATURE ANALYSIS")
    print("=" * 60)

    analyze_current_features()
    analyze_feature_importance_conceptually()
    suggest_feature_optimization()
    analyze_correlation_patterns()

    print("\n" + "=" * 60)
    print("✅ Analysis Complete!")
    print("\n💡 Key Takeaways:")
    print("   • Focus on Close, lagged prices, and key MAs")
    print("   • RSI and MACD provide additional momentum signals")
    print("   • Volume and volatility add context but may have lower direct impact")
    print("   • Watch for multicollinearity between similar features")
    print("   • Consider removing redundant Bollinger Band features")


if __name__ == "__main__":
    main()
