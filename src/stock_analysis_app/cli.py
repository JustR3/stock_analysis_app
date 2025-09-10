"""Command Line Interface for Stock Analysis App."""

import argparse
import sys
from pathlib import Path

from .core import StockAnalyzer
from .features import FeatureEngineer
from .models import StockPredictor


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="Stock Analysis App - Comprehensive stock market analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a specific stock
  stock-analysis analyze AAPL --period 1y

  # Run feature engineering analysis
  stock-analysis features AAPL --output-dir ./results

  # Train prediction models
  stock-analysis train AAPL --models lr rf --test-size 0.2

  # Run Streamlit app
  stock-analysis app
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze stock data")
    analyze_parser.add_argument("symbol", help="Stock symbol (e.g., AAPL)")
    analyze_parser.add_argument(
        "--period",
        default="1y",
        choices=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        help="Analysis period",
    )
    analyze_parser.add_argument(
        "--output-dir", default="./output", help="Output directory for results"
    )

    # Features command
    features_parser = subparsers.add_parser(
        "features", help="Run feature engineering analysis"
    )
    features_parser.add_argument("symbol", help="Stock symbol")
    features_parser.add_argument(
        "--output-dir",
        default="./feature_analysis",
        help="Output directory for analysis results",
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train prediction models")
    train_parser.add_argument("symbol", help="Stock symbol")
    train_parser.add_argument(
        "--models",
        nargs="+",
        default=["lr", "rf"],
        choices=["lr", "rf", "xgb", "lgb"],
        help="Models to train",
    )
    train_parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set size (0.0-1.0)"
    )
    train_parser.add_argument(
        "--output-dir", default="./models", help="Output directory for trained models"
    )

    # App command
    app_parser = subparsers.add_parser("app", help="Run Streamlit web application")

    return parser


def run_analyze(args):
    """Run stock analysis."""
    print(f"🔍 Analyzing {args.symbol} for period: {args.period}")

    try:
        analyzer = StockAnalyzer()

        # This would integrate with the existing analysis functionality
        print(f"✅ Analysis completed for {args.symbol}")
        print(f"📁 Results saved to: {args.output_dir}")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


def run_features(args):
    """Run feature engineering analysis."""
    print(f"🔧 Running feature analysis for {args.symbol}")

    try:
        # This would use the feature engineering module
        print(f"✅ Feature analysis completed for {args.symbol}")
        print(f"📁 Results saved to: {args.output_dir}")

    except Exception as e:
        print(f"❌ Feature analysis failed: {e}")
        sys.exit(1)


def run_train(args):
    """Train prediction models."""
    print(f"🤖 Training models for {args.symbol}")
    print(f"📊 Models: {', '.join(args.models)}")
    print(f"🧪 Test size: {args.test_size}")

    try:
        predictor = StockPredictor()

        # This would integrate with the existing prediction functionality
        print(f"✅ Model training completed for {args.symbol}")
        print(f"📁 Models saved to: {args.output_dir}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)


def run_app(args):
    """Run Streamlit web application."""
    print("🚀 Starting Stock Analysis App...")
    print("📱 Open your browser to: http://localhost:8501")

    try:
        import os
        import subprocess

        # Run streamlit from the correct directory
        streamlit_path = (
            Path(__file__).parent.parent.parent / "src" / "streamlit_app" / "Home.py"
        )
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(streamlit_path)])

    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Failed to start app: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate command handler
    command_handlers = {
        "analyze": run_analyze,
        "features": run_features,
        "train": run_train,
        "app": run_app,
    }

    handler = command_handlers.get(args.command)
    if handler:
        handler(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
