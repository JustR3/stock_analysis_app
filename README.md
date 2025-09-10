# Stock Analysis App 📈

[![CI](https://github.com/JustR3/stock_analysis_app/actions/workflows/ci.yml/badge.svg)](https://github.com/JustR3/stock_analysis_app/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modern, comprehensive stock market analysis and prediction platform built with Python, featuring advanced machine learning capabilities and an interactive web interface.

## ✨ Features

- **📊 Advanced Data Analysis**: Comprehensive technical analysis with 15+ indicators
- **🤖 Machine Learning**: Multiple ML models (Linear Regression, Random Forest, XGBoost, LightGBM)
- **📈 Feature Engineering**: Lagged features, momentum indicators, and automated feature selection
- **🎯 Interactive Dashboard**: Modern Streamlit web interface
- **📉 Risk Analysis**: Volatility analysis and risk metrics
- **🔄 Automated Workflows**: CI/CD with GitHub Actions
- **📚 Well-Documented**: Comprehensive documentation and examples

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip
- virtualenv (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/JustR3/stock_analysis_app.git
   cd stock_analysis_app
   ```

2. **Create virtual environment**
   ```bash
   make venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package**
   ```bash
   make install-all  # Installs all dependencies including dev tools
   ```

4. **Set up development environment**
   ```bash
   make setup-dev
   ```

### Usage

#### 🖥️ Web Application
```bash
make app  # Runs Streamlit app at http://localhost:8501
```

#### 🧠 Machine Learning Models
```bash
# Train models for a stock
stock-analysis train AAPL --models lr rf xgb

# Analyze features
stock-analysis features AAPL

# Run analysis
stock-analysis analyze MSFT --period 1y
```

#### 📊 Jupyter Notebooks
```bash
# Launch Jupyter for exploratory analysis
jupyter notebook notebooks/exploratory/
```

## 📁 Project Structure

```
stock_analysis_app/
├── 📂 .github/
│   └── workflows/           # GitHub Actions CI/CD
├── 📂 config/               # Configuration files
│   ├── model/              # Model hyperparameters
│   ├── data/               # Data processing configs
│   └── app/                # Application settings
├── 📂 docs/                 # Documentation
│   ├── source/             # Sphinx documentation
│   └── build/              # Built documentation
├── 📂 notebooks/            # Jupyter notebooks
│   ├── exploratory/        # Data exploration
│   └── experiments/        # ML experiments
├── 📂 scripts/              # Utility scripts
│   ├── data/               # Data processing scripts
│   ├── ml/                 # ML training scripts
│   └── utils/              # General utilities
├── 📂 src/                  # Source code
│   └── stock_analysis_app/ # Main package
│       ├── cli.py          # Command-line interface
│       ├── core/           # Core business logic
│       ├── data/           # Data management
│       ├── features/       # Feature engineering
│       ├── models/         # ML models
│       ├── utils/          # Utilities
│       └── viz/            # Visualization
├── 📂 tests/                # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── conftest.py         # Test configuration
├── 📂 data/                 # Data directory
│   ├── cache/              # Cached data
│   ├── raw/                # Raw data files
│   └── processed/          # Processed data
├── 📂 models/               # Trained models
├── 📂 workflows/            # MLflow/Prefect workflows
├── 📋 Makefile             # Development tasks
├── 📋 pyproject.toml       # Project configuration
├── 📋 .pre-commit-config.yaml # Code quality hooks
├── 📋 .gitignore           # Git ignore rules
└── 📋 requirements.txt     # Dependencies
```

## 🛠️ Development

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Run all quality checks
make pre-commit
```

### Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test types
make test-unit
make test-integration
```

### Documentation

```bash
# Build documentation
make docs

# Serve documentation locally
make docs-serve
```

## 🤖 Machine Learning Pipeline

### Feature Engineering
- **Basic Features**: OHLCV data, volume metrics
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Lagged Features**: Previous day prices and indicators
- **Derived Features**: Momentum, volatility measures
- **Automated Selection**: Feature importance ranking

### Models Supported
- **Linear Regression**: Baseline model
- **Random Forest**: Ensemble method with feature importance
- **XGBoost**: Gradient boosting (optional)
- **LightGBM**: Fast gradient boosting (optional)

### Model Training
```bash
# Train multiple models
stock-analysis train AAPL --models lr rf xgb lgb --test-size 0.2

# Evaluate models
python scripts/ml/evaluate_models.py
```

## 📊 Data Pipeline

### Data Sources
- **Yahoo Finance**: Real-time stock data
- **Local Cache**: Efficient data storage
- **Processed Datasets**: Clean, feature-engineered data

### Data Processing
```bash
# Download sample data
make download-data

# Process raw data
python scripts/data/process_data.py
```

## 🔧 Configuration

### Environment Variables
```bash
# Create .env file
cp .env.example .env

# Edit configuration
nano config/app/config.yaml
```

### Model Configuration
```yaml
# config/model/default.yaml
models:
  random_forest:
    n_estimators: 100
    max_depth: 10
    random_state: 42

features:
  lag_periods: [1, 2, 3]
  technical_indicators: ['rsi', 'macd', 'bollinger']
```

## 🚀 Deployment

### Docker
```bash
# Build image
make docker-build

# Run container
make docker-run
```

### Production Deployment
```bash
# Build package
make build

# Publish to PyPI
make publish
```

## 📈 Performance Metrics

- **Model Accuracy**: R², RMSE, MAE metrics
- **Feature Importance**: SHAP values, permutation importance
- **Cross-validation**: 5-fold CV scores
- **Backtesting**: Historical performance analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Setup
```bash
# Clone and setup
git clone https://github.com/JustR3/stock_analysis_app.git
cd stock_analysis_app

# Setup development environment
make setup-dev

# Run tests
make ci  # Runs full CI pipeline locally
```

## 📚 Documentation

- **[API Reference](https://stock-analysis-app.readthedocs.io/)**: Complete API documentation
- **[Examples](notebooks/)**: Jupyter notebooks with examples
- **[Contributing Guide](docs/source/contributing.md)**: How to contribute

## 🐛 Issues & Support

- **Bug Reports**: [GitHub Issues](https://github.com/JustR3/stock_analysis_app/issues)
- **Discussions**: [GitHub Discussions](https://github.com/JustR3/stock_analysis_app/discussions)
- **Documentation**: [Read the Docs](https://stock-analysis-app.readthedocs.io/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Data Sources**: Yahoo Finance API
- **ML Libraries**: scikit-learn, XGBoost, LightGBM
- **Web Framework**: Streamlit
- **Visualization**: Matplotlib, Seaborn, Plotly

---

**Made with ❤️ for the quantitative finance community**

⭐ Star this repo if you find it useful!
