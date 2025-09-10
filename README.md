# Stock Analysis App

A Python application for stock market analysis, visualization, and prediction. The app provides tools for collecting, analyzing, and visualizing stock data, with a focus on single stock analysis. Now with a modern Streamlit web interface that starts with empty stock symbol inputs, allowing users to analyze any stock of their choice!

## Features

- **Interactive Web Interface**:
  - Modern, responsive Streamlit dashboard
  - Real-time stock data visualization
  - Technical analysis indicators
  - Configurable settings
- **Data Collection**: Fetch historical stock data using yfinance
- **Data Management**: Efficient caching system for stock data
- **Analysis Tools**:
  - Returns calculation (daily, weekly, monthly)
  - Volatility analysis
  - Moving averages
  - Summary statistics
  - Technical indicators (RSI, MACD, Bollinger Bands)
- **Visualization**:
  - Interactive price charts
  - Technical indicator plots
  - Professional styling
- **Trading Strategies**:
  - Momentum-based trading signals
- **Price Prediction**:
  - Feature preparation
  - Model training
  - Future price predictions

## Project Structure

```
stock_analysis_app/
├── src/
│   ├── single_stock_analysis/    # Core analysis modules
│   │   ├── data_collector.py    # Data fetching from yfinance
│   │   ├── data_manager.py      # Data caching and management
│   │   ├── stock_analyzer.py    # Analysis tools
│   │   ├── visualizer.py        # Plotting functions
│   │   ├── strategies/          # Trading strategies
│   │   │   └── momentum.py
│   │   └── prediction/          # Price prediction
│   │       └── predictor.py
│   └── streamlit_app/           # Streamlit web interface
│       ├── Home.py              # Main dashboard
│       ├── pages/               # Additional pages
│       │   └── Technical_Analysis.py
│       ├── components/          # Reusable components
│       ├── utils/               # Utility functions
│       │   ├── config_manager.py
│       │   └── stock_utils.py
│       └── config/              # Configuration files
│           ├── config.yaml
│           └── dev.yaml
├── data/
│   ├── cache/                  # Cached stock data
│   └── plots/                  # Generated plots
├── tests/                      # Unit tests
└── requirements.txt            # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JustR3/stock_analysis_app.git
cd stock_analysis_app
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

```bash
# For development
export APP_ENV=dev
streamlit run src/streamlit_app/Home.py

# For production
export APP_ENV=prod
streamlit run src/streamlit_app/Home.py
```

**Getting Started**: The app opens with empty stock symbol input boxes. Simply enter any valid stock ticker symbol (e.g., AAPL, GOOGL, MSFT, TSLA) to begin analysis. The app supports both the main dashboard and technical analysis tabs.

### Basic Analysis (Python API)

```python
from src.single_stock_analysis.data_manager import StockDataManager
from src.single_stock_analysis.visualizer import StockVisualizer

# Initialize managers
data_manager = StockDataManager()
visualizer = StockVisualizer()

# Get stock data (uses cache if available)
data = data_manager.get_stock_data("AAPL")  # Replace "AAPL" with any stock symbol

# Generate plots
visualizer.plot_price_and_moving_averages(data, "AAPL")
visualizer.plot_returns_distribution(data, "AAPL", period='daily')
```

### Running Tests

```bash
python -m unittest discover tests
```

## Configuration

The app uses a flexible configuration system:
- Base configuration in `config.yaml`
- Environment-specific overrides in `dev.yaml` or `prod.yaml`
- Configurable settings for:
  - App appearance
  - Data caching
  - Technical indicators
  - Visualization options
  - API settings

## Data Caching

The app implements a smart caching system to optimize API usage:
- **Stock Data**: 4-hour cache during market hours, 24-hour cache outside market hours
- **Stock Info**: 6-hour cache during market hours, 24-hour cache outside market hours
- **Format**: Stock data cached in Parquet format for fast loading
- **Analysis**: Technical indicators and analysis parameters are pre-calculated and stored
- **Rate Limiting**: Built-in exponential backoff and retry mechanism for API rate limits
- **Manual Refresh**: Cache can be force-refreshed when needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Dependencies

**Core Dependencies:**
- **pandas** (>=1.3.0) - Data manipulation and analysis
- **numpy** (>=1.21.0) - Numerical computing
- **yfinance** (>=0.1.70) - Yahoo Finance API integration
- **streamlit** (>=1.24.0) - Web application framework
- **plotly** (>=5.13.0) - Interactive charting

**Data Science & ML:**
- **scikit-learn** (>=0.24.2) - Machine learning algorithms
- **tensorflow** (>=2.6.0) - Deep learning framework
- **keras** (>=2.6.0) - Neural network API
- **statsmodels** (>=0.12.2) - Statistical modeling

**Visualization:**
- **matplotlib** (>=3.4.2) - Static plotting
- **seaborn** (>=0.11.1) - Statistical visualization

**Technical Analysis:**
- **ta** (>=0.7.0) - Technical analysis indicators

**Utilities:**
- **requests** (>=2.26.0) - HTTP requests
- **beautifulsoup4** (>=4.9.3) - HTML parsing
- **pyyaml** (>=6.0.1) - YAML configuration
- **python-dotenv** (>=0.19.0) - Environment variables
- **pytest** (>=6.2.5) - Testing framework

See `requirements.txt` for complete list with specific versions. 