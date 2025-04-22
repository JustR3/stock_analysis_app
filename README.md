# Stock Analysis App

A Python application for stock market analysis, visualization, and prediction. The app provides tools for collecting, analyzing, and visualizing stock data, with a focus on single stock analysis.

## Features

- **Data Collection**: Fetch historical stock data using yfinance
- **Data Management**: Efficient caching system for stock data
- **Analysis Tools**:
  - Returns calculation (daily, weekly, monthly)
  - Volatility analysis
  - Moving averages
  - Summary statistics
- **Visualization**:
  - Price and moving averages plots
  - Returns distribution analysis
  - Professional seaborn styling
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
│   └── single_stock_analysis/
│       ├── data_collector.py    # Data fetching from yfinance
│       ├── data_manager.py      # Data caching and management
│       ├── stock_analyzer.py    # Analysis tools
│       ├── visualizer.py        # Plotting functions
│       ├── example.py           # Usage examples
│       ├── strategies/          # Trading strategies
│       │   └── momentum.py
│       └── prediction/          # Price prediction
│           └── predictor.py
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

### Basic Analysis

```python
from src.single_stock_analysis.data_manager import StockDataManager
from src.single_stock_analysis.visualizer import StockVisualizer

# Initialize managers
data_manager = StockDataManager()
visualizer = StockVisualizer()

# Get stock data (uses cache if available)
data = data_manager.get_stock_data("AAPL")

# Generate plots
visualizer.plot_price_and_moving_averages(data, "AAPL")
visualizer.plot_returns_distribution(data, "AAPL", period='daily')
```

### Running Tests

```bash
python -m unittest discover tests
```

## Data Caching

The app implements an efficient caching system:
- Stock data is cached in Parquet format
- Cache is valid for 1 day
- Analysis parameters are pre-calculated and stored
- Cache can be force-refreshed if needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Dependencies

- pandas
- numpy
- yfinance
- seaborn
- matplotlib
- scipy

See `requirements.txt` for specific versions. 