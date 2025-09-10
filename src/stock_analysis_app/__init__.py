"""Stock Analysis App - A comprehensive stock market analysis and prediction platform."""

__version__ = "1.0.0"
__author__ = "Stock Analysis Team"
__email__ = "team@stockanalysis.com"

from .core import StockAnalyzer, DataManager
from .features import FeatureEngineer
from .models import StockPredictor
from .viz import StockVisualizer

__all__ = [
    "StockAnalyzer",
    "DataManager",
    "FeatureEngineer",
    "StockPredictor",
    "StockVisualizer",
    "__version__",
]
