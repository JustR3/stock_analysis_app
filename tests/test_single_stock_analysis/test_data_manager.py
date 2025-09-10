"""
Tests for the data manager module.
"""
import unittest

import pandas as pd

from src.single_stock_analysis.data_manager import StockDataManager


class TestStockDataManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.manager = StockDataManager()

    def test_get_stock_data(self):
        """Test getting stock data."""
        data = self.manager.get_stock_data("AAPL")
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertIn("Close", data.columns)
        self.assertIn("daily_return", data.columns)
        self.assertIn("volatility", data.columns)


if __name__ == "__main__":
    unittest.main()
