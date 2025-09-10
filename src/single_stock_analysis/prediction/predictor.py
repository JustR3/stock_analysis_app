"""
Stock price prediction implementation with multiple ML models.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, data: pd.DataFrame, model_dir: str = "models"):
        """
        Initialize the stock predictor.

        Args:
            data (pd.DataFrame): Stock data with analysis parameters
            model_dir (str): Directory to save trained models
        """
        self.data = data.copy()
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.target_name = 'Close'

        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize different ML models for prediction."""
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        }

        logger.info(f"Initialized {len(self.models)} ML models")

    def prepare_features(self, prediction_days: int = 5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features for prediction using technical indicators.

        Args:
            prediction_days (int): Number of days ahead to predict

        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Features, targets, and feature names
        """
        try:
            # Clean data and ensure we have required columns
            df = self.data.copy()

            # Basic price features
            base_features = ['Open', 'High', 'Low', 'Close', 'Volume']

            # Technical indicator features (if available)
            technical_features = []
            if 'daily_return' in df.columns:
                technical_features.append('daily_return')
            if 'volatility' in df.columns:
                technical_features.append('volatility')

            # Moving average features
            ma_features = [col for col in df.columns if col.startswith('MA_')]
            technical_features.extend(ma_features)

            # RSI features
            rsi_features = [col for col in df.columns if col.startswith('RSI')]
            technical_features.extend(rsi_features)

            # MACD features
            macd_features = [col for col in df.columns if col.startswith('MACD')]
            technical_features.extend(macd_features)

            # Bollinger Band features
            bb_features = [col for col in df.columns if col.startswith('BB_')]
            technical_features.extend(bb_features)

            # Combine all features
            all_features = base_features + technical_features

            # Ensure all features exist
            available_features = [col for col in all_features if col in df.columns]

            if len(available_features) < 2:
                logger.warning("Insufficient features available for prediction")
                return np.array([]), np.array([]), []

            # Create target variable (future price)
            df['target'] = df[self.target_name].shift(-prediction_days)

            # Remove rows with NaN values
            df_clean = df[available_features + ['target']].dropna()

            if len(df_clean) < 50:  # Minimum sample size
                logger.warning("Insufficient data for training")
                return np.array([]), np.array([]), []

            # Prepare features and target
            X = df_clean[available_features].values
            y = df_clean['target'].values

            logger.info(f"Prepared {len(available_features)} features for {len(X)} samples")

            return X, y, available_features

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return np.array([]), np.array([]), []

    def train_model(self, model_name: str = 'random_forest', test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the specified ML model.

        Args:
            model_name (str): Name of the model to train
            test_size (float): Proportion of data to use for testing

        Returns:
            Dict[str, Any]: Training results and metrics
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found")

            # Prepare features
            X, y, feature_names = self.prepare_features()

            if len(X) == 0 or len(y) == 0:
                raise ValueError("No valid data for training")

            # Split data (time series split)
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            logger.info(f"Training {model_name} model...")
            model = self.models[model_name]
            model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)

            # Calculate metrics
            metrics = {
                'train_mse': mean_squared_error(y_train, y_pred_train),
                'test_mse': mean_squared_error(y_test, y_pred_test),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test)
            }

            # Save model and scaler
            self._save_model(model_name, model, scaler, feature_names)

            # Log results
            logger.info(f"Model {model_name} trained successfully")
            logger.info(f"Test R² Score: {metrics['test_r2']:.4f}")
            logger.info(f"Test MSE: {metrics['test_mse']:.4f}")

            return {
                'model_name': model_name,
                'metrics': metrics,
                'feature_names': feature_names,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }

        except Exception as e:
            logger.error(f"Error training model {model_name}: {str(e)}")
            return {'error': str(e)}

    def predict(self, days_ahead: int = 5, model_name: str = 'random_forest') -> pd.Series:
        """
        Make predictions for future prices.

        Args:
            days_ahead (int): Number of days to predict ahead
            model_name (str): Name of the trained model to use

        Returns:
            pd.Series: Predicted prices with dates as index
        """
        try:
            # Load model and scaler
            model, scaler, feature_names = self._load_model(model_name)

            if model is None:
                raise ValueError(f"Model '{model_name}' not found. Please train the model first.")

            # Prepare latest features for prediction
            X, _, _ = self.prepare_features(days_ahead)

            if len(X) == 0:
                raise ValueError("No valid data for prediction")

            # Use the most recent data point
            latest_features = X[-1].reshape(1, -1)

            # Scale features
            latest_features_scaled = scaler.transform(latest_features)

            # Make prediction
            prediction = model.predict(latest_features_scaled)[0]

            # Create prediction date
            last_date = self.data.index[-1]
            prediction_date = last_date + timedelta(days=days_ahead)

            # Create result series
            result = pd.Series(
                [prediction],
                index=[prediction_date],
                name='predicted_price'
            )

            logger.info(f"Predicted price for {prediction_date.date()}: ${prediction:.2f}")

            return result

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return pd.Series()

    def evaluate_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all available models.

        Returns:
            Dict[str, Dict[str, Any]]: Evaluation results for each model
        """
        results = {}

        for model_name in self.models.keys():
            try:
                logger.info(f"Evaluating {model_name}...")
                result = self.train_model(model_name)
                results[model_name] = result
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}

        return results

    def _save_model(self, model_name: str, model: Any, scaler: Any, feature_names: List[str]):
        """Save trained model, scaler, and feature names."""
        try:
            model_path = os.path.join(self.model_dir, f"{model_name}_model.pkl")
            scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
            features_path = os.path.join(self.model_dir, f"{model_name}_features.pkl")

            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            joblib.dump(feature_names, features_path)

            logger.info(f"Saved {model_name} model to {self.model_dir}")

        except Exception as e:
            logger.error(f"Error saving model {model_name}: {str(e)}")

    def _load_model(self, model_name: str) -> Tuple[Any, Any, List[str]]:
        """Load trained model, scaler, and feature names."""
        try:
            model_path = os.path.join(self.model_dir, f"{model_name}_model.pkl")
            scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
            features_path = os.path.join(self.model_dir, f"{model_name}_features.pkl")

            if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
                logger.warning(f"Model files for {model_name} not found")
                return None, None, []

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            feature_names = joblib.load(features_path)

            logger.info(f"Loaded {model_name} model from {self.model_dir}")
            return model, scaler, feature_names

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None, None, []

    def get_available_models(self) -> List[str]:
        """Get list of available (trained) models."""
        available_models = []
        for model_name in self.models.keys():
            model_path = os.path.join(self.model_dir, f"{model_name}_model.pkl")
            if os.path.exists(model_path):
                available_models.append(model_name)
        return available_models 