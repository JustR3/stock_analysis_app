"""
Feature Engineering EDA (Exploratory Data Analysis) Module
Provides comprehensive analysis for feature selection and optimization
"""
import logging
import warnings
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class FeatureEDA:
    """
    Comprehensive EDA for feature engineering and selection
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize FeatureEDA with stock data

        Args:
            data (pd.DataFrame): Stock data with features and target
        """
        self.data = data.copy()
        self.features = None
        self.target = None
        self.correlation_matrix = None
        self.feature_importance = None
        self.scaler = StandardScaler()

        # Set style for plots
        plt.style.use("default")
        sns.set_palette("husl")

    def prepare_features_and_target(
        self, target_col: str = "Close", prediction_days: int = 1
    ) -> bool:
        """
        Prepare features and target variable

        Args:
            target_col (str): Target column name
            prediction_days (int): Days ahead to predict

        Returns:
            bool: True if successful
        """
        try:
            df = self.data.copy()

            # Basic price features
            base_features = ["Open", "High", "Low", "Close", "Volume"]

            # Create lagged features
            df["Close_lag_1"] = df["Close"].shift(1)
            df["Close_lag_2"] = df["Close"].shift(2)
            lag_features = ["Close_lag_1", "Close_lag_2"]

            # Technical indicator features
            technical_features = []
            if "daily_return" in df.columns:
                technical_features.append("daily_return")
            if "volatility" in df.columns:
                technical_features.append("volatility")

            # Moving average features
            ma_features = [col for col in df.columns if col.startswith("MA_")]
            technical_features.extend(ma_features)

            # RSI features
            rsi_features = [col for col in df.columns if col.startswith("RSI")]
            technical_features.extend(rsi_features)

            # MACD features
            macd_features = [col for col in df.columns if col.startswith("MACD")]
            technical_features.extend(macd_features)

            # Bollinger Band features
            bb_features = [col for col in df.columns if col.startswith("BB_")]
            technical_features.extend(bb_features)

            # Combine all features
            all_features = base_features + lag_features + technical_features

            # Ensure all features exist
            available_features = [col for col in all_features if col in df.columns]

            # Create target variable (future price)
            df["target"] = df[target_col].shift(-prediction_days)

            # Remove rows with NaN values
            df_clean = df[available_features + ["target"]].dropna()

            if len(df_clean) < 50:
                logger.warning("Insufficient data for analysis")
                return False

            self.features = df_clean[available_features]
            self.target = df_clean["target"]
            self.feature_names = available_features

            logger.info(
                f"Prepared {len(available_features)} features "
                f"for {len(df_clean)} samples"
            )
            return True

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return False

    def analyze_correlations(self, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Analyze feature correlations

        Args:
            threshold (float): Correlation threshold for high correlation detection

        Returns:
            Dict[str, Any]: Correlation analysis results
        """
        try:
            # Calculate correlation matrix
            corr_matrix = self.features.corr()

            # Find highly correlated features
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = abs(corr_matrix.iloc[i, j])
                    if corr_value > threshold:
                        high_corr_pairs.append(
                            {
                                "feature1": corr_matrix.columns[i],
                                "feature2": corr_matrix.columns[j],
                                "correlation": corr_matrix.iloc[i, j],
                            }
                        )

            # Sort by absolute correlation
            high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

            self.correlation_matrix = corr_matrix

            return {
                "correlation_matrix": corr_matrix,
                "high_correlation_pairs": high_corr_pairs,
                "summary": {
                    "total_features": len(corr_matrix.columns),
                    "highly_correlated_pairs": len(high_corr_pairs),
                    "correlation_threshold": threshold,
                },
            }

        except Exception as e:
            logger.error(f"Error analyzing correlations: {str(e)}")
            return {}

    def analyze_feature_importance(self, n_estimators: int = 100) -> Dict[str, Any]:
        """
        Analyze feature importance using Random Forest

        Args:
            n_estimators (int): Number of trees in Random Forest

        Returns:
            Dict[str, Any]: Feature importance analysis results
        """
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(self.features)
            y = self.target.values

            # Train Random Forest
            rf = RandomForestRegressor(
                n_estimators=n_estimators, random_state=42, n_jobs=-1
            )
            rf.fit(X_scaled, y)

            # Get feature importance
            importance_scores = rf.feature_importances_

            # Create importance DataFrame
            importance_df = pd.DataFrame(
                {"feature": self.feature_names, "importance": importance_scores}
            ).sort_values("importance", ascending=False)

            # Calculate cumulative importance
            importance_df["cumulative_importance"] = importance_df[
                "importance"
            ].cumsum()

            # Cross-validation score
            cv_scores = cross_val_score(
                rf, X_scaled, y, cv=5, scoring="neg_mean_squared_error"
            )
            cv_rmse = np.sqrt(-cv_scores.mean())

            self.feature_importance = importance_df

            return {
                "importance_df": importance_df,
                "cv_rmse": cv_rmse,
                "cv_scores": cv_scores,
                "top_features": importance_df.head(10).to_dict("records"),
                "model": rf,
            }

        except Exception as e:
            logger.error(f"Error analyzing feature importance: {str(e)}")
            return {}

    def select_optimal_features(
        self, method: str = "importance", k: int = 10
    ) -> Dict[str, Any]:
        """
        Select optimal features using various methods

        Args:
            method (str): Selection method ('importance', 'correlation', 'univariate')
            k (int): Number of features to select

        Returns:
            Dict[str, Any]: Selected features and analysis
        """
        try:
            selected_features = []

            if method == "importance" and self.feature_importance is not None:
                # Select top k features by importance
                selected_features = self.feature_importance.head(k)["feature"].tolist()

            elif method == "correlation" and self.correlation_matrix is not None:
                # Select features with low correlation to each other
                selected_features = self._select_uncorrelated_features(k)

            elif method == "univariate":
                # Univariate feature selection
                selector = SelectKBest(score_func=f_regression, k=k)
                selector.fit(self.features, self.target)
                selected_indices = selector.get_support(indices=True)
                selected_features = [self.feature_names[i] for i in selected_indices]

            return {
                "selected_features": selected_features,
                "method": method,
                "num_features": len(selected_features),
                "original_features": self.feature_names,
            }

        except Exception as e:
            logger.error(f"Error selecting optimal features: {str(e)}")
            return {}

    def _select_uncorrelated_features(self, k: int) -> List[str]:
        """
        Select features with low correlation to each other

        Args:
            k (int): Number of features to select

        Returns:
            List[str]: Selected feature names
        """
        try:
            selected = []
            remaining = self.feature_names.copy()

            while len(selected) < k and remaining:
                # Select feature with highest importance from remaining
                if self.feature_importance is not None:
                    candidates = [
                        f
                        for f in remaining
                        if f in self.feature_importance["feature"].values
                    ]
                    if candidates:
                        best_feature = self.feature_importance[
                            self.feature_importance["feature"].isin(candidates)
                        ].iloc[0]["feature"]
                    else:
                        best_feature = remaining[0]
                else:
                    best_feature = remaining[0]

                selected.append(best_feature)
                remaining.remove(best_feature)

                # Remove highly correlated features
                if self.correlation_matrix is not None:
                    correlations = abs(self.correlation_matrix[best_feature])
                    to_remove = correlations[correlations > 0.8].index.tolist()
                    remaining = [f for f in remaining if f not in to_remove]

            return selected

        except Exception as e:
            logger.error(f"Error selecting uncorrelated features: {str(e)}")
            return []

    def plot_correlation_matrix(
        self, figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Plot correlation matrix heatmap

        Args:
            figsize (Tuple[int, int]): Figure size

        Returns:
            plt.Figure: Correlation matrix plot
        """
        try:
            if self.correlation_matrix is None:
                logger.warning(
                    "No correlation matrix available. Run analyze_correlations() first."
                )
                return None

            fig, ax = plt.subplots(figsize=figsize)

            # Create mask for upper triangle
            mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))

            # Plot heatmap
            sns.heatmap(
                self.correlation_matrix,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
            )

            plt.title("Feature Correlation Matrix", fontsize=16, pad=20)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()

            return fig

        except Exception as e:
            logger.error(f"Error plotting correlation matrix: {str(e)}")
            return None

    def plot_feature_importance(
        self, top_n: int = 20, figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot feature importance bar chart

        Args:
            top_n (int): Number of top features to show
            figsize (Tuple[int, int]): Figure size

        Returns:
            plt.Figure: Feature importance plot
        """
        try:
            if self.feature_importance is None:
                logger.warning(
                    "No feature importance data available. "
                    "Run analyze_feature_importance() first."
                )
                return None

            fig, ax = plt.subplots(figsize=figsize)

            # Plot top N features
            top_features = self.feature_importance.head(top_n)

            bars = ax.barh(
                range(len(top_features)),
                top_features["importance"],
                color="skyblue",
                alpha=0.8,
            )

            # Add value labels
            for i, (idx, row) in enumerate(top_features.iterrows()):
                ax.text(row["importance"] + 0.001, i, ".3f", va="center", fontsize=10)

            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features["feature"])
            ax.set_xlabel("Feature Importance")
            ax.set_title(f"Top {top_n} Feature Importance", fontsize=16, pad=20)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            return None

    def generate_eda_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive EDA report

        Returns:
            Dict[str, Any]: Complete EDA report
        """
        try:
            report = {
                "dataset_info": {
                    "total_samples": len(self.data),
                    "total_features": len(self.feature_names)
                    if self.feature_names
                    else 0,
                    "feature_names": self.feature_names,
                    "date_range": f"{self.data.index.min()} to {self.data.index.max()}"
                    if len(self.data) > 0
                    else None,
                },
                "correlation_analysis": self.analyze_correlations(),
                "feature_importance_analysis": self.analyze_feature_importance(),
                "recommendations": self._generate_recommendations(),
            }

            return report

        except Exception as e:
            logger.error(f"Error generating EDA report: {str(e)}")
            return {}

    def _generate_recommendations(self) -> Dict[str, Any]:
        """
        Generate feature engineering recommendations

        Returns:
            Dict[str, Any]: Recommendations for feature selection
        """
        try:
            recommendations = {
                "suggested_features": [],
                "features_to_remove": [],
                "correlation_issues": [],
                "importance_threshold": 0.01,
            }

            if self.feature_importance is not None:
                # Suggest keeping features with importance > 1%
                important_features = self.feature_importance[
                    self.feature_importance["importance"] > 0.01
                ]["feature"].tolist()

                recommendations["suggested_features"] = important_features
                recommendations["features_to_remove"] = [
                    f for f in self.feature_names if f not in important_features
                ]

            if self.correlation_matrix is not None:
                # Identify highly correlated feature pairs
                corr_pairs = []
                for i in range(len(self.correlation_matrix.columns)):
                    for j in range(i + 1, len(self.correlation_matrix.columns)):
                        corr_value = abs(self.correlation_matrix.iloc[i, j])
                        if corr_value > 0.8:
                            corr_pairs.append(
                                {
                                    "features": [
                                        self.correlation_matrix.columns[i],
                                        self.correlation_matrix.columns[j],
                                    ],
                                    "correlation": self.correlation_matrix.iloc[i, j],
                                }
                            )

                recommendations["correlation_issues"] = corr_pairs

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {}
