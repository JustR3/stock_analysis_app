"""
Advanced Time Series Analysis and Stochastic Calculus Models
Provides sophisticated forecasting and risk analysis capabilities
"""
import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import arch

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model

# Scipy for stochastic calculus
from scipy import stats
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Time series and statistical libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)


class TimeSeriesAnalyzer:
    """
    Advanced time series analysis with stochastic calculus
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with stock data

        Args:
            data (pd.DataFrame): Stock price data with datetime index
        """
        self.data = data.copy()
        self.returns = None
        self.volatility = None
        self.models = {}
        self.forecasts = {}

        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        # Calculate returns and volatility
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for time series analysis"""
        try:
            # Calculate returns
            self.returns = np.log(
                self.data["Close"] / self.data["Close"].shift(1)
            ).dropna()

            # Calculate rolling volatility
            self.volatility = self.returns.rolling(window=30).std() * np.sqrt(
                252
            )  # Annualized

            logger.info("Data preparation completed")

        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")

    def test_stationarity(self, data: pd.Series) -> Dict[str, Any]:
        """
        Test for stationarity using ADF and KPSS tests

        Args:
            data (pd.Series): Time series data

        Returns:
            Dict[str, Any]: Stationarity test results
        """
        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(data.dropna(), autolag="AIC")
            adf_statistic = adf_result[0]
            adf_pvalue = adf_result[1]
            adf_critical_values = adf_result[4]

            # KPSS test
            kpss_result = kpss(data.dropna(), regression="c")
            kpss_statistic = kpss_result[0]
            kpss_pvalue = kpss_result[1]
            kpss_critical_values = kpss_result[3]

            return {
                "adf": {
                    "statistic": adf_statistic,
                    "p_value": adf_pvalue,
                    "critical_values": adf_critical_values,
                    "stationary": adf_pvalue < 0.05,
                },
                "kpss": {
                    "statistic": kpss_statistic,
                    "p_value": kpss_pvalue,
                    "critical_values": kpss_critical_values,
                    "stationary": kpss_pvalue >= 0.05,
                },
                "conclusion": self._interpret_stationarity_tests(
                    adf_pvalue, kpss_pvalue
                ),
            }

        except Exception as e:
            logger.error(f"Error testing stationarity: {str(e)}")
            return {}

    def _interpret_stationarity_tests(self, adf_p: float, kpss_p: float) -> str:
        """Interpret stationarity test results"""
        if adf_p < 0.05 and kpss_p >= 0.05:
            return "Stationary"
        elif adf_p >= 0.05 and kpss_p < 0.05:
            return "Non-stationary"
        elif adf_p < 0.05 and kpss_p < 0.05:
            return "Difference stationary"
        else:
            return "Inconclusive"


class ARIMAModel:
    """
    ARIMA/SARIMA time series forecasting model
    """

    def __init__(self, data: pd.Series):
        """
        Initialize ARIMA model

        Args:
            data (pd.Series): Time series data
        """
        self.data = data.copy()
        self.model = None
        self.fitted_model = None
        self.forecast = None
        self.residuals = None

    def fit_arima(
        self,
        order: Tuple[int, int, int] = (5, 1, 0),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Fit ARIMA or SARIMA model

        Args:
            order (Tuple[int, int, int]): (p, d, q) parameters
            seasonal_order (Optional[Tuple]): Seasonal parameters (P, D, Q, s)

        Returns:
            Dict[str, Any]: Model fit results
        """
        try:
            if seasonal_order:
                self.model = SARIMAX(
                    self.data, order=order, seasonal_order=seasonal_order
                )
            else:
                self.model = ARIMA(self.data, order=order)

            self.fitted_model = self.model.fit()
            self.residuals = self.fitted_model.resid

            # Model diagnostics
            diagnostics = {
                "aic": self.fitted_model.aic,
                "bic": self.fitted_model.bic,
                "hqic": self.fitted_model.hqic,
                "log_likelihood": self.fitted_model.llf,
                "converged": self.fitted_model.mle_retvals["converged"],
            }

            logger.info(
                f"ARIMA model fitted successfully. AIC: {diagnostics['aic']:.2f}"
            )
            return diagnostics

        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            return {}

    def forecast(self, steps: int = 30, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Generate forecasts with confidence intervals

        Args:
            steps (int): Number of steps to forecast
            alpha (float): Significance level for confidence intervals

        Returns:
            Dict[str, Any]: Forecast results
        """
        try:
            if self.fitted_model is None:
                raise ValueError("Model must be fitted before forecasting")

            forecast_result = self.fitted_model.get_forecast(steps=steps)

            forecast_mean = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int(alpha=alpha)

            forecast_dates = pd.date_range(
                start=self.data.index[-1] + pd.Timedelta(days=1),
                periods=steps,
                freq="B",  # Business days
            )

            forecast_df = pd.DataFrame(
                {
                    "forecast": forecast_mean.values,
                    "lower_ci": forecast_ci.iloc[:, 0].values,
                    "upper_ci": forecast_ci.iloc[:, 1].values,
                },
                index=forecast_dates,
            )

            self.forecast = forecast_df

            return {
                "forecast": forecast_df,
                "mean_forecast": forecast_mean.values[-1],
                "confidence_interval": forecast_ci.iloc[-1, :].values,
            }

        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            return {}


class StochasticCalculus:
    """
    Stochastic calculus models for financial time series
    """

    def __init__(self, data: pd.Series):
        """
        Initialize stochastic calculus analyzer

        Args:
            data (pd.Series): Price time series data
        """
        self.data = data.copy()
        self.returns = np.log(data / data.shift(1)).dropna()
        self.drift = None
        self.volatility = None

        self._estimate_parameters()

    def _estimate_parameters(self):
        """Estimate drift and volatility parameters"""
        try:
            # Estimate drift (mean return)
            self.drift = self.returns.mean() * 252  # Annualized

            # Estimate volatility (standard deviation)
            self.volatility = self.returns.std() * np.sqrt(252)  # Annualized

            logger.info(".4f")

        except Exception as e:
            logger.error(f"Error estimating parameters: {str(e)}")

    def simulate_geometric_brownian_motion(
        self,
        initial_price: float,
        time_horizon: float = 1.0,
        num_simulations: int = 1000,
        num_steps: int = 252,
    ) -> Dict[str, Any]:
        """
        Simulate Geometric Brownian Motion paths

        Args:
            initial_price (float): Starting price
            time_horizon (float): Time horizon in years
            num_simulations (int): Number of simulation paths
            num_steps (int): Number of time steps per simulation

        Returns:
            Dict[str, Any]: Simulation results
        """
        try:
            dt = time_horizon / num_steps

            # Generate Brownian motion increments
            dW = np.random.normal(0, np.sqrt(dt), (num_simulations, num_steps))

            # Initialize price paths
            price_paths = np.zeros((num_simulations, num_steps + 1))
            price_paths[:, 0] = initial_price

            # Simulate GBM paths
            for t in range(1, num_steps + 1):
                price_paths[:, t] = price_paths[:, t - 1] * np.exp(
                    (self.drift - 0.5 * self.volatility**2) * dt
                    + self.volatility * dW[:, t - 1]
                )

            # Calculate statistics
            final_prices = price_paths[:, -1]
            mean_final_price = np.mean(final_prices)
            std_final_price = np.std(final_prices)

            # Calculate quantiles
            quantiles = np.percentile(final_prices, [5, 25, 50, 75, 95])

            return {
                "price_paths": price_paths,
                "final_prices": final_prices,
                "statistics": {
                    "mean": mean_final_price,
                    "std": std_final_price,
                    "median": quantiles[2],
                    "quantile_5": quantiles[0],
                    "quantile_95": quantiles[4],
                },
                "parameters": {
                    "drift": self.drift,
                    "volatility": self.volatility,
                    "initial_price": initial_price,
                    "time_horizon": time_horizon,
                },
            }

        except Exception as e:
            logger.error(f"Error simulating GBM: {str(e)}")
            return {}

    def black_scholes_price(
        self, S: float, K: float, T: float, r: float = 0.05, option_type: str = "call"
    ) -> float:
        """
        Calculate Black-Scholes option price

        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration (years)
            r (float): Risk-free rate
            option_type (str): 'call' or 'put'

        Returns:
            float: Option price
        """
        try:
            d1 = (np.log(S / K) + (r + 0.5 * self.volatility**2) * T) / (
                self.volatility * np.sqrt(T)
            )
            d2 = d1 - self.volatility * np.sqrt(T)

            if option_type.lower() == "call":
                price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
            elif option_type.lower() == "put":
                price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(
                    -d1
                )
            else:
                raise ValueError("option_type must be 'call' or 'put'")

            return price

        except Exception as e:
            logger.error(f"Error calculating Black-Scholes price: {str(e)}")
            return 0.0


class GARCHModel:
    """
    GARCH volatility modeling
    """

    def __init__(self, returns: pd.Series):
        """
        Initialize GARCH model

        Args:
            returns (pd.Series): Return series for volatility modeling
        """
        self.returns = returns.copy()
        self.model = None
        self.fitted_model = None
        self.conditional_volatility = None

    def fit_garch(self, p: int = 1, q: int = 1, dist: str = "normal") -> Dict[str, Any]:
        """
        Fit GARCH(p,q) model

        Args:
            p (int): ARCH order
            q (int): GARCH order
            dist (str): Distribution assumption

        Returns:
            Dict[str, Any]: Model fit results
        """
        try:
            self.model = arch_model(self.returns, vol="Garch", p=p, q=q, dist=dist)
            self.fitted_model = self.model.fit(disp="off")

            # Extract conditional volatility
            self.conditional_volatility = self.fitted_model.conditional_volatility

            # Model diagnostics
            diagnostics = {
                "aic": self.fitted_model.aic,
                "bic": self.fitted_model.bic,
                "log_likelihood": self.fitted_model.log_likelihood,
                "converged": self.fitted_model.convergence_flag == 0,
                "parameters": dict(self.fitted_model.params),
            }

            logger.info(
                f"GARCH({p},{q}) model fitted successfully. AIC: {diagnostics['aic']:.2f}"
            )
            return diagnostics

        except Exception as e:
            logger.error(f"Error fitting GARCH model: {str(e)}")
            return {}

    def forecast_volatility(self, horizon: int = 30) -> Dict[str, Any]:
        """
        Forecast future volatility

        Args:
            horizon (int): Forecast horizon

        Returns:
            Dict[str, Any]: Volatility forecast results
        """
        try:
            if self.fitted_model is None:
                raise ValueError("Model must be fitted before forecasting")

            # Generate volatility forecast
            vol_forecast = self.fitted_model.forecast(horizon=horizon)

            # Extract forecasted volatility
            forecasted_vol = (
                vol_forecast.variance.iloc[-1, :] ** 0.5
            )  # Take square root for std dev

            # Create forecast dates
            forecast_dates = pd.date_range(
                start=self.returns.index[-1] + pd.Timedelta(days=1),
                periods=horizon,
                freq="B",
            )

            forecast_df = pd.DataFrame(
                {"volatility_forecast": forecasted_vol.values}, index=forecast_dates
            )

            return {
                "volatility_forecast": forecast_df,
                "mean_forecast": forecasted_vol.mean(),
                "final_forecast": forecasted_vol.iloc[-1],
            }

        except Exception as e:
            logger.error(f"Error forecasting volatility: {str(e)}")
            return {}


class RiskMetrics:
    """
    Advanced risk metrics using stochastic methods
    """

    def __init__(self, returns: pd.Series):
        """
        Initialize risk metrics calculator

        Args:
            returns (pd.Series): Return series for risk analysis
        """
        self.returns = returns.copy()
        self.confidence_levels = [0.90, 0.95, 0.99]

    def calculate_var(
        self, confidence_level: float = 0.95, method: str = "historical"
    ) -> Dict[str, Any]:
        """
        Calculate Value at Risk using different methods

        Args:
            confidence_level (float): Confidence level (0-1)
            method (str): Calculation method ('historical', 'parametric', 'monte_carlo')

        Returns:
            Dict[str, Any]: VaR calculation results
        """
        try:
            if method == "historical":
                # Historical VaR
                var = np.percentile(self.returns, (1 - confidence_level) * 100)
                var_pct = var * 100  # Convert to percentage

            elif method == "parametric":
                # Parametric VaR (assuming normal distribution)
                mean_return = self.returns.mean()
                std_return = self.returns.std()
                z_score = stats.norm.ppf(1 - confidence_level)
                var = mean_return + z_score * std_return
                var_pct = var * 100

            elif method == "monte_carlo":
                # Monte Carlo VaR (simple simulation)
                num_simulations = 10000
                simulated_returns = np.random.normal(
                    self.returns.mean(), self.returns.std(), num_simulations
                )
                var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
                var_pct = var * 100

            else:
                raise ValueError(
                    "Method must be 'historical', 'parametric', or 'monte_carlo'"
                )

            return {
                "VaR": var,
                "VaR_percent": var_pct,
                "confidence_level": confidence_level,
                "method": method,
                "expected_shortfall": self._calculate_expected_shortfall(
                    var, confidence_level
                ),
            }

        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return {}

    def _calculate_expected_shortfall(
        self, var_threshold: float, confidence_level: float
    ) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        try:
            # Find returns below VaR threshold
            tail_returns = self.returns[self.returns <= var_threshold]

            if len(tail_returns) > 0:
                return tail_returns.mean()
            else:
                return var_threshold  # Fallback

        except Exception as e:
            logger.error(f"Error calculating Expected Shortfall: {str(e)}")
            return 0.0

    def calculate_max_drawdown(self) -> Dict[str, Any]:
        """
        Calculate maximum drawdown and drawdown statistics

        Returns:
            Dict[str, Any]: Maximum drawdown analysis
        """
        try:
            # Calculate cumulative returns
            cumulative = (1 + self.returns).cumprod()

            # Calculate running maximum
            running_max = cumulative.expanding().max()

            # Calculate drawdown
            drawdown = (cumulative - running_max) / running_max

            # Find maximum drawdown
            max_drawdown = drawdown.min()
            max_drawdown_date = drawdown.idxmin()

            # Calculate drawdown duration
            peak_date = running_max.loc[:max_drawdown_date].idxmax()
            drawdown_duration = (max_drawdown_date - peak_date).days

            return {
                "max_drawdown": max_drawdown,
                "max_drawdown_percent": max_drawdown * 100,
                "max_drawdown_date": max_drawdown_date,
                "peak_date": peak_date,
                "drawdown_duration_days": drawdown_duration,
                "current_drawdown": drawdown.iloc[-1],
            }

        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return {}


class AdvancedPredictor:
    """
    Advanced predictor combining multiple time series models
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize advanced predictor

        Args:
            data (pd.DataFrame): Stock data
        """
        self.data = data.copy()
        self.ts_analyzer = TimeSeriesAnalyzer(data)
        self.stochastic = StochasticCalculus(data["Close"])
        self.risk_metrics = RiskMetrics(self.ts_analyzer.returns)

        # Model storage
        self.arima_model = None
        self.garch_model = None
        self.predictions = {}

    def fit_models(self) -> Dict[str, Any]:
        """
        Fit all available models

        Returns:
            Dict[str, Any]: Model fitting results
        """
        results = {}

        try:
            # Fit ARIMA model
            logger.info("Fitting ARIMA model...")
            arima = ARIMAModel(self.data["Close"])
            arima_results = arima.fit_arima(order=(2, 1, 2))
            self.arima_model = arima
            results["arima"] = arima_results

            # Fit GARCH model
            logger.info("Fitting GARCH model...")
            garch = GARCHModel(self.ts_analyzer.returns)
            garch_results = garch.fit_garch(p=1, q=1)
            self.garch_model = garch
            results["garch"] = garch_results

            logger.info("All models fitted successfully")
            return results

        except Exception as e:
            logger.error(f"Error fitting models: {str(e)}")
            return {}

    def generate_predictions(self, horizon: int = 30) -> Dict[str, Any]:
        """
        Generate predictions using multiple models

        Args:
            horizon (int): Prediction horizon in days

        Returns:
            Dict[str, Any]: Prediction results from all models
        """
        predictions = {}

        try:
            # ARIMA predictions
            if self.arima_model:
                logger.info("Generating ARIMA predictions...")
                arima_pred = self.arima_model.forecast(steps=horizon)
                predictions["arima"] = arima_pred

            # GARCH volatility predictions
            if self.garch_model:
                logger.info("Generating GARCH volatility predictions...")
                garch_pred = self.garch_model.forecast_volatility(horizon=horizon)
                predictions["garch"] = garch_pred

            # Stochastic simulations
            logger.info("Generating stochastic simulations...")
            gbm_sim = self.stochastic.simulate_geometric_brownian_motion(
                initial_price=self.data["Close"].iloc[-1],
                time_horizon=horizon / 252,  # Convert to years
                num_simulations=1000,
                num_steps=horizon,
            )
            predictions["gbm"] = gbm_sim

            # Risk metrics
            logger.info("Calculating risk metrics...")
            var_95 = self.risk_metrics.calculate_var(confidence_level=0.95)
            max_dd = self.risk_metrics.calculate_max_drawdown()

            predictions["risk_metrics"] = {"VaR_95": var_95, "max_drawdown": max_dd}

            self.predictions = predictions
            logger.info("All predictions generated successfully")
            return predictions

        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return {}

    def plot_predictions(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot comprehensive prediction results

        Args:
            figsize (Tuple[int, int]): Figure size

        Returns:
            plt.Figure: Prediction plots
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            fig.suptitle("Advanced Time Series Predictions", fontsize=16)

            # Plot 1: ARIMA Forecast
            if (
                "arima" in self.predictions
                and self.predictions["arima"].get("forecast") is not None
            ):
                forecast_df = self.predictions["arima"]["forecast"]
                axes[0, 0].plot(
                    self.data.index[-50:],
                    self.data["Close"].iloc[-50:],
                    label="Historical",
                    color="blue",
                )
                axes[0, 0].plot(
                    forecast_df.index,
                    forecast_df["forecast"],
                    label="ARIMA Forecast",
                    color="red",
                    linestyle="--",
                )
                axes[0, 0].fill_between(
                    forecast_df.index,
                    forecast_df["lower_ci"],
                    forecast_df["upper_ci"],
                    alpha=0.3,
                    color="red",
                )
                axes[0, 0].set_title("ARIMA Price Forecast")
                axes[0, 0].legend()
                axes[0, 0].tick_params(axis="x", rotation=45)

            # Plot 2: GBM Simulation
            if "gbm" in self.predictions and "price_paths" in self.predictions["gbm"]:
                price_paths = self.predictions["gbm"]["price_paths"]
                # Plot a subset of simulation paths
                for i in range(min(50, price_paths.shape[0])):
                    axes[0, 1].plot(price_paths[i, :], alpha=0.1, color="blue")
                axes[0, 1].axhline(
                    y=self.data["Close"].iloc[-1],
                    color="red",
                    linestyle="--",
                    label="Current Price",
                )
                axes[0, 1].set_title("GBM Price Simulations")
                axes[0, 1].legend()

            # Plot 3: Volatility Forecast
            if (
                "garch" in self.predictions
                and self.predictions["garch"].get("volatility_forecast") is not None
            ):
                vol_df = self.predictions["garch"]["volatility_forecast"]
                axes[1, 0].plot(
                    self.ts_analyzer.volatility.index[-50:],
                    self.ts_analyzer.volatility.iloc[-50:],
                    label="Historical Volatility",
                )
                axes[1, 0].plot(
                    vol_df.index,
                    vol_df["volatility_forecast"],
                    label="GARCH Forecast",
                    color="orange",
                )
                axes[1, 0].set_title("Volatility Forecast (GARCH)")
                axes[1, 0].legend()
                axes[1, 0].tick_params(axis="x", rotation=45)

            # Plot 4: Risk Metrics
            if "risk_metrics" in self.predictions:
                risk_data = self.predictions["risk_metrics"]
                if "VaR_95" in risk_data and "VaR_percent" in risk_data["VaR_95"]:
                    var_pct = risk_data["VaR_95"]["VaR_percent"]
                    axes[1, 1].bar(["VaR 95%"], [var_pct], color="red", alpha=0.7)
                    axes[1, 1].set_title("Risk Metrics")
                    axes[1, 1].set_ylabel("Daily Return (%)")
                    axes[1, 1].text(
                        0,
                        var_pct / 2,
                        ".2f",
                        ha="center",
                        va="center",
                        fontweight="bold",
                    )

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error plotting predictions: {str(e)}")
            return None
