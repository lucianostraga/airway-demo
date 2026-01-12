"""
Forecasting Models for ULD Demand Prediction

Implements the ensemble architecture from the mathematical framework:
1. LightGBM - Gradient boosting for tabular features
2. Prophet-style decomposition - Trend + Seasonality
3. ARIMA - Short-term autoregressive
4. Meta-learner - Stacking combination

Each base learner is trained to predict quantiles for uncertainty.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np
import pandas as pd


class BaseForecaster(Protocol):
    """Protocol for base forecasting models."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseForecaster":
        """Fit the model to training data."""
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point predictions."""
        ...

    def predict_quantiles(
        self, X: pd.DataFrame, quantiles: list[float]
    ) -> dict[float, np.ndarray]:
        """Generate quantile predictions for uncertainty."""
        ...


@dataclass
class LightGBMConfig:
    """Configuration for LightGBM forecaster."""

    objective: str = "regression"  # or 'quantile'
    num_leaves: int = 31
    learning_rate: float = 0.05
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    min_child_samples: int = 20
    lambda_l1: float = 0.1
    lambda_l2: float = 0.1
    n_estimators: int = 1000
    early_stopping_rounds: int = 50
    verbose: int = -1


class LightGBMForecaster:
    """
    LightGBM-based forecaster with quantile regression support.

    For point forecasts: Uses squared error loss
    For quantiles: Uses pinball loss L_alpha(y, y_hat) = (y - y_hat) * (alpha - I[y < y_hat])

    The gradient boosting ensemble learns:
    f(x) = sum_{m=1}^{M} gamma_m * h_m(x)

    where h_m is a regression tree and gamma_m is the step size.
    """

    def __init__(self, config: Optional[LightGBMConfig] = None):
        self.config = config or LightGBMConfig()
        self._model = None
        self._quantile_models: dict[float, object] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LightGBMForecaster":
        """
        Fit the LightGBM model.

        Uses early stopping if validation data is provided.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required: pip install lightgbm")

        params = {
            "objective": self.config.objective,
            "num_leaves": self.config.num_leaves,
            "learning_rate": self.config.learning_rate,
            "feature_fraction": self.config.feature_fraction,
            "bagging_fraction": self.config.bagging_fraction,
            "bagging_freq": self.config.bagging_freq,
            "min_child_samples": self.config.min_child_samples,
            "lambda_l1": self.config.lambda_l1,
            "lambda_l2": self.config.lambda_l2,
            "verbose": self.config.verbose,
        }

        train_set = lgb.Dataset(X, label=y)

        callbacks = []
        valid_sets = [train_set]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
            valid_sets.append(val_set)
            valid_names.append("valid")
            callbacks.append(
                lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)
            )

        self._model = lgb.train(
            params,
            train_set,
            num_boost_round=self.config.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks if callbacks else None,
        )

        return self

    def fit_quantile(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        quantile: float,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LightGBMForecaster":
        """
        Fit a quantile regression model.

        Pinball loss: L_alpha(y, y_hat) = (y - y_hat) * (alpha - I[y < y_hat])

        The quantile model estimates the conditional quantile Q_alpha(Y|X).
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required: pip install lightgbm")

        params = {
            "objective": "quantile",
            "alpha": quantile,
            "num_leaves": self.config.num_leaves,
            "learning_rate": self.config.learning_rate,
            "feature_fraction": self.config.feature_fraction,
            "bagging_fraction": self.config.bagging_fraction,
            "bagging_freq": self.config.bagging_freq,
            "min_child_samples": self.config.min_child_samples,
            "lambda_l1": self.config.lambda_l1,
            "lambda_l2": self.config.lambda_l2,
            "verbose": self.config.verbose,
        }

        train_set = lgb.Dataset(X, label=y)

        callbacks = []
        valid_sets = [train_set]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
            valid_sets.append(val_set)
            valid_names.append("valid")
            callbacks.append(
                lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)
            )

        model = lgb.train(
            params,
            train_set,
            num_boost_round=self.config.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks if callbacks else None,
        )

        self._quantile_models[quantile] = model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point predictions."""
        if self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._model.predict(X)

    def predict_quantiles(
        self, X: pd.DataFrame, quantiles: list[float]
    ) -> dict[float, np.ndarray]:
        """Generate quantile predictions."""
        results = {}
        for q in quantiles:
            if q in self._quantile_models:
                results[q] = self._quantile_models[q].predict(X)
            else:
                raise ValueError(f"Quantile {q} model not fitted. Call fit_quantile() first.")
        return results

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if self._model is None:
            raise ValueError("Model not fitted.")
        return pd.Series(
            self._model.feature_importance(importance_type="gain"),
            index=self._model.feature_name(),
        ).sort_values(ascending=False)


@dataclass
class ProphetStyleConfig:
    """Configuration for Prophet-style decomposition."""

    growth: str = "linear"  # or 'logistic'
    seasonality_mode: str = "multiplicative"  # or 'additive'
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = False
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    interval_width: float = 0.80


class ProphetForecaster:
    """
    Prophet-style structural time series forecaster.

    Decomposes time series as:
    y(t) = g(t) + s(t) + h(t) + epsilon

    where:
    - g(t): Trend (piecewise linear or logistic growth)
    - s(t): Seasonal (Fourier series)
    - h(t): Holiday/event effects
    - epsilon: Noise

    Uncertainty is quantified via Bayesian estimation.
    """

    def __init__(self, config: Optional[ProphetStyleConfig] = None):
        self.config = config or ProphetStyleConfig()
        self._model = None

    def fit(
        self,
        df: pd.DataFrame,
        date_col: str = "ds",
        target_col: str = "y",
        regressors: Optional[list[str]] = None,
    ) -> "ProphetForecaster":
        """
        Fit Prophet model.

        Args:
            df: DataFrame with date and target columns
            date_col: Name of date column
            target_col: Name of target column
            regressors: Optional list of additional regressor columns
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("prophet is required: pip install prophet")

        self._model = Prophet(
            growth=self.config.growth,
            seasonality_mode=self.config.seasonality_mode,
            yearly_seasonality=self.config.yearly_seasonality,
            weekly_seasonality=self.config.weekly_seasonality,
            daily_seasonality=self.config.daily_seasonality,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            seasonality_prior_scale=self.config.seasonality_prior_scale,
            interval_width=self.config.interval_width,
        )

        # Add regressors if provided
        if regressors:
            for reg in regressors:
                self._model.add_regressor(reg, standardize=True)

        # Prepare data in Prophet format
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df.columns = ["ds", "y"]

        if regressors:
            for reg in regressors:
                prophet_df[reg] = df[reg].values

        self._model.fit(prophet_df)
        return self

    def predict(
        self,
        df: pd.DataFrame,
        date_col: str = "ds",
        regressors: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate predictions with uncertainty intervals.

        Returns DataFrame with columns:
        - yhat: Point prediction
        - yhat_lower: Lower bound of prediction interval
        - yhat_upper: Upper bound of prediction interval
        - trend, seasonal: Decomposition components
        """
        if self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        future_df = df[[date_col]].copy()
        future_df.columns = ["ds"]

        if regressors:
            for reg in regressors:
                future_df[reg] = df[reg].values

        forecast = self._model.predict(future_df)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend", "weekly", "yearly"]]


@dataclass
class EnsembleConfig:
    """Configuration for ensemble forecaster."""

    base_models: list[str] = field(
        default_factory=lambda: ["lightgbm", "prophet"]
    )
    meta_learner: str = "ridge"  # or 'linear', 'lightgbm'
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])


class EnsembleForecaster:
    """
    Stacking ensemble combining multiple base learners.

    Architecture:
    1. Train base learners on training data
    2. Generate base predictions on validation fold
    3. Train meta-learner using base predictions as features
    4. For new data: get base predictions, feed to meta-learner

    Optimal weights are learned via cross-validation:
    w* = argmin_w sum_t L(y_t, sum_k w_k * f_k(X_t))

    For quantile regression, we use pinball loss.
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self._base_models: dict[str, BaseForecaster] = {}
        self._meta_model = None
        self._fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> "EnsembleForecaster":
        """
        Fit the stacking ensemble.

        Two-stage training:
        1. Fit base models on training data
        2. Fit meta-learner on validation predictions
        """
        # Stage 1: Fit base models
        if "lightgbm" in self.config.base_models:
            lgb_model = LightGBMForecaster()
            lgb_model.fit(X_train, y_train, X_val, y_val)
            self._base_models["lightgbm"] = lgb_model

        # Stage 2: Generate base predictions for meta-learning
        meta_features = self._get_meta_features(X_val)

        # Stage 3: Fit meta-learner
        from sklearn.linear_model import Ridge

        self._meta_model = Ridge(alpha=1.0)
        self._meta_model.fit(meta_features, y_val)

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions."""
        if not self._fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        meta_features = self._get_meta_features(X)
        return self._meta_model.predict(meta_features)

    def _get_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Get predictions from all base models as meta-features."""
        predictions = []
        for name, model in self._base_models.items():
            pred = model.predict(X)
            predictions.append(pred)
        return np.column_stack(predictions)

    def get_base_weights(self) -> dict[str, float]:
        """Get the learned weights for each base model."""
        if not self._fitted or self._meta_model is None:
            raise ValueError("Ensemble not fitted.")
        return dict(zip(self._base_models.keys(), self._meta_model.coef_))


class ULDForecaster:
    """
    Main forecaster class for ULD demand prediction.

    Orchestrates the full forecasting pipeline:
    1. Feature engineering
    2. Model training (base learners + ensemble)
    3. Uncertainty quantification
    4. Hierarchical reconciliation

    Supports multiple forecast horizons:
    - Day-of-ops (h=0): Real-time updates
    - 7-day: Weekly planning
    - 30-day: Monthly capacity planning
    """

    def __init__(
        self,
        horizon: int = 7,
        quantiles: list[float] = None,
    ):
        """
        Args:
            horizon: Forecast horizon in days
            quantiles: Quantiles for prediction intervals
        """
        self.horizon = horizon
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self._ensemble = None
        self._quantile_models: dict[float, LightGBMForecaster] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> "ULDForecaster":
        """
        Fit the forecaster.

        Trains:
        1. Point prediction ensemble
        2. Quantile models for each specified quantile
        """
        # Fit ensemble for point predictions
        self._ensemble = EnsembleForecaster()
        self._ensemble.fit(X_train, y_train, X_val, y_val)

        # Fit quantile models
        for q in self.quantiles:
            qr_model = LightGBMForecaster()
            qr_model.fit_quantile(X_train, y_train, q, X_val, y_val)
            self._quantile_models[q] = qr_model

        return self

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Generate predictions with uncertainty intervals.

        Returns:
            Dictionary with:
            - 'point': Point predictions
            - 'lower': Lower bound (e.g., 10th percentile)
            - 'upper': Upper bound (e.g., 90th percentile)
            - 'quantiles': All quantile predictions
        """
        if self._ensemble is None:
            raise ValueError("Forecaster not fitted. Call fit() first.")

        point = self._ensemble.predict(X)

        quantile_preds = {}
        for q, model in self._quantile_models.items():
            quantile_preds[q] = model.predict(X)

        # Get lower/upper from quantile predictions
        lower_q = min(self.quantiles)
        upper_q = max(self.quantiles)

        return {
            "point": point,
            "lower": quantile_preds.get(lower_q, point),
            "upper": quantile_preds.get(upper_q, point),
            "quantiles": quantile_preds,
        }
