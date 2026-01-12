"""
Uncertainty Quantification for ULD Demand Forecasting

Implements distribution-free prediction intervals using conformal prediction:
1. Split Conformal - Basic conformal with fixed width intervals
2. Conformalized Quantile Regression (CQR) - Adaptive interval width

Key property: Coverage guarantee
P(Y_new in [L, U]) >= 1 - alpha

for any distribution, without parametric assumptions.

References:
- Romano, Patterson, Candes (2019). Conformalized Quantile Regression. NeurIPS.
- Vovk, Gammerman, Shafer (2005). Algorithmic Learning in a Random World.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple

import numpy as np
import pandas as pd


class PointPredictor(Protocol):
    """Protocol for point prediction models."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PointPredictor":
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


class QuantilePredictor(Protocol):
    """Protocol for quantile prediction models."""

    def fit(self, X: pd.DataFrame, y: pd.Series, quantile: float) -> "QuantilePredictor":
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


@dataclass
class ConformalConfig:
    """Configuration for conformal prediction."""

    alpha: float = 0.1  # Miscoverage rate (1 - alpha = coverage)
    method: str = "cqr"  # 'split' or 'cqr'


class SplitConformal:
    """
    Split Conformal Prediction for distribution-free intervals.

    Algorithm:
    1. Split data into training and calibration sets
    2. Train model on training set
    3. Compute residuals on calibration set: R_i = |y_i - y_hat_i|
    4. Find quantile q of residuals at level (1-alpha)(n+1)/n
    5. Prediction interval: [y_hat - q, y_hat + q]

    Theorem (Coverage Guarantee):
    For exchangeable data, P(Y_new in [y_hat - q, y_hat + q]) >= 1 - alpha
    """

    def __init__(self, base_model: PointPredictor, alpha: float = 0.1):
        """
        Args:
            base_model: Any point prediction model
            alpha: Miscoverage rate (default 0.1 for 90% coverage)
        """
        self.base_model = base_model
        self.alpha = alpha
        self._quantile: Optional[float] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_calib: pd.DataFrame,
        y_calib: pd.Series,
    ) -> "SplitConformal":
        """
        Fit the conformal predictor.

        Steps:
        1. Train base model on training data
        2. Compute nonconformity scores on calibration data
        3. Find the (1-alpha) quantile of scores
        """
        # Step 1: Fit base model
        self.base_model.fit(X_train, y_train)

        # Step 2: Compute residuals on calibration set
        y_pred_calib = self.base_model.predict(X_calib)
        residuals = np.abs(y_calib.values - y_pred_calib)

        # Step 3: Find quantile with finite sample correction
        n = len(residuals)
        # Correction: ceil((n+1)(1-alpha)) / n ensures coverage >= 1-alpha
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        quantile_level = min(quantile_level, 1.0)  # Cap at 1.0

        self._quantile = np.quantile(residuals, quantile_level)
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with conformal intervals.

        Returns:
            Tuple of (point_prediction, lower_bound, upper_bound)
        """
        if self._quantile is None:
            raise ValueError("Conformal predictor not fitted. Call fit() first.")

        point = self.base_model.predict(X)
        lower = point - self._quantile
        upper = point + self._quantile

        return point, lower, upper

    @property
    def interval_width(self) -> float:
        """Width of the prediction interval (constant for split conformal)."""
        if self._quantile is None:
            raise ValueError("Conformal predictor not fitted.")
        return 2 * self._quantile


class ConformQR:
    """
    Conformalized Quantile Regression (CQR).

    Combines quantile regression with conformal prediction for:
    - Adaptive interval width (wider when uncertainty is high)
    - Coverage guarantee (without distributional assumptions)

    Algorithm:
    1. Train quantile models for alpha/2 and 1-alpha/2
    2. Define nonconformity score: R_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))
    3. Find quantile q of scores on calibration set
    4. Interval: [q_lo(x) - q, q_hi(x) + q]

    The interval adapts to local uncertainty while maintaining coverage.

    Reference:
    Romano, Patterson, Candes (2019). Conformalized Quantile Regression.
    """

    def __init__(
        self,
        model_lower: QuantilePredictor,
        model_upper: QuantilePredictor,
        alpha: float = 0.1,
    ):
        """
        Args:
            model_lower: Quantile model for lower bound (trained at alpha/2)
            model_upper: Quantile model for upper bound (trained at 1-alpha/2)
            alpha: Miscoverage rate
        """
        self.model_lower = model_lower
        self.model_upper = model_upper
        self.alpha = alpha
        self._correction: Optional[float] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_calib: pd.DataFrame,
        y_calib: pd.Series,
    ) -> "ConformQR":
        """
        Fit the CQR predictor.

        Steps:
        1. Train lower quantile model at alpha/2
        2. Train upper quantile model at 1-alpha/2
        3. Compute nonconformity scores on calibration set
        4. Find conformal correction factor
        """
        # Step 1 & 2: Train quantile models
        self.model_lower.fit(X_train, y_train, quantile=self.alpha / 2)
        self.model_upper.fit(X_train, y_train, quantile=1 - self.alpha / 2)

        # Step 3: Compute nonconformity scores on calibration set
        pred_lower = self.model_lower.predict(X_calib)
        pred_upper = self.model_upper.predict(X_calib)

        # Score: how much does the true value fall outside the predicted interval?
        scores = np.maximum(
            pred_lower - y_calib.values,  # Below lower bound
            y_calib.values - pred_upper,  # Above upper bound
        )

        # Step 4: Find conformal correction
        n = len(scores)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        quantile_level = min(quantile_level, 1.0)

        self._correction = np.quantile(scores, quantile_level)
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with conformalized intervals.

        Returns:
            Tuple of (point_prediction, lower_bound, upper_bound)
        """
        if self._correction is None:
            raise ValueError("CQR predictor not fitted. Call fit() first.")

        pred_lower = self.model_lower.predict(X)
        pred_upper = self.model_upper.predict(X)

        # Apply conformal correction
        lower = pred_lower - self._correction
        upper = pred_upper + self._correction

        # Point prediction is median of interval
        point = (lower + upper) / 2

        return point, lower, upper

    @property
    def correction_factor(self) -> float:
        """The conformal correction factor applied to intervals."""
        if self._correction is None:
            raise ValueError("CQR predictor not fitted.")
        return self._correction


class AdaptiveConformalPredictor:
    """
    Adaptive conformal prediction with local correction.

    Instead of a single global correction, estimates local correction
    based on feature similarity to calibration points.

    Uses kernel density estimation or nearest neighbors to weight
    calibration scores by similarity to the test point.
    """

    def __init__(
        self,
        base_model: PointPredictor,
        alpha: float = 0.1,
        k_neighbors: int = 100,
    ):
        """
        Args:
            base_model: Point prediction model
            alpha: Miscoverage rate
            k_neighbors: Number of neighbors for local correction
        """
        self.base_model = base_model
        self.alpha = alpha
        self.k_neighbors = k_neighbors
        self._X_calib: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_calib: pd.DataFrame,
        y_calib: pd.Series,
    ) -> "AdaptiveConformalPredictor":
        """Fit the adaptive conformal predictor."""
        # Fit base model
        self.base_model.fit(X_train, y_train)

        # Store calibration data for local correction
        self._X_calib = X_calib.values
        y_pred_calib = self.base_model.predict(X_calib)
        self._residuals = np.abs(y_calib.values - y_pred_calib)

        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with locally adaptive intervals.

        For each test point, finds k nearest calibration points and
        computes the local quantile of residuals.
        """
        if self._X_calib is None or self._residuals is None:
            raise ValueError("Predictor not fitted. Call fit() first.")

        from sklearn.neighbors import NearestNeighbors

        # Find k nearest neighbors for each test point
        nn = NearestNeighbors(n_neighbors=self.k_neighbors)
        nn.fit(self._X_calib)
        distances, indices = nn.kneighbors(X.values)

        point = self.base_model.predict(X)
        lower = np.zeros(len(X))
        upper = np.zeros(len(X))

        for i in range(len(X)):
            # Local residuals from k neighbors
            local_residuals = self._residuals[indices[i]]

            # Local quantile with finite sample correction
            n = len(local_residuals)
            quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            quantile_level = min(quantile_level, 1.0)

            local_q = np.quantile(local_residuals, quantile_level)
            lower[i] = point[i] - local_q
            upper[i] = point[i] + local_q

        return point, lower, upper


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """
    Pinball loss for quantile regression evaluation.

    L_alpha(y, y_hat) = (y - y_hat) * (alpha - I[y < y_hat])

    Properties:
    - alpha = 0.5: Equivalent to MAE (median regression)
    - alpha < 0.5: Penalizes over-prediction more
    - alpha > 0.5: Penalizes under-prediction more

    Args:
        y_true: True values
        y_pred: Predicted values
        alpha: Quantile level (0 < alpha < 1)

    Returns:
        Mean pinball loss
    """
    residuals = y_true - y_pred
    return np.mean(np.where(residuals >= 0, alpha * residuals, (alpha - 1) * residuals))


def interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """
    Interval Score for evaluating prediction intervals.

    IS_alpha(l, u, y) = (u - l) + (2/alpha)(l - y)_+ + (2/alpha)(y - u)_+

    Penalizes:
    - Wide intervals (first term)
    - Under-coverage on lower end (second term)
    - Under-coverage on upper end (third term)

    Lower is better.

    Args:
        y_true: True values
        lower: Lower bounds
        upper: Upper bounds
        alpha: Miscoverage rate (1 - coverage level)

    Returns:
        Mean interval score
    """
    width = upper - lower
    lower_violation = np.maximum(lower - y_true, 0)
    upper_violation = np.maximum(y_true - upper, 0)

    score = width + (2 / alpha) * lower_violation + (2 / alpha) * upper_violation
    return np.mean(score)


def winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """
    Winkler Score for prediction interval evaluation.

    Equivalent to interval score but uses different formulation.
    Commonly used in M-competitions.

    Args:
        y_true: True values
        lower: Lower bounds
        upper: Upper bounds
        alpha: Miscoverage rate

    Returns:
        Mean Winkler score
    """
    width = upper - lower
    in_interval = (y_true >= lower) & (y_true <= upper)

    score = np.where(
        in_interval,
        width,
        width + (2 / alpha) * np.where(y_true < lower, lower - y_true, y_true - upper),
    )
    return np.mean(score)


def calibration_check(
    y_true: np.ndarray,
    quantile_preds: dict[float, np.ndarray],
) -> dict[float, float]:
    """
    Check calibration of quantile predictions.

    For well-calibrated quantile predictions, the empirical coverage
    should match the nominal quantile level.

    E.g., for 90th percentile predictions, ~90% of true values
    should be below the prediction.

    Args:
        y_true: True values
        quantile_preds: Dict mapping quantile level to predictions

    Returns:
        Dict mapping quantile level to empirical coverage
    """
    results = {}
    for q, y_pred in quantile_preds.items():
        coverage = np.mean(y_true <= y_pred)
        results[q] = coverage
    return results
