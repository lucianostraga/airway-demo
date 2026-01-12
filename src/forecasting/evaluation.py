"""
Forecast Evaluation and Cross-Validation

Implements comprehensive evaluation methodology:
1. Point forecast metrics (MAPE, RMSE, MAE, Bias, MASE)
2. Probabilistic metrics (CRPS, Interval Score, Coverage)
3. Time series cross-validation
4. Statistical significance testing (Diebold-Mariano, MCS)

This goes beyond simple accuracy metrics to assess:
- Calibration: Are prediction intervals reliable?
- Sharpness: Are intervals as tight as possible while maintaining coverage?
- Skill: How much better than naive baselines?
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    # Point forecast metrics
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Squared Error
    mae: float  # Mean Absolute Error
    bias: float  # Mean Error (signed)
    mase: float  # Mean Absolute Scaled Error (vs naive)

    # Probabilistic metrics (if intervals provided)
    coverage: Optional[float] = None  # Actual coverage rate
    interval_width: Optional[float] = None  # Mean interval width
    crps: Optional[float] = None  # Continuous Ranked Probability Score
    interval_score: Optional[float] = None  # Winkler score

    # Skill scores
    skill_vs_naive: Optional[float] = None  # Improvement over seasonal naive

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "mape": self.mape,
            "rmse": self.rmse,
            "mae": self.mae,
            "bias": self.bias,
            "mase": self.mase,
            "coverage": self.coverage,
            "interval_width": self.interval_width,
            "crps": self.crps,
            "interval_score": self.interval_score,
            "skill_vs_naive": self.skill_vs_naive,
        }


class ForecastEvaluator:
    """
    Comprehensive forecast evaluation.

    Computes a portfolio of metrics to assess forecast quality:

    Point Forecast Metrics:
    - MAPE: Mean Absolute Percentage Error (scale-independent)
    - RMSE: Root Mean Squared Error (penalizes large errors)
    - MAE: Mean Absolute Error (robust to outliers)
    - Bias: Systematic over/under prediction
    - MASE: Mean Absolute Scaled Error (skill vs naive)

    Probabilistic Metrics:
    - Coverage: Fraction of actuals within prediction intervals
    - Sharpness: Average interval width (narrower is better)
    - CRPS: Continuous Ranked Probability Score (proper scoring rule)
    - Interval Score: Penalizes under-coverage and wide intervals
    """

    def __init__(self, seasonal_period: int = 7):
        """
        Args:
            seasonal_period: Period for seasonal naive baseline (default: 7 for weekly)
        """
        self.seasonal_period = seasonal_period

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_lower: Optional[np.ndarray] = None,
        y_upper: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        alpha: float = 0.1,
    ) -> EvaluationResult:
        """
        Evaluate forecast quality.

        Args:
            y_true: Actual values
            y_pred: Point predictions
            y_lower: Lower bound of prediction interval
            y_upper: Upper bound of prediction interval
            y_train: Training data (for MASE calculation)
            alpha: Nominal miscoverage rate

        Returns:
            EvaluationResult with all metrics
        """
        # Point forecast metrics
        mape = self._mape(y_true, y_pred)
        rmse = self._rmse(y_true, y_pred)
        mae = self._mae(y_true, y_pred)
        bias = self._bias(y_true, y_pred)

        # MASE (requires training data for scaling)
        if y_train is not None and len(y_train) > self.seasonal_period:
            mase = self._mase(y_true, y_pred, y_train)
        else:
            mase = np.nan

        # Probabilistic metrics (if intervals provided)
        coverage = None
        interval_width = None
        interval_score = None

        if y_lower is not None and y_upper is not None:
            coverage = self._coverage(y_true, y_lower, y_upper)
            interval_width = self._interval_width(y_lower, y_upper)
            interval_score = self._interval_score(y_true, y_lower, y_upper, alpha)

        # Skill score
        skill = None
        if y_train is not None and len(y_train) > self.seasonal_period:
            skill = self._skill_score(y_true, y_pred, y_train)

        return EvaluationResult(
            mape=mape,
            rmse=rmse,
            mae=mae,
            bias=bias,
            mase=mase,
            coverage=coverage,
            interval_width=interval_width,
            interval_score=interval_score,
            skill_vs_naive=skill,
        )

    def _mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error.

        MAPE = (100/n) * sum(|y - y_hat| / |y|)

        Note: Undefined when y = 0. Uses small epsilon to avoid division by zero.
        """
        epsilon = 1e-8
        return float(100 * np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), epsilon)))

    def _rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error.

        RMSE = sqrt((1/n) * sum((y - y_hat)^2))

        Penalizes large errors more than MAE.
        """
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def _mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error.

        MAE = (1/n) * sum(|y - y_hat|)

        More robust to outliers than RMSE.
        """
        return float(np.mean(np.abs(y_true - y_pred)))

    def _bias(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Forecast Bias (Mean Error).

        Bias = (1/n) * sum(y_hat - y)

        Positive: Over-prediction
        Negative: Under-prediction
        """
        return float(np.mean(y_pred - y_true))

    def _mase(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray,
    ) -> float:
        """
        Mean Absolute Scaled Error.

        MASE = MAE / MAE_naive

        where MAE_naive is the in-sample MAE of seasonal naive forecast.

        Interpretation:
        - MASE < 1: Better than seasonal naive
        - MASE = 1: Same as seasonal naive
        - MASE > 1: Worse than seasonal naive
        """
        m = self.seasonal_period

        # In-sample naive errors: y_t - y_{t-m}
        naive_errors = y_train[m:] - y_train[:-m]
        mae_naive = np.mean(np.abs(naive_errors))

        if mae_naive < 1e-8:
            return np.nan

        mae_forecast = self._mae(y_true, y_pred)
        return float(mae_forecast / mae_naive)

    def _coverage(
        self,
        y_true: np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
    ) -> float:
        """
        Empirical Coverage Rate.

        Coverage = (1/n) * sum(I[y in [lower, upper]])

        For well-calibrated (1-alpha) intervals, coverage should equal 1-alpha.
        """
        in_interval = (y_true >= y_lower) & (y_true <= y_upper)
        return float(np.mean(in_interval))

    def _interval_width(
        self,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
    ) -> float:
        """
        Mean Interval Width (Sharpness).

        Width = (1/n) * sum(upper - lower)

        Given same coverage, narrower intervals are preferred.
        """
        return float(np.mean(y_upper - y_lower))

    def _interval_score(
        self,
        y_true: np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
        alpha: float,
    ) -> float:
        """
        Interval Score (Winkler Score).

        IS = (u - l) + (2/alpha)(l - y)_+ + (2/alpha)(y - u)_+

        Proper scoring rule that penalizes:
        1. Wide intervals (first term)
        2. Under-coverage on lower end (second term)
        3. Under-coverage on upper end (third term)

        Lower is better.
        """
        width = y_upper - y_lower
        lower_violation = np.maximum(y_lower - y_true, 0)
        upper_violation = np.maximum(y_true - y_upper, 0)

        score = width + (2 / alpha) * lower_violation + (2 / alpha) * upper_violation
        return float(np.mean(score))

    def _skill_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray,
    ) -> float:
        """
        Skill Score vs Seasonal Naive.

        Skill = 1 - (MAE_model / MAE_naive)

        Interpretation:
        - Skill > 0: Better than naive
        - Skill = 0: Same as naive
        - Skill < 0: Worse than naive
        - Skill = 1: Perfect forecast
        """
        m = self.seasonal_period

        # Generate naive forecast
        # Use last available seasonal value
        n_forecast = len(y_true)
        naive_pred = y_train[-m:].tolist() * (n_forecast // m + 1)
        naive_pred = np.array(naive_pred[:n_forecast])

        mae_model = self._mae(y_true, y_pred)
        mae_naive = self._mae(y_true, naive_pred)

        if mae_naive < 1e-8:
            return np.nan

        return float(1 - mae_model / mae_naive)


def crps_gaussian(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """
    Continuous Ranked Probability Score for Gaussian forecasts.

    CRPS(F, y) = E_F[|X - y|] - 0.5 * E_F[|X - X'|]

    For Gaussian: CRPS = sigma * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))

    where z = (y - mu) / sigma, Phi is CDF, phi is PDF.

    Lower is better. CRPS = 0 for perfect deterministic forecast.
    """
    z = (y_true - mu) / sigma

    crps = sigma * (
        z * (2 * stats.norm.cdf(z) - 1)
        + 2 * stats.norm.pdf(z)
        - 1 / np.sqrt(np.pi)
    )

    return float(np.mean(crps))


def crps_empirical(
    y_true: np.ndarray,
    samples: np.ndarray,
) -> float:
    """
    Empirical CRPS from forecast samples.

    CRPS = (1/n) * sum(|x_i - y|) - (1/2n^2) * sum(|x_i - x_j|)

    Args:
        y_true: Actual values (n_points,)
        samples: Forecast samples (n_points, n_samples)

    Returns:
        Mean CRPS
    """
    n_points, n_samples = samples.shape
    crps_values = []

    for i in range(n_points):
        y = y_true[i]
        x = samples[i]

        # First term: mean distance to observation
        term1 = np.mean(np.abs(x - y))

        # Second term: mean pairwise distance
        term2 = 0.0
        for j in range(n_samples):
            for k in range(j + 1, n_samples):
                term2 += np.abs(x[j] - x[k])
        term2 = 2 * term2 / (n_samples * (n_samples - 1))

        crps_values.append(term1 - 0.5 * term2)

    return float(np.mean(crps_values))


class TimeSeriesCrossValidator:
    """
    Time Series Cross-Validation with expanding or sliding window.

    Protocol:
    [====Train====][Gap][==Test==]
    [======Train======][Gap][==Test==]
    [========Train========][Gap][==Test==]

    Key differences from standard CV:
    1. Temporal order is preserved (no random shuffling)
    2. Test always follows train (no leakage)
    3. Optional gap to prevent lag feature leakage
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 7,
        gap: int = 1,
        min_train_size: int = 365,
        expanding: bool = True,
    ):
        """
        Args:
            n_splits: Number of CV folds
            test_size: Size of test window
            gap: Gap between train and test (prevents leakage)
            min_train_size: Minimum training size
            expanding: If True, training window expands; if False, slides
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.min_train_size = min_train_size
        self.expanding = expanding

    def split(self, X: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each fold.

        Args:
            X: DataFrame to split

        Returns:
            List of (train_indices, test_indices) tuples
        """
        n = len(X)

        # Calculate fold positions
        total_required = self.min_train_size + self.gap + self.test_size
        remaining = n - total_required
        step = remaining // (self.n_splits - 1) if self.n_splits > 1 else 0

        splits = []
        for i in range(self.n_splits):
            test_end = self.min_train_size + self.gap + self.test_size + i * step
            test_start = test_end - self.test_size

            if self.expanding:
                train_start = 0
            else:
                train_start = test_start - self.gap - self.min_train_size

            train_end = test_start - self.gap

            if test_end <= n and train_end > train_start:
                train_idx = np.arange(train_start, train_end)
                test_idx = np.arange(test_start, test_end)
                splits.append((train_idx, test_idx))

        return splits

    def cross_validate(
        self,
        model_factory: Callable,
        X: pd.DataFrame,
        y: pd.Series,
        evaluator: ForecastEvaluator,
    ) -> list[EvaluationResult]:
        """
        Run cross-validation and collect results.

        Args:
            model_factory: Callable that returns a fresh model instance
            X: Feature DataFrame
            y: Target Series
            evaluator: ForecastEvaluator instance

        Returns:
            List of EvaluationResult for each fold
        """
        results = []

        for train_idx, test_idx in self.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train model
            model = model_factory()
            model.fit(X_train, y_train)

            # Predict
            predictions = model.predict(X_test)

            # Handle different prediction formats
            if isinstance(predictions, dict):
                y_pred = predictions.get("point", predictions.get("mean"))
                y_lower = predictions.get("lower")
                y_upper = predictions.get("upper")
            else:
                y_pred = predictions
                y_lower = None
                y_upper = None

            # Evaluate
            result = evaluator.evaluate(
                y_true=y_test.values,
                y_pred=y_pred,
                y_lower=y_lower,
                y_upper=y_upper,
                y_train=y_train.values,
            )
            results.append(result)

        return results


class DieboldMariano:
    """
    Diebold-Mariano test for comparing forecast accuracy.

    Tests H0: E[d_t] = 0 where d_t = L(e1_t) - L(e2_t)

    The DM statistic:
    DM = d_bar / sqrt(V(d_bar) / T)

    Under H0, DM ~ N(0, 1) asymptotically.

    Reference:
    Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy.
    Journal of Business & Economic Statistics.
    """

    def __init__(self, loss_fn: Optional[Callable] = None):
        """
        Args:
            loss_fn: Loss function (default: squared error)
        """
        self.loss_fn = loss_fn or (lambda y, yhat: (y - yhat) ** 2)

    def test(
        self,
        y_true: np.ndarray,
        y_pred1: np.ndarray,
        y_pred2: np.ndarray,
        h: int = 1,
    ) -> dict:
        """
        Perform Diebold-Mariano test.

        Args:
            y_true: Actual values
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2
            h: Forecast horizon (for HAC variance)

        Returns:
            Dictionary with DM statistic, p-value, and interpretation
        """
        # Compute loss differentials
        loss1 = self.loss_fn(y_true, y_pred1)
        loss2 = self.loss_fn(y_true, y_pred2)
        d = loss1 - loss2

        # Mean difference
        d_bar = np.mean(d)

        # HAC variance estimator (Newey-West)
        T = len(d)
        gamma_0 = np.var(d)
        gamma_sum = 0.0

        for k in range(1, h):
            weight = 1 - k / h  # Bartlett kernel
            gamma_k = np.cov(d[k:], d[:-k])[0, 1]
            gamma_sum += weight * gamma_k

        var_d_bar = (gamma_0 + 2 * gamma_sum) / T

        # DM statistic
        if var_d_bar <= 0:
            return {
                "dm_statistic": np.nan,
                "p_value": np.nan,
                "significant": False,
                "interpretation": "Variance estimation failed",
            }

        dm_stat = d_bar / np.sqrt(var_d_bar)

        # Two-sided p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        # Interpretation
        if p_value < 0.05:
            if dm_stat > 0:
                interp = "Model 2 significantly better than Model 1"
            else:
                interp = "Model 1 significantly better than Model 2"
        else:
            interp = "No significant difference between models"

        return {
            "dm_statistic": float(dm_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "mean_loss_diff": float(d_bar),
            "interpretation": interp,
        }


def pit_histogram(
    y_true: np.ndarray,
    cdf_values: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Probability Integral Transform (PIT) histogram.

    For calibrated probabilistic forecasts, the PIT values
    u_i = F(y_i) should be Uniform[0, 1].

    The PIT histogram reveals:
    - Uniform: Well-calibrated
    - U-shaped: Under-dispersed (too confident)
    - Inverted U: Over-dispersed (too uncertain)
    - Skewed: Biased

    Args:
        y_true: Actual values
        cdf_values: CDF values F(y) for each observation

    Returns:
        (bin_edges, frequencies) for plotting
    """
    frequencies, bin_edges = np.histogram(cdf_values, bins=n_bins, range=(0, 1))
    frequencies = frequencies / len(cdf_values)
    return bin_edges, frequencies


def reliability_diagram(
    y_true: np.ndarray,
    quantile_preds: dict[float, np.ndarray],
) -> pd.DataFrame:
    """
    Reliability diagram data for quantile calibration.

    For each nominal quantile level, computes empirical coverage.
    Perfect calibration: empirical = nominal for all quantiles.

    Args:
        y_true: Actual values
        quantile_preds: Dict mapping quantile level to predictions

    Returns:
        DataFrame with nominal and empirical quantiles
    """
    results = []
    for nominal_q, preds in sorted(quantile_preds.items()):
        empirical_q = np.mean(y_true <= preds)
        results.append({
            "nominal": nominal_q,
            "empirical": empirical_q,
            "calibration_error": abs(empirical_q - nominal_q),
        })
    return pd.DataFrame(results)
