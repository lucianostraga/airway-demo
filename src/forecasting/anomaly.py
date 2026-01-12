"""
Anomaly Detection and Regime Change Detection

Implements:
1. BOCPD - Bayesian Online Change Point Detection
2. CUSUM - Cumulative Sum for drift detection
3. Isolation Forest - Multivariate anomaly detection

These methods detect:
- Sudden level shifts (change points)
- Gradual drift from baseline
- Outliers in feature space

Reference:
Adams & MacKay (2007). Bayesian Online Changepoint Detection.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import stats


@dataclass
class BOCPDConfig:
    """Configuration for Bayesian Online Change Point Detection."""

    hazard_rate: float = 1 / 250  # Prior probability of change point
    prior_mean: float = 0.0
    prior_var: float = 1.0
    observation_var: float = 1.0


class BayesianOnlineChangePointDetector:
    """
    Bayesian Online Changepoint Detection (BOCPD).

    Maintains a distribution over run lengths r_t (time since last change point).
    When P(r_t = 0) is high, a change point is likely.

    The algorithm:
    1. For each new observation, update the run length distribution
    2. P(r_t = 0) indicates probability of a change point at time t
    3. Weighted prediction uses run length posterior

    Key equations:
    - P(r_t | x_{1:t}) = P(x_t | r_{t-1}, x_{1:t-1}) * P(r_t | r_{t-1}) * P(r_{t-1} | x_{1:t-1})
    - P(r_t = 0) = sum over r_{t-1} of P(r_t=0 | r_{t-1}) * P(r_{t-1} | x_{1:t-1})

    Reference:
    Adams, R. P., & MacKay, D. J. C. (2007). Bayesian Online Changepoint Detection.
    """

    def __init__(self, config: Optional[BOCPDConfig] = None):
        self.config = config or BOCPDConfig()
        self._run_length_probs: list[np.ndarray] = []
        self._changepoint_probs: list[float] = []
        self._t = 0

        # Sufficient statistics for Gaussian conjugate prior (Normal-Gamma)
        self._sum_x: list[float] = [0.0]  # Sum of observations in run
        self._sum_x2: list[float] = [0.0]  # Sum of squared observations
        self._n: list[int] = [0]  # Number of observations in run

    def hazard(self, r: int) -> float:
        """
        Hazard function: P(change point | run length = r).

        Constant hazard corresponds to geometric prior on run length.
        More sophisticated: increasing hazard for regime-specific duration priors.
        """
        return self.config.hazard_rate

    def predictive(self, x: float, r: int) -> float:
        """
        Predictive probability: P(x_t | r_{t-1}, x_{1:t-1}).

        Uses conjugate Normal-Gamma prior for Gaussian observations.
        For run length r, uses sufficient statistics from that run.
        """
        if r >= len(self._n) or self._n[r] == 0:
            # No data in this run, use prior
            return stats.norm.pdf(
                x, loc=self.config.prior_mean, scale=np.sqrt(self.config.prior_var)
            )

        # Posterior predictive (Student-t for Normal-Gamma conjugate)
        n = self._n[r]
        sum_x = self._sum_x[r]
        sum_x2 = self._sum_x2[r]

        # Posterior parameters
        mu_n = (self.config.prior_mean + sum_x) / (1 + n)

        # Simplified: use known variance assumption
        var = self.config.observation_var
        return stats.norm.pdf(x, loc=mu_n, scale=np.sqrt(var))

    def update(self, x: float) -> float:
        """
        Process a new observation and return change point probability.

        Args:
            x: New observation

        Returns:
            P(r_t = 0 | x_{1:t}): Probability of change point at time t
        """
        self._t += 1

        if self._t == 1:
            # First observation
            self._run_length_probs.append(np.array([1.0]))
            self._update_sufficient_stats(x, 0)
            self._changepoint_probs.append(0.0)
            return 0.0

        prev_probs = self._run_length_probs[-1]
        n_runs = len(prev_probs)

        # Growth probabilities: P(r_t = r_{t-1} + 1)
        growth_probs = np.zeros(n_runs + 1)
        for r in range(n_runs):
            pred_prob = self.predictive(x, r)
            growth_probs[r + 1] = pred_prob * (1 - self.hazard(r)) * prev_probs[r]

        # Changepoint probability: P(r_t = 0)
        cp_prob = 0.0
        for r in range(n_runs):
            pred_prob = self.predictive(x, r)
            cp_prob += pred_prob * self.hazard(r) * prev_probs[r]
        growth_probs[0] = cp_prob

        # Normalize
        total = growth_probs.sum()
        if total > 0:
            growth_probs /= total

        # Store
        self._run_length_probs.append(growth_probs)

        # Update sufficient statistics for each run length
        self._extend_sufficient_stats()
        for r in range(len(growth_probs)):
            if growth_probs[r] > 1e-6:
                self._update_sufficient_stats(x, r)

        self._changepoint_probs.append(growth_probs[0])
        return growth_probs[0]

    def _extend_sufficient_stats(self):
        """Extend sufficient statistics for new run length."""
        # Add new entry for run length 0 (new regime starts fresh)
        self._sum_x = [0.0] + self._sum_x
        self._sum_x2 = [0.0] + self._sum_x2
        self._n = [0] + self._n

    def _update_sufficient_stats(self, x: float, r: int):
        """Update sufficient statistics for run length r."""
        if r < len(self._sum_x):
            self._sum_x[r] += x
            self._sum_x2[r] += x * x
            self._n[r] += 1

    def get_changepoint_probability(self) -> float:
        """Get the most recent change point probability."""
        if not self._changepoint_probs:
            return 0.0
        return self._changepoint_probs[-1]

    def detect(self, threshold: float = 0.5) -> bool:
        """
        Detect if a change point occurred.

        Args:
            threshold: Alert threshold for P(change point)

        Returns:
            True if change point detected
        """
        return self.get_changepoint_probability() > threshold

    def get_expected_run_length(self) -> float:
        """Get expected run length (time since last change point)."""
        if not self._run_length_probs:
            return 0.0
        probs = self._run_length_probs[-1]
        run_lengths = np.arange(len(probs))
        return np.sum(run_lengths * probs)


@dataclass
class CUSUMConfig:
    """Configuration for CUSUM detector."""

    target_mean: float = 0.0  # Expected value under null hypothesis
    slack: float = 0.5  # Allowable deviation (in std units)
    threshold: float = 5.0  # Decision threshold
    reset_on_alarm: bool = True


class CUSUM:
    """
    Cumulative Sum (CUSUM) for drift/shift detection.

    Tracks cumulative deviations from target mean:
    S_t^+ = max(0, S_{t-1}^+ + (x_t - mu_0 - k))  # Upward shift
    S_t^- = max(0, S_{t-1}^- - (x_t - mu_0 + k))  # Downward shift

    Alarm when S^+ > h or S^- > h

    Properties:
    - Optimal for detecting shifts of known size
    - Trade-off between detection speed and false alarm rate
    - Parameters: k (allowable slack), h (decision interval)

    Relationship to hypothesis testing:
    - k = delta/2 where delta is the shift size to detect
    - h controls false alarm rate (ARL_0)
    """

    def __init__(self, config: Optional[CUSUMConfig] = None):
        self.config = config or CUSUMConfig()
        self._S_pos = 0.0  # Cumulative sum for positive shift
        self._S_neg = 0.0  # Cumulative sum for negative shift
        self._history_pos: list[float] = []
        self._history_neg: list[float] = []
        self._alarms: list[tuple[int, str]] = []  # (time, direction)
        self._t = 0

    def update(self, x: float) -> tuple[bool, str]:
        """
        Process a new observation.

        Args:
            x: New observation

        Returns:
            (alarm, direction): Whether alarm triggered and which direction
        """
        self._t += 1
        k = self.config.slack
        mu = self.config.target_mean
        h = self.config.threshold

        # Update cumulative sums
        self._S_pos = max(0, self._S_pos + (x - mu - k))
        self._S_neg = max(0, self._S_neg - (x - mu + k))

        self._history_pos.append(self._S_pos)
        self._history_neg.append(self._S_neg)

        # Check for alarm
        alarm = False
        direction = "none"

        if self._S_pos > h:
            alarm = True
            direction = "up"
            self._alarms.append((self._t, "up"))
            if self.config.reset_on_alarm:
                self._S_pos = 0.0

        if self._S_neg > h:
            alarm = True
            direction = "down" if direction == "none" else "both"
            self._alarms.append((self._t, "down"))
            if self.config.reset_on_alarm:
                self._S_neg = 0.0

        return alarm, direction

    def reset(self):
        """Reset the CUSUM statistics."""
        self._S_pos = 0.0
        self._S_neg = 0.0

    @property
    def current_statistic(self) -> tuple[float, float]:
        """Get current (S+, S-) values."""
        return self._S_pos, self._S_neg

    def get_alarms(self) -> list[tuple[int, str]]:
        """Get list of all alarms (time, direction)."""
        return self._alarms.copy()


class IsolationForest:
    """
    Isolation Forest for multivariate anomaly detection.

    Key insight: Anomalies are "few and different" - they are isolated
    more quickly in random trees than normal points.

    Anomaly score:
    s(x, n) = 2^{-E[h(x)] / c(n)}

    where:
    - h(x): Path length to isolate point x
    - c(n): Average path length for unsuccessful search in BST
    - E[h(x)]: Expected path length across forest

    Interpretation:
    - s approx 1: Definite anomaly (short path)
    - s approx 0.5: Normal point
    - s < 0.5: Very normal (dense region)

    Uses scikit-learn implementation with additional utilities.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            n_estimators: Number of isolation trees
            contamination: Expected proportion of anomalies
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self._model = None
        self._threshold: Optional[float] = None

    def fit(self, X: np.ndarray) -> "IsolationForest":
        """
        Fit the isolation forest.

        Args:
            X: Training data (n_samples, n_features)
        """
        try:
            from sklearn.ensemble import IsolationForest as SklearnIF
        except ImportError:
            raise ImportError("scikit-learn is required: pip install scikit-learn")

        self._model = SklearnIF(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self._model.fit(X)

        # Determine threshold from training data
        scores = self._model.score_samples(X)
        self._threshold = np.percentile(scores, 100 * self.contamination)

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.

        Args:
            X: Data to score (n_samples, n_features)

        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # sklearn returns negative scores (more negative = more anomalous)
        # We negate to get positive scores where higher = more anomalous
        return -self._model.score_samples(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.

        Args:
            X: Data to predict (n_samples, n_features)

        Returns:
            Binary predictions (1 = anomaly, 0 = normal)
        """
        if self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # sklearn returns -1 for anomalies, 1 for normal
        preds = self._model.predict(X)
        return (preds == -1).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Get decision function values.

        Positive = normal, negative = anomaly.
        """
        if self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._model.decision_function(X)


class AnomalyDetector:
    """
    Unified anomaly detection combining multiple methods.

    Combines:
    1. CUSUM for level shifts
    2. Isolation Forest for multivariate outliers
    3. Statistical thresholds (z-score) for simple outliers

    Returns ensemble score and explanations.
    """

    def __init__(
        self,
        use_cusum: bool = True,
        use_isolation_forest: bool = True,
        use_zscore: bool = True,
        zscore_threshold: float = 3.0,
    ):
        self.use_cusum = use_cusum
        self.use_isolation_forest = use_isolation_forest
        self.use_zscore = use_zscore
        self.zscore_threshold = zscore_threshold

        self._cusum: Optional[CUSUM] = None
        self._iforest: Optional[IsolationForest] = None
        self._mean: Optional[float] = None
        self._std: Optional[float] = None

    def fit(self, X: np.ndarray, target_col: int = 0) -> "AnomalyDetector":
        """
        Fit the anomaly detector.

        Args:
            X: Training data (n_samples, n_features)
            target_col: Column index to use for CUSUM
        """
        if self.use_cusum:
            target = X[:, target_col]
            self._mean = float(np.mean(target))
            self._std = float(np.std(target))
            self._cusum = CUSUM(
                CUSUMConfig(
                    target_mean=self._mean,
                    slack=0.5 * self._std,
                    threshold=5.0 * self._std,
                )
            )

        if self.use_isolation_forest:
            self._iforest = IsolationForest()
            self._iforest.fit(X)

        if self.use_zscore:
            if self._mean is None:
                target = X[:, target_col]
                self._mean = float(np.mean(target))
                self._std = float(np.std(target))

        return self

    def detect(
        self,
        x: np.ndarray,
        target_idx: int = 0,
    ) -> dict:
        """
        Detect anomalies in a single observation.

        Args:
            x: Single observation (n_features,)
            target_idx: Index of target variable for CUSUM

        Returns:
            Dictionary with detection results and explanations
        """
        results = {
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "explanations": [],
        }

        scores = []

        if self.use_cusum and self._cusum is not None:
            alarm, direction = self._cusum.update(x[target_idx])
            if alarm:
                results["is_anomaly"] = True
                results["explanations"].append(f"CUSUM alarm: {direction} shift detected")
            S_pos, S_neg = self._cusum.current_statistic
            cusum_score = max(S_pos, S_neg) / (self._cusum.config.threshold or 1.0)
            scores.append(min(cusum_score, 1.0))

        if self.use_isolation_forest and self._iforest is not None:
            if_score = self._iforest.score(x.reshape(1, -1))[0]
            if self._iforest.predict(x.reshape(1, -1))[0] == 1:
                results["is_anomaly"] = True
                results["explanations"].append("Isolation Forest: outlier in feature space")
            scores.append(min(if_score, 1.0))

        if self.use_zscore and self._mean is not None and self._std is not None:
            zscore = abs(x[target_idx] - self._mean) / max(self._std, 1e-6)
            if zscore > self.zscore_threshold:
                results["is_anomaly"] = True
                results["explanations"].append(f"Z-score: {zscore:.2f} > {self.zscore_threshold}")
            scores.append(min(zscore / self.zscore_threshold, 1.0))

        if scores:
            results["anomaly_score"] = float(np.mean(scores))

        return results
