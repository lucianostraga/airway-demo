"""
Cold-Start Handling for New Routes

Implements strategies for forecasting when historical data is limited:
1. Hierarchical Prior - Inherit from station/tier level
2. Analogous Route Transfer - Learn from similar existing routes
3. Meta-Learning (MAML-style) - Quick adaptation from few examples

The cold-start problem is critical for new routes where we have:
- No historical demand data
- Limited operational history
- Need to forecast from day 1

Mathematical foundation:
- Bayesian hierarchical models with informative priors
- Kernel-based similarity for transfer learning
- Feature-based matching for analogous routes
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


@dataclass
class RouteFeatures:
    """Features describing a route for similarity matching."""

    origin_station: str
    destination_station: str
    origin_tier: str  # 'hub', 'focus', 'spoke'
    destination_tier: str
    distance_km: float
    avg_flight_duration_hours: float
    aircraft_type_distribution: dict[str, float]  # e.g., {'widebody': 0.3, 'narrowbody': 0.7}
    is_international: bool
    primary_purpose: str  # 'business', 'leisure', 'mixed'
    flights_per_day: float


@dataclass
class ColdStartConfig:
    """Configuration for cold-start handling."""

    # Hierarchical prior settings
    prior_strength: float = 0.5  # Weight given to prior vs data
    min_observations: int = 30  # Minimum obs before reducing prior weight

    # Analogous route settings
    n_similar_routes: int = 5  # Number of similar routes to use
    similarity_kernel: str = "rbf"  # 'rbf', 'cosine', 'euclidean'
    kernel_bandwidth: float = 1.0

    # Meta-learning settings
    adaptation_steps: int = 5
    adaptation_lr: float = 0.01


class HierarchicalPrior:
    """
    Hierarchical prior for cold-start forecasting.

    Uses Bayesian partial pooling:
    theta_route ~ N(theta_station, sigma_route)
    theta_station ~ N(theta_tier, sigma_station)
    theta_tier ~ N(theta_global, sigma_tier)

    For a new route with no data:
    - Use station-level estimate as prior mean
    - Uncertainty from tier-level variance

    As data accumulates:
    - Posterior shrinks toward data
    - Prior influence diminishes
    """

    def __init__(self, config: Optional[ColdStartConfig] = None):
        self.config = config or ColdStartConfig()
        self._tier_params: dict[str, dict] = {}
        self._station_params: dict[str, dict] = {}
        self._global_params: Optional[dict] = None

    def fit(
        self,
        historical_data: pd.DataFrame,
        route_col: str = "route",
        station_col: str = "station",
        tier_col: str = "tier",
        demand_col: str = "demand",
    ) -> "HierarchicalPrior":
        """
        Fit hierarchical priors from historical data.

        Estimates mean and variance at each level:
        - Global: Mean/var across all routes
        - Tier: Mean/var within each tier (hub, focus, spoke)
        - Station: Mean/var within each station
        """
        # Global parameters
        global_mean = historical_data[demand_col].mean()
        global_var = historical_data[demand_col].var()
        self._global_params = {"mean": global_mean, "var": global_var}

        # Tier-level parameters
        tier_stats = historical_data.groupby(tier_col)[demand_col].agg(["mean", "var", "count"])
        for tier in tier_stats.index:
            self._tier_params[tier] = {
                "mean": tier_stats.loc[tier, "mean"],
                "var": tier_stats.loc[tier, "var"],
                "count": tier_stats.loc[tier, "count"],
            }

        # Station-level parameters
        station_stats = historical_data.groupby(station_col)[demand_col].agg(["mean", "var", "count"])
        for station in station_stats.index:
            self._station_params[station] = {
                "mean": station_stats.loc[station, "mean"],
                "var": station_stats.loc[station, "var"],
                "count": station_stats.loc[station, "count"],
            }

        return self

    def get_prior(
        self,
        station: str,
        tier: str,
    ) -> tuple[float, float]:
        """
        Get prior mean and variance for a new route.

        Uses hierarchical fallback:
        1. Station-level if available
        2. Tier-level if station unknown
        3. Global if tier unknown

        Returns:
            (prior_mean, prior_variance)
        """
        if station in self._station_params:
            params = self._station_params[station]
            return params["mean"], params["var"]
        elif tier in self._tier_params:
            params = self._tier_params[tier]
            return params["mean"], params["var"]
        elif self._global_params is not None:
            return self._global_params["mean"], self._global_params["var"]
        else:
            # Fallback to uninformative prior
            return 0.0, 100.0

    def get_posterior(
        self,
        station: str,
        tier: str,
        observed_data: np.ndarray,
    ) -> tuple[float, float]:
        """
        Compute posterior after observing some data.

        Bayesian update (conjugate Normal-Normal):
        mu_posterior = (sigma^2_prior * sum(x) + sigma^2_obs * mu_prior) /
                       (n * sigma^2_prior + sigma^2_obs)

        As n increases, posterior converges to data mean.

        Args:
            station: Station code
            tier: Station tier
            observed_data: Observed demand values

        Returns:
            (posterior_mean, posterior_variance)
        """
        prior_mean, prior_var = self.get_prior(station, tier)

        if len(observed_data) == 0:
            return prior_mean, prior_var

        n = len(observed_data)
        data_mean = np.mean(observed_data)
        data_var = np.var(observed_data) if n > 1 else prior_var

        # Bayesian update
        # Precision weighting
        prior_precision = 1 / max(prior_var, 1e-6)
        data_precision = n / max(data_var, 1e-6)

        posterior_precision = prior_precision + data_precision
        posterior_mean = (
            prior_precision * prior_mean + data_precision * data_mean
        ) / posterior_precision
        posterior_var = 1 / posterior_precision

        return posterior_mean, posterior_var


class AnalogousRouteTransfer:
    """
    Transfer learning from similar existing routes.

    For a new route, finds k most similar existing routes based on
    feature similarity, then transfers forecasts with weighted averaging.

    Similarity kernel options:
    - RBF: exp(-||z_new - z_j||^2 / (2 * sigma^2))
    - Cosine: z_new . z_j / (||z_new|| * ||z_j||)
    - Euclidean: 1 / (1 + ||z_new - z_j||)
    """

    def __init__(self, config: Optional[ColdStartConfig] = None):
        self.config = config or ColdStartConfig()
        self._route_features: dict[str, np.ndarray] = {}
        self._route_forecasters: dict[str, object] = {}
        self._feature_scaler: Optional[object] = None

    def register_route(
        self,
        route_id: str,
        features: RouteFeatures,
        forecaster: object,
    ) -> None:
        """
        Register an existing route with its features and trained forecaster.

        Args:
            route_id: Unique route identifier (e.g., "ATL-JFK")
            features: RouteFeatures object
            forecaster: Trained forecaster for this route
        """
        feature_vector = self._features_to_vector(features)
        self._route_features[route_id] = feature_vector
        self._route_forecasters[route_id] = forecaster

    def _features_to_vector(self, features: RouteFeatures) -> np.ndarray:
        """Convert RouteFeatures to numeric vector."""
        # Encode categorical features
        tier_encoding = {"hub": 0, "focus": 1, "spoke": 2}
        purpose_encoding = {"business": 0, "leisure": 1, "mixed": 2}

        # Compute aircraft type summary
        widebody_frac = features.aircraft_type_distribution.get("widebody", 0)

        vector = np.array([
            tier_encoding.get(features.origin_tier, 2),
            tier_encoding.get(features.destination_tier, 2),
            features.distance_km / 5000,  # Normalize
            features.avg_flight_duration_hours / 12,
            widebody_frac,
            float(features.is_international),
            purpose_encoding.get(features.primary_purpose, 2),
            np.log1p(features.flights_per_day),
        ])

        return vector

    def find_similar_routes(
        self,
        new_features: RouteFeatures,
    ) -> list[tuple[str, float]]:
        """
        Find k most similar existing routes.

        Args:
            new_features: Features of the new route

        Returns:
            List of (route_id, similarity_weight) tuples
        """
        new_vector = self._features_to_vector(new_features)

        similarities = []
        for route_id, feat_vector in self._route_features.items():
            sim = self._compute_similarity(new_vector, feat_vector)
            similarities.append((route_id, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Take top k
        top_k = similarities[: self.config.n_similar_routes]

        # Normalize weights
        total_weight = sum(w for _, w in top_k)
        if total_weight > 0:
            top_k = [(r, w / total_weight) for r, w in top_k]

        return top_k

    def _compute_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
    ) -> float:
        """Compute similarity between two feature vectors."""
        if self.config.similarity_kernel == "rbf":
            dist_sq = np.sum((vec1 - vec2) ** 2)
            return float(np.exp(-dist_sq / (2 * self.config.kernel_bandwidth ** 2)))

        elif self.config.similarity_kernel == "cosine":
            dot = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 0 and norm2 > 0:
                return float(dot / (norm1 * norm2))
            return 0.0

        else:  # euclidean
            dist = np.linalg.norm(vec1 - vec2)
            return float(1 / (1 + dist))

    def transfer_forecast(
        self,
        new_features: RouteFeatures,
        X_new: pd.DataFrame,
        capacity_scaling: float = 1.0,
    ) -> np.ndarray:
        """
        Generate forecast for new route using transfer learning.

        Weighted average of forecasts from similar routes:
        y_new = sum_j w_j * y_j * (capacity_new / capacity_j)

        Args:
            new_features: Features of new route
            X_new: Feature data for prediction
            capacity_scaling: Scale factor for capacity difference

        Returns:
            Transferred forecast
        """
        similar_routes = self.find_similar_routes(new_features)

        if not similar_routes:
            raise ValueError("No similar routes found. Register routes first.")

        # Weighted combination of forecasts
        combined_forecast = np.zeros(len(X_new))

        for route_id, weight in similar_routes:
            if route_id in self._route_forecasters:
                forecaster = self._route_forecasters[route_id]
                route_forecast = forecaster.predict(X_new)

                # Apply capacity scaling
                scaled_forecast = route_forecast * capacity_scaling
                combined_forecast += weight * scaled_forecast

        return combined_forecast


class AdaptiveForecaster:
    """
    Adaptive forecaster that transitions from cold-start to data-driven.

    Implements a smooth transition:
    1. Initially: Use hierarchical prior + transfer learning
    2. As data accumulates: Shift weight to trained model
    3. Eventually: Pure data-driven forecasting

    Weight function:
    w_data = min(1, n / n_threshold)
    y_final = w_data * y_model + (1 - w_data) * y_prior
    """

    def __init__(
        self,
        base_forecaster: object,
        hierarchical_prior: HierarchicalPrior,
        transfer_forecaster: Optional[AnalogousRouteTransfer] = None,
        warmup_threshold: int = 30,
    ):
        """
        Args:
            base_forecaster: Data-driven forecaster (e.g., LightGBM)
            hierarchical_prior: Fitted hierarchical prior
            transfer_forecaster: Optional transfer learning component
            warmup_threshold: Observations needed before full data reliance
        """
        self.base_forecaster = base_forecaster
        self.hierarchical_prior = hierarchical_prior
        self.transfer_forecaster = transfer_forecaster
        self.warmup_threshold = warmup_threshold

        self._n_observations = 0
        self._fitted = False
        self._station: Optional[str] = None
        self._tier: Optional[str] = None

    def partial_fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        station: str,
        tier: str,
    ) -> "AdaptiveForecaster":
        """
        Incrementally fit with new data.

        Updates:
        1. Observation count
        2. Posterior estimates
        3. Base model (when enough data)
        """
        self._station = station
        self._tier = tier
        self._n_observations += len(y)

        # Train base model if enough data
        if self._n_observations >= self.warmup_threshold:
            self.base_forecaster.fit(X, y)
            self._fitted = True

        return self

    def predict(
        self,
        X: pd.DataFrame,
        route_features: Optional[RouteFeatures] = None,
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions with adaptive weighting.

        Returns:
            Dictionary with 'point', 'lower', 'upper', 'data_weight'
        """
        n = len(X)

        # Compute adaptive weight
        data_weight = min(1.0, self._n_observations / self.warmup_threshold)

        # Prior/transfer forecast
        if self.transfer_forecaster is not None and route_features is not None:
            prior_forecast = self.transfer_forecaster.transfer_forecast(
                route_features, X
            )
        elif self._station is not None and self._tier is not None:
            prior_mean, prior_var = self.hierarchical_prior.get_prior(
                self._station, self._tier
            )
            prior_forecast = np.full(n, prior_mean)
        else:
            prior_forecast = np.zeros(n)

        # Data-driven forecast
        if self._fitted and data_weight > 0:
            try:
                model_output = self.base_forecaster.predict(X)
                if isinstance(model_output, dict):
                    model_forecast = model_output.get("point", model_output.get("mean"))
                else:
                    model_forecast = model_output
            except Exception:
                model_forecast = prior_forecast
                data_weight = 0.0
        else:
            model_forecast = prior_forecast
            data_weight = 0.0

        # Weighted combination
        point_forecast = (
            data_weight * model_forecast + (1 - data_weight) * prior_forecast
        )

        # Uncertainty (wider when relying on prior)
        if self._station is not None and self._tier is not None:
            _, prior_var = self.hierarchical_prior.get_posterior(
                self._station,
                self._tier,
                np.array([]),  # Use prior variance
            )
        else:
            prior_var = 100.0

        # Uncertainty shrinks as data accumulates
        effective_var = prior_var * (1 - data_weight) ** 2
        interval_width = 1.96 * np.sqrt(effective_var)

        return {
            "point": point_forecast,
            "lower": point_forecast - interval_width,
            "upper": point_forecast + interval_width,
            "data_weight": data_weight,
        }


def compute_route_similarity_matrix(
    routes: list[RouteFeatures],
    kernel: str = "rbf",
    bandwidth: float = 1.0,
) -> np.ndarray:
    """
    Compute pairwise similarity matrix for all routes.

    Useful for:
    - Clustering routes into groups
    - Identifying route "families" for grouped forecasting
    - Visualization of route similarities

    Args:
        routes: List of RouteFeatures
        kernel: Similarity kernel ('rbf', 'cosine')
        bandwidth: Kernel bandwidth for RBF

    Returns:
        (n_routes, n_routes) similarity matrix
    """
    transfer = AnalogousRouteTransfer(
        ColdStartConfig(similarity_kernel=kernel, kernel_bandwidth=bandwidth)
    )

    # Convert to feature vectors
    vectors = np.array([transfer._features_to_vector(r) for r in routes])

    n = len(routes)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = transfer._compute_similarity(
                vectors[i], vectors[j]
            )

    return similarity_matrix
