"""
Hierarchical Forecasting Reconciliation

Implements optimal reconciliation methods to ensure forecast coherence
across the network hierarchy:

Network (total) -> Hub Region -> Station -> Route -> ULD Type

Methods:
1. Bottom-up: Aggregate bottom-level forecasts
2. Top-down: Disaggregate top-level proportionally
3. MinT (Minimum Trace): Optimal linear combination

Reference:
Wickramasuriya, Athanasopoulos, Hyndman (2019).
Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series.
Journal of the American Statistical Association.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class ReconciliationMethod(Enum):
    """Available reconciliation methods."""

    BOTTOM_UP = "bottom_up"
    TOP_DOWN = "top_down"
    OLS = "ols"
    WLS_STRUCTURAL = "wls_struct"
    WLS_VARIANCE = "wls_var"
    MINT_SHRINK = "mint_shrink"


@dataclass
class HierarchySpec:
    """Specification of the hierarchical structure."""

    # Levels from bottom to top
    levels: list[str]  # e.g., ['route_uld', 'station', 'hub', 'network']

    # Mapping from each level to its parent
    # e.g., {'ATL-JFK-AKE': 'ATL', 'ATL': 'ATL-region', 'ATL-region': 'network'}
    parent_map: dict[str, str]

    # All nodes at each level
    nodes_by_level: dict[str, list[str]]


class SummingMatrix:
    """
    Constructs and manages the summing matrix S.

    The summing matrix S maps bottom-level series to all series:
    y = S * b

    where:
    - y: Vector of all forecasts (n_total x 1)
    - S: Summing matrix (n_total x n_bottom)
    - b: Bottom-level forecasts (n_bottom x 1)

    Each row of S indicates which bottom-level series sum to that series.
    """

    def __init__(self, hierarchy: HierarchySpec):
        self.hierarchy = hierarchy
        self._S: Optional[np.ndarray] = None
        self._node_to_idx: dict[str, int] = {}
        self._bottom_to_idx: dict[str, int] = {}

    def build(self) -> np.ndarray:
        """
        Construct the summing matrix S.

        Returns:
            S: (n_total x n_bottom) summing matrix
        """
        # Get all nodes in hierarchical order (top to bottom)
        all_nodes = []
        for level in reversed(self.hierarchy.levels):
            all_nodes.extend(self.hierarchy.nodes_by_level[level])

        # Create node-to-index mapping
        self._node_to_idx = {node: i for i, node in enumerate(all_nodes)}

        # Bottom level is first in the levels list
        bottom_level = self.hierarchy.levels[0]
        bottom_nodes = self.hierarchy.nodes_by_level[bottom_level]
        self._bottom_to_idx = {node: i for i, node in enumerate(bottom_nodes)}

        n_total = len(all_nodes)
        n_bottom = len(bottom_nodes)

        # Initialize summing matrix
        S = np.zeros((n_total, n_bottom))

        # Fill in the matrix
        for node in all_nodes:
            row_idx = self._node_to_idx[node]

            if node in bottom_nodes:
                # Bottom-level: just itself
                col_idx = self._bottom_to_idx[node]
                S[row_idx, col_idx] = 1.0
            else:
                # Aggregate: sum of all bottom-level descendants
                descendants = self._get_bottom_descendants(node)
                for desc in descendants:
                    col_idx = self._bottom_to_idx[desc]
                    S[row_idx, col_idx] = 1.0

        self._S = S
        return S

    def _get_bottom_descendants(self, node: str) -> list[str]:
        """Get all bottom-level descendants of a node."""
        bottom_level = self.hierarchy.levels[0]
        bottom_nodes = set(self.hierarchy.nodes_by_level[bottom_level])

        # BFS to find all descendants
        descendants = []
        to_visit = [node]

        # Invert parent_map to get children
        children_map: dict[str, list[str]] = {}
        for child, parent in self.hierarchy.parent_map.items():
            if parent not in children_map:
                children_map[parent] = []
            children_map[parent].append(child)

        while to_visit:
            current = to_visit.pop(0)
            if current in bottom_nodes:
                descendants.append(current)
            elif current in children_map:
                to_visit.extend(children_map[current])

        return descendants

    @property
    def S(self) -> np.ndarray:
        """Get the summing matrix."""
        if self._S is None:
            self.build()
        return self._S


class HierarchicalReconciler:
    """
    Reconciles forecasts to ensure hierarchical coherence.

    Given base forecasts y_hat at all levels, computes reconciled forecasts:
    y_tilde = S * G * y_hat

    where G is the reconciliation matrix that depends on the method:
    - OLS: G = (S'S)^{-1} S'
    - WLS: G = (S'W^{-1}S)^{-1} S'W^{-1}
    - MinT: G uses shrunk covariance matrix

    The reconciled forecasts are coherent: they satisfy the hierarchy constraints.
    """

    def __init__(
        self,
        hierarchy: HierarchySpec,
        method: ReconciliationMethod = ReconciliationMethod.MINT_SHRINK,
    ):
        self.hierarchy = hierarchy
        self.method = method
        self._summing = SummingMatrix(hierarchy)
        self._G: Optional[np.ndarray] = None
        self._W: Optional[np.ndarray] = None

    def fit(
        self,
        residuals: Optional[np.ndarray] = None,
        shrinkage_target: str = "diagonal",
    ) -> "HierarchicalReconciler":
        """
        Fit the reconciliation matrix.

        For variance-weighted methods, uses historical residuals to estimate W.

        Args:
            residuals: (n_obs x n_series) matrix of historical forecast errors
            shrinkage_target: Target for shrinkage estimator ('diagonal' or 'identity')
        """
        S = self._summing.S
        n_total, n_bottom = S.shape

        if self.method == ReconciliationMethod.BOTTOM_UP:
            # G selects bottom-level rows
            self._G = np.eye(n_bottom)

        elif self.method == ReconciliationMethod.OLS:
            # G = (S'S)^{-1} S'
            self._G = np.linalg.solve(S.T @ S, S.T)

        elif self.method == ReconciliationMethod.WLS_STRUCTURAL:
            # W = diag(S * 1), proportional to aggregation level
            w = np.sum(S, axis=1)
            W_inv = np.diag(1 / w)
            self._W = np.diag(w)
            self._G = np.linalg.solve(S.T @ W_inv @ S, S.T @ W_inv)

        elif self.method == ReconciliationMethod.WLS_VARIANCE:
            if residuals is None:
                raise ValueError("WLS-variance requires residuals")
            # W = diag(sample variances)
            variances = np.var(residuals, axis=0)
            variances = np.maximum(variances, 1e-6)  # Avoid division by zero
            W_inv = np.diag(1 / variances)
            self._W = np.diag(variances)
            self._G = np.linalg.solve(S.T @ W_inv @ S, S.T @ W_inv)

        elif self.method == ReconciliationMethod.MINT_SHRINK:
            if residuals is None:
                raise ValueError("MinT-shrink requires residuals")
            # Shrinkage estimator for covariance
            W = self._shrink_covariance(residuals, target=shrinkage_target)
            self._W = W
            W_inv = np.linalg.inv(W)
            self._G = np.linalg.solve(S.T @ W_inv @ S, S.T @ W_inv)

        return self

    def _shrink_covariance(
        self,
        residuals: np.ndarray,
        target: str = "diagonal",
    ) -> np.ndarray:
        """
        Compute shrinkage estimator for covariance matrix.

        Uses the Ledoit-Wolf shrinkage formula:
        W_shrunk = (1 - alpha) * W_sample + alpha * W_target

        where alpha is chosen to minimize expected loss.

        Args:
            residuals: (n_obs x n_series) residual matrix
            target: 'diagonal' (sample variances) or 'identity'

        Returns:
            Shrunk covariance matrix
        """
        n, p = residuals.shape

        # Sample covariance
        sample_cov = np.cov(residuals, rowvar=False)

        # Target matrix
        if target == "diagonal":
            target_cov = np.diag(np.diag(sample_cov))
        else:  # identity
            target_cov = np.eye(p) * np.trace(sample_cov) / p

        # Ledoit-Wolf shrinkage intensity
        # Simplified formula - full formula involves fourth moments
        X_centered = residuals - residuals.mean(axis=0)

        # Frobenius norms
        delta = sample_cov - target_cov
        delta_norm_sq = np.sum(delta**2)

        if delta_norm_sq < 1e-10:
            return sample_cov

        # Estimate shrinkage intensity
        # Using simplified estimator for computational efficiency
        sum_sq = np.sum(X_centered**2, axis=0)
        sum_fourth = np.sum(X_centered**4, axis=0)

        # Shrinkage intensity (approximate)
        alpha = min(1.0, max(0.0, (1 / (n * (n - 1))) * (
            np.sum(sum_fourth) / p - np.sum(sum_sq)**2 / (n * p)
        ) / delta_norm_sq))

        return (1 - alpha) * sample_cov + alpha * target_cov

    def reconcile(self, base_forecasts: np.ndarray) -> np.ndarray:
        """
        Reconcile base forecasts to be coherent.

        Args:
            base_forecasts: (n_series,) or (n_points, n_series) base forecasts

        Returns:
            Reconciled forecasts with same shape
        """
        if self._G is None:
            raise ValueError("Reconciler not fitted. Call fit() first.")

        S = self._summing.S

        # Handle both 1D and 2D input
        is_1d = base_forecasts.ndim == 1
        if is_1d:
            base_forecasts = base_forecasts.reshape(1, -1)

        # Reconcile: y_tilde = S @ G @ y_hat
        bottom_forecasts = base_forecasts @ self._G.T
        reconciled = bottom_forecasts @ S.T

        if is_1d:
            reconciled = reconciled.flatten()

        return reconciled

    def reconcile_probabilistic(
        self,
        base_forecasts: np.ndarray,
        base_covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconcile probabilistic forecasts.

        Propagates uncertainty through the reconciliation:
        y_tilde ~ N(S @ G @ mu, S @ G @ Sigma @ G' @ S')

        Args:
            base_forecasts: Mean forecasts (n_series,)
            base_covariance: Forecast covariance (n_series x n_series)

        Returns:
            (reconciled_mean, reconciled_covariance)
        """
        if self._G is None:
            raise ValueError("Reconciler not fitted. Call fit() first.")

        S = self._summing.S

        # Reconciled mean
        reconciled_mean = S @ self._G @ base_forecasts

        # Reconciled covariance
        SG = S @ self._G
        reconciled_cov = SG @ base_covariance @ SG.T

        return reconciled_mean, reconciled_cov


class TemporalReconciler:
    """
    Temporal reconciliation for multi-horizon forecasts.

    Ensures that forecasts at different temporal aggregations are coherent:
    - Daily forecasts should sum to weekly
    - Weekly forecasts should sum to monthly

    Uses similar methodology to cross-sectional reconciliation.
    """

    def __init__(
        self,
        horizons: list[int],  # e.g., [1, 7, 30] for daily, weekly, monthly
        base_period: int = 1,  # Base period (e.g., 1 for daily)
    ):
        self.horizons = horizons
        self.base_period = base_period

    def build_temporal_summing_matrix(self) -> np.ndarray:
        """
        Build summing matrix for temporal aggregation.

        For horizons [1, 7, 30]:
        - Daily (1): 30 bottom-level series
        - Weekly (7): 4 aggregated series (days 1-7, 8-14, 15-21, 22-28) + 2 partial
        - Monthly (30): 1 top-level series
        """
        # Simplified: assume 30-day period
        n_days = max(self.horizons)

        # Count series at each level
        n_bottom = n_days  # Daily

        # Build aggregation structure
        aggregations = []

        for h in sorted(self.horizons, reverse=True):
            if h == self.base_period:
                continue
            n_aggs = n_days // h
            for i in range(n_aggs):
                start_day = i * h
                end_day = (i + 1) * h
                aggregations.append((start_day, end_day))

        # Total series
        n_total = n_bottom + len(aggregations) + 1  # +1 for total

        # Build S matrix
        S = np.zeros((n_total, n_bottom))

        # Bottom level (identity for first n_bottom rows)
        for i in range(n_bottom):
            S[i, i] = 1.0

        # Aggregations
        for idx, (start, end) in enumerate(aggregations):
            row = n_bottom + idx
            for day in range(start, end):
                S[row, day] = 1.0

        # Total (last row)
        S[-1, :] = 1.0

        return S

    def reconcile(
        self,
        daily_forecasts: np.ndarray,
        weekly_forecasts: np.ndarray,
        monthly_forecast: float,
    ) -> dict[str, np.ndarray]:
        """
        Reconcile temporal forecasts.

        Args:
            daily_forecasts: (30,) daily forecasts
            weekly_forecasts: (4,) weekly forecasts
            monthly_forecast: Single monthly forecast

        Returns:
            Dictionary with reconciled forecasts at each level
        """
        # Stack all forecasts
        base = np.concatenate([
            daily_forecasts,
            weekly_forecasts,
            [monthly_forecast],
        ])

        S = self.build_temporal_summing_matrix()

        # OLS reconciliation
        G = np.linalg.solve(S.T @ S, S.T)
        bottom = G @ base
        reconciled = S @ bottom

        n_daily = len(daily_forecasts)
        n_weekly = len(weekly_forecasts)

        return {
            "daily": reconciled[:n_daily],
            "weekly": reconciled[n_daily:n_daily + n_weekly],
            "monthly": reconciled[-1],
        }
