"""
Forecast domain models.

Represents demand and supply forecasts at various granularities.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, computed_field

from .uld import ULDType


class ForecastGranularity(str, Enum):
    """Time granularity for forecasts."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"


class ForecastConfidence(str, Enum):
    """Confidence level categories."""

    HIGH = "high"  # 80%+ confidence
    MEDIUM = "medium"  # 60-80% confidence
    LOW = "low"  # <60% confidence


class QuantileForecast(BaseModel):
    """
    Quantile-based probabilistic forecast.

    Uses Conformalized Quantile Regression (CQR) for calibrated intervals.
    """

    q05: float  # 5th percentile
    q25: float  # 25th percentile (lower quartile)
    q50: float  # 50th percentile (median)
    q75: float  # 75th percentile (upper quartile)
    q95: float  # 95th percentile

    model_config = {"frozen": True}

    @computed_field
    @property
    def point_estimate(self) -> float:
        """Return median as point estimate."""
        return self.q50

    @computed_field
    @property
    def uncertainty_range(self) -> float:
        """90% prediction interval width."""
        return self.q95 - self.q05

    @computed_field
    @property
    def iqr(self) -> float:
        """Interquartile range (50% interval)."""
        return self.q75 - self.q25


class DemandForecast(BaseModel):
    """
    ULD demand forecast for a station-time combination.

    Predicts how many ULDs will be needed based on:
    - Flight schedule
    - Passenger bookings
    - Cargo bookings
    - Historical patterns
    - Weather/events
    """

    station: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
    forecast_time: datetime
    generated_at: datetime
    granularity: ForecastGranularity

    # Demand by ULD type (probabilistic)
    demand_by_type: dict[ULDType, QuantileForecast]

    # Total demand (all types)
    total_demand: QuantileForecast

    # Contributing factors
    scheduled_departures: int = 0
    booked_passengers: int = 0
    cargo_weight_kg: float = 0.0

    # Confidence and anomalies
    confidence: ForecastConfidence = ForecastConfidence.MEDIUM
    is_anomaly: bool = False
    anomaly_score: float = 0.0

    # Model metadata
    model_version: str = "v1"
    features_used: list[str] = Field(default_factory=list)

    model_config = {"frozen": True}

    @computed_field
    @property
    def total_point_estimate(self) -> float:
        """Total demand point estimate."""
        return self.total_demand.point_estimate

    @computed_field
    @property
    def high_uncertainty(self) -> bool:
        """Flag if uncertainty is unusually high."""
        return self.total_demand.uncertainty_range > self.total_demand.q50 * 0.5


class SupplyForecast(BaseModel):
    """
    ULD supply forecast for a station-time combination.

    Predicts available ULD inventory based on:
    - Current inventory
    - Expected arrivals from incoming flights
    - Expected departures
    - Maintenance schedules
    """

    station: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
    forecast_time: datetime
    generated_at: datetime
    granularity: ForecastGranularity

    # Supply by ULD type (probabilistic)
    supply_by_type: dict[ULDType, QuantileForecast]

    # Total supply
    total_supply: QuantileForecast

    # Movement components
    current_inventory: int = 0
    expected_arrivals: int = 0
    expected_departures: int = 0
    in_maintenance: int = 0

    # Confidence
    confidence: ForecastConfidence = ForecastConfidence.MEDIUM

    model_config = {"frozen": True}

    @computed_field
    @property
    def net_change(self) -> int:
        """Expected net change in inventory."""
        return self.expected_arrivals - self.expected_departures


class ImbalanceForecast(BaseModel):
    """
    Supply-demand imbalance forecast.

    Negative values indicate shortage, positive indicates surplus.
    """

    station: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
    forecast_time: datetime
    generated_at: datetime

    # Imbalance by type
    imbalance_by_type: dict[ULDType, QuantileForecast]

    # Total imbalance
    total_imbalance: QuantileForecast

    # Shortage probability (P(supply < demand))
    shortage_probability: float = 0.0

    # Severity classification
    severity: str = "normal"  # normal, warning, critical

    model_config = {"frozen": True}

    @computed_field
    @property
    def is_shortage_likely(self) -> bool:
        """True if shortage probability > 50%."""
        return self.shortage_probability > 0.5

    @computed_field
    @property
    def is_critical(self) -> bool:
        """True if critical shortage expected."""
        return self.severity == "critical" or self.shortage_probability > 0.8


class NetworkForecast(BaseModel):
    """
    Network-wide forecast combining all stations.

    Uses MinT reconciliation to ensure hierarchical consistency.
    """

    generated_at: datetime
    forecast_horizon_hours: int
    granularity: ForecastGranularity

    # Station-level forecasts
    station_demand: dict[str, DemandForecast]
    station_supply: dict[str, SupplyForecast]
    station_imbalance: dict[str, ImbalanceForecast]

    # Network totals
    total_network_demand: QuantileForecast
    total_network_supply: QuantileForecast

    # Stations with issues
    shortage_stations: list[str] = Field(default_factory=list)
    surplus_stations: list[str] = Field(default_factory=list)

    # Reconciliation metadata
    is_reconciled: bool = False
    reconciliation_adjustments: dict[str, float] = Field(default_factory=dict)

    model_config = {"frozen": True}

    @computed_field
    @property
    def network_balanced(self) -> bool:
        """True if network supply roughly equals demand."""
        supply = self.total_network_supply.point_estimate
        demand = self.total_network_demand.point_estimate
        return abs(supply - demand) < demand * 0.1

    def get_priority_stations(self, top_n: int = 5) -> list[str]:
        """Get stations most in need of attention."""
        # Sort by shortage probability
        priorities = []
        for station, imbalance in self.station_imbalance.items():
            priorities.append((station, imbalance.shortage_probability))
        priorities.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in priorities[:top_n]]


class ForecastEvaluation(BaseModel):
    """
    Forecast evaluation metrics.

    Tracks accuracy and calibration of forecasts.
    """

    station: str | None = None  # None for network-level
    evaluation_period_start: datetime
    evaluation_period_end: datetime
    n_forecasts: int

    # Point forecast accuracy
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    mape: float | None = None  # Mean Absolute Percentage Error (if > 0)

    # Probabilistic calibration
    coverage_90: float  # % of actuals in 90% PI
    coverage_50: float  # % of actuals in 50% PI

    # Sharpness (narrower is better, given calibration)
    mean_interval_width_90: float
    mean_interval_width_50: float

    # Skill scores
    crps: float | None = None  # Continuous Ranked Probability Score
    pinball_loss: dict[str, float] = Field(default_factory=dict)

    model_config = {"frozen": True}

    @computed_field
    @property
    def is_well_calibrated(self) -> bool:
        """True if coverage is within 5% of nominal."""
        return 0.85 <= self.coverage_90 <= 0.95 and 0.45 <= self.coverage_50 <= 0.55
