"""
Domain models for ULD Forecasting System.

Core business entities representing the airline ULD operations domain.
All models use Pydantic for validation and serialization.
"""

from .uld import ULD, ULDType, ULDStatus, ULDPosition, ULDInventory
from .station import Station, StationTier, StationCapacity, DELTA_STATIONS
from .flight import Flight, FlightStatus, FlightSchedule, Route, AircraftType, AIRCRAFT_TYPES
from .forecast import (
    DemandForecast,
    SupplyForecast,
    ImbalanceForecast,
    NetworkForecast,
    ForecastGranularity,
    ForecastConfidence,
    ForecastEvaluation,
    QuantileForecast,
)
from .recommendation import (
    RepositioningRecommendation,
    AllocationRecommendation,
    FlightLoadPlan,
    StationActionPlan,
    NetworkOptimizationResult,
    CostBenefit,
    RecommendationPriority,
    RecommendationType,
)

__all__ = [
    # ULD
    "ULD",
    "ULDType",
    "ULDStatus",
    "ULDPosition",
    "ULDInventory",
    # Station
    "Station",
    "StationTier",
    "StationCapacity",
    "DELTA_STATIONS",
    # Flight
    "Flight",
    "FlightStatus",
    "FlightSchedule",
    "Route",
    "AircraftType",
    "AIRCRAFT_TYPES",
    # Forecast
    "DemandForecast",
    "SupplyForecast",
    "ImbalanceForecast",
    "NetworkForecast",
    "ForecastGranularity",
    "ForecastConfidence",
    "ForecastEvaluation",
    "QuantileForecast",
    # Recommendation
    "RepositioningRecommendation",
    "AllocationRecommendation",
    "FlightLoadPlan",
    "StationActionPlan",
    "NetworkOptimizationResult",
    "CostBenefit",
    "RecommendationPriority",
    "RecommendationType",
]
