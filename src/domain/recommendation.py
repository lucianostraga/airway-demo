"""
Recommendation domain models.

Represents actionable recommendations for ULD repositioning and allocation.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, computed_field

from .uld import ULD, ULDType


class RecommendationPriority(str, Enum):
    """Priority level for recommendations."""

    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"  # Action needed within hours
    MEDIUM = "medium"  # Action needed within 24 hours
    LOW = "low"  # Optimization opportunity


class RecommendationType(str, Enum):
    """Type of recommendation."""

    REPOSITION = "reposition"  # Move ULD between stations
    ALLOCATE = "allocate"  # Assign ULD to flight
    REPAIR = "repair"  # Send ULD for maintenance
    LEASE = "lease"  # Lease additional ULDs
    RETURN_LEASE = "return_lease"  # Return leased ULDs


class CostBenefit(BaseModel):
    """
    Cost-benefit analysis for a recommendation.

    All costs in USD.
    """

    # Direct costs
    transportation_cost: float = 0.0  # Trucking/repositioning flight
    handling_cost: float = 0.0  # Ground handling
    opportunity_cost: float = 0.0  # Value of ULD if kept in place

    # Benefits / avoided costs
    avoided_shortage_cost: float = 0.0  # Cost of running out
    revenue_protected: float = 0.0  # Revenue enabled by this action
    avoided_delay_cost: float = 0.0  # Delay penalties avoided

    model_config = {"frozen": True}

    @computed_field
    @property
    def total_cost(self) -> float:
        """Total implementation cost."""
        return self.transportation_cost + self.handling_cost + self.opportunity_cost

    @computed_field
    @property
    def total_benefit(self) -> float:
        """Total benefit / avoided cost."""
        return self.avoided_shortage_cost + self.revenue_protected + self.avoided_delay_cost

    @computed_field
    @property
    def net_benefit(self) -> float:
        """Net benefit (positive is good)."""
        return self.total_benefit - self.total_cost

    @computed_field
    @property
    def roi(self) -> float | None:
        """Return on investment ratio."""
        if self.total_cost == 0:
            return None
        return self.net_benefit / self.total_cost


class RepositioningRecommendation(BaseModel):
    """
    Recommendation to move ULD(s) between stations.

    Can be via ground transport or on a flight.
    """

    recommendation_id: str
    recommendation_type: RecommendationType = RecommendationType.REPOSITION
    priority: RecommendationPriority

    # What to move
    uld_type: ULDType
    quantity: int
    specific_ulds: list[str] = Field(default_factory=list)  # Specific ULD IDs if known

    # Movement details
    origin_station: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
    destination_station: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
    transport_method: str = "flight"  # flight, ground, rail

    # Timing
    recommended_departure: datetime
    required_by: datetime  # Must arrive before this time
    estimated_duration_hours: float

    # Context
    reason: str
    shortage_probability_at_dest: float = 0.0
    surplus_at_origin: int = 0

    # Economics
    cost_benefit: CostBenefit

    # Confidence
    confidence_score: float = 0.8

    model_config = {"frozen": False}

    @computed_field
    @property
    def is_economic(self) -> bool:
        """True if net benefit is positive."""
        return self.cost_benefit.net_benefit > 0

    @computed_field
    @property
    def urgency_hours(self) -> float:
        """Hours until required_by deadline."""
        return (self.required_by - datetime.now()).total_seconds() / 3600


class AllocationRecommendation(BaseModel):
    """
    Recommendation to allocate ULD(s) to a flight.
    """

    recommendation_id: str
    recommendation_type: RecommendationType = RecommendationType.ALLOCATE
    priority: RecommendationPriority

    # Flight details
    flight_number: str
    flight_date: datetime
    origin: str
    destination: str

    # Allocation
    uld_type: ULDType
    quantity: int
    specific_ulds: list[str] = Field(default_factory=list)

    # Reasoning
    reason: str
    estimated_demand: int
    current_allocation: int

    # Confidence
    confidence_score: float = 0.8

    model_config = {"frozen": False}

    @computed_field
    @property
    def additional_needed(self) -> int:
        """Additional ULDs needed beyond current allocation."""
        return max(0, self.estimated_demand - self.current_allocation)


class FlightLoadPlan(BaseModel):
    """
    Complete ULD loading plan for a flight.
    """

    flight_number: str
    flight_date: datetime
    origin: str
    destination: str
    aircraft_type: str

    # Capacity
    available_positions: int
    max_cargo_weight_kg: float

    # Planned loading
    ulds_planned: list[str] = Field(default_factory=list)
    weight_planned_kg: float = 0.0

    # By type
    by_type: dict[ULDType, int] = Field(default_factory=dict)

    # Status
    is_complete: bool = False
    warnings: list[str] = Field(default_factory=list)

    model_config = {"frozen": False}

    @computed_field
    @property
    def utilization(self) -> float:
        """Position utilization percentage."""
        if self.available_positions == 0:
            return 0.0
        return len(self.ulds_planned) / self.available_positions

    @computed_field
    @property
    def weight_utilization(self) -> float:
        """Weight utilization percentage."""
        if self.max_cargo_weight_kg == 0:
            return 0.0
        return self.weight_planned_kg / self.max_cargo_weight_kg

    def add_uld(self, uld_id: str, uld_type: ULDType, weight_kg: float) -> bool:
        """Add a ULD to the plan. Returns False if no capacity."""
        if len(self.ulds_planned) >= self.available_positions:
            self.warnings.append(f"Cannot add {uld_id}: no position available")
            return False
        if self.weight_planned_kg + weight_kg > self.max_cargo_weight_kg:
            self.warnings.append(f"Cannot add {uld_id}: weight limit exceeded")
            return False

        self.ulds_planned.append(uld_id)
        self.weight_planned_kg += weight_kg
        self.by_type[uld_type] = self.by_type.get(uld_type, 0) + 1
        return True


class StationActionPlan(BaseModel):
    """
    Complete action plan for a station.

    Aggregates all recommendations into an actionable plan.
    """

    station: str
    generated_at: datetime
    valid_until: datetime

    # Situation assessment
    current_inventory: dict[ULDType, int] = Field(default_factory=dict)
    forecasted_demand: dict[ULDType, int] = Field(default_factory=dict)
    forecasted_supply: dict[ULDType, int] = Field(default_factory=dict)

    # Recommendations
    repositioning_out: list[RepositioningRecommendation] = Field(default_factory=list)
    repositioning_in: list[RepositioningRecommendation] = Field(default_factory=list)
    allocations: list[AllocationRecommendation] = Field(default_factory=list)

    # Summary metrics
    total_ulds_to_send: int = 0
    total_ulds_expected: int = 0
    estimated_end_inventory: dict[ULDType, int] = Field(default_factory=dict)

    # Economic summary
    total_cost: float = 0.0
    total_benefit: float = 0.0

    model_config = {"frozen": False}

    @computed_field
    @property
    def has_critical_actions(self) -> bool:
        """True if any critical priority recommendations."""
        all_recs = self.repositioning_out + self.repositioning_in
        return any(r.priority == RecommendationPriority.CRITICAL for r in all_recs)

    @computed_field
    @property
    def action_count(self) -> int:
        """Total number of recommended actions."""
        return len(self.repositioning_out) + len(self.repositioning_in) + len(self.allocations)


class NetworkOptimizationResult(BaseModel):
    """
    Result of network-wide ULD optimization.

    Output of the OR-Tools based optimizer.
    """

    generated_at: datetime
    optimization_horizon_hours: int
    solver_status: str  # optimal, feasible, infeasible, timeout

    # Solution
    repositioning_moves: list[RepositioningRecommendation]
    station_plans: dict[str, StationActionPlan]

    # Objective function components
    total_shortage_cost: float = 0.0
    total_repositioning_cost: float = 0.0
    total_handling_cost: float = 0.0
    objective_value: float = 0.0

    # Solve statistics
    solve_time_seconds: float = 0.0
    gap_percent: float | None = None  # Optimality gap

    # Constraints
    total_ulds_moved: int = 0
    max_moves_per_station: int = 0

    model_config = {"frozen": True}

    @computed_field
    @property
    def is_optimal(self) -> bool:
        """True if optimal solution found."""
        return self.solver_status == "optimal"

    @computed_field
    @property
    def total_system_cost(self) -> float:
        """Total cost across all components."""
        return self.total_shortage_cost + self.total_repositioning_cost + self.total_handling_cost

    def get_station_plan(self, station: str) -> StationActionPlan | None:
        """Get action plan for a specific station."""
        return self.station_plans.get(station)

    def get_moves_from(self, station: str) -> list[RepositioningRecommendation]:
        """Get all repositioning moves originating from a station."""
        return [m for m in self.repositioning_moves if m.origin_station == station]

    def get_moves_to(self, station: str) -> list[RepositioningRecommendation]:
        """Get all repositioning moves destined for a station."""
        return [m for m in self.repositioning_moves if m.destination_station == station]
