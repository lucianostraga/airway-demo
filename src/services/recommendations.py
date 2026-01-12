"""
Recommendation Service.

Generates actionable recommendations for ULD repositioning and allocation.
"""

from datetime import datetime, timedelta, timezone
from typing import Any
import uuid

from src.domain import (
    ULDType,
    RepositioningRecommendation,
    AllocationRecommendation,
    StationActionPlan,
    CostBenefit,
    RecommendationPriority,
    RecommendationType,
    ImbalanceForecast,
    DemandForecast,
    SupplyForecast,
    NetworkForecast,
    DELTA_STATIONS,
)


class RecommendationService:
    """
    Service for generating ULD management recommendations.

    Analyzes forecasts and current state to recommend:
    - Repositioning moves between stations
    - Flight allocations
    - Action priorities

    Usage:
        service = RecommendationService()
        recs = await service.generate_repositioning_recommendations(network_forecast)
    """

    # Cost parameters (USD)
    COST_PARAMS = {
        "ground_transport_per_uld": 150.0,  # Trucking between stations
        "flight_transport_per_uld": 50.0,  # Deadhead on flight
        "handling_per_uld": 25.0,  # Ground handling at each station
        "shortage_cost_per_uld": 500.0,  # Cost of running out
        "delay_cost_per_minute": 100.0,  # Flight delay cost
    }

    # Priority thresholds
    SHORTAGE_PROB_CRITICAL = 0.8
    SHORTAGE_PROB_HIGH = 0.5
    SHORTAGE_PROB_MEDIUM = 0.3

    def __init__(self):
        self._rec_counter = 0

    def _generate_rec_id(self) -> str:
        """Generate unique recommendation ID."""
        self._rec_counter += 1
        return f"REC-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{self._rec_counter:05d}"

    async def generate_repositioning_recommendations(
        self,
        network_forecast: NetworkForecast,
    ) -> list[RepositioningRecommendation]:
        """
        Generate repositioning recommendations from network forecast.

        Identifies shortage and surplus stations and recommends moves.

        Args:
            network_forecast: Network-wide forecast

        Returns:
            List of repositioning recommendations
        """
        recommendations = []

        shortage_stations = network_forecast.shortage_stations
        surplus_stations = network_forecast.surplus_stations

        # Match shortages with surpluses
        for shortage_station in shortage_stations:
            imbalance = network_forecast.station_imbalance.get(shortage_station)
            if not imbalance:
                continue

            shortage_qty = abs(int(imbalance.total_imbalance.q50))
            if shortage_qty == 0:
                continue

            # Find best surplus station to source from
            best_source = None
            best_score = -1

            for surplus_station in surplus_stations:
                surplus_imbalance = network_forecast.station_imbalance.get(surplus_station)
                if not surplus_imbalance:
                    continue

                surplus_qty = int(surplus_imbalance.total_imbalance.q50)
                if surplus_qty <= 0:
                    continue

                # Score based on surplus quantity and "distance" (simplified)
                score = surplus_qty  # Would factor in transport cost

                if score > best_score:
                    best_score = score
                    best_source = surplus_station

            if best_source:
                # Determine priority
                priority = self._determine_priority(imbalance.shortage_probability)

                # Calculate cost/benefit
                qty = min(shortage_qty, int(best_score))
                cost_benefit = self._calculate_cost_benefit(
                    quantity=qty,
                    shortage_prob=imbalance.shortage_probability,
                    transport_method="flight",
                )

                rec = RepositioningRecommendation(
                    recommendation_id=self._generate_rec_id(),
                    priority=priority,
                    uld_type=ULDType.AKE,  # Would analyze by type
                    quantity=qty,
                    origin_station=best_source,
                    destination_station=shortage_station,
                    transport_method="flight",
                    recommended_departure=datetime.now(timezone.utc) + timedelta(hours=2),
                    required_by=datetime.now(timezone.utc) + timedelta(hours=12),
                    estimated_duration_hours=3.0,
                    reason=f"Shortage forecast at {shortage_station} (P={imbalance.shortage_probability:.0%})",
                    shortage_probability_at_dest=imbalance.shortage_probability,
                    surplus_at_origin=int(best_score),
                    cost_benefit=cost_benefit,
                )
                recommendations.append(rec)

        return recommendations

    async def generate_allocation_recommendations(
        self,
        station: str,
        demand_forecast: DemandForecast,
        current_allocations: dict[str, int],  # flight_number -> allocated ULDs
    ) -> list[AllocationRecommendation]:
        """
        Generate allocation recommendations for flights at a station.

        Args:
            station: Station code
            demand_forecast: Demand forecast
            current_allocations: Current ULD allocations to flights

        Returns:
            List of allocation recommendations
        """
        recommendations = []

        # This would analyze specific flight demands
        # Simplified: recommend allocations for any under-allocated flights

        for flight_number, allocated in current_allocations.items():
            estimated = 8  # Would calculate from passenger/cargo booking

            if allocated < estimated:
                priority = (
                    RecommendationPriority.HIGH
                    if estimated - allocated > 4
                    else RecommendationPriority.MEDIUM
                )

                rec = AllocationRecommendation(
                    recommendation_id=self._generate_rec_id(),
                    priority=priority,
                    flight_number=flight_number,
                    flight_date=datetime.now(timezone.utc),
                    origin=station,
                    destination="XXX",  # Would look up
                    uld_type=ULDType.AKE,
                    quantity=estimated - allocated,
                    reason=f"Under-allocated by {estimated - allocated} ULDs",
                    estimated_demand=estimated,
                    current_allocation=allocated,
                )
                recommendations.append(rec)

        return recommendations

    async def generate_station_action_plan(
        self,
        station: str,
        network_forecast: NetworkForecast,
    ) -> StationActionPlan:
        """
        Generate complete action plan for a station.

        Args:
            station: Station code
            network_forecast: Network forecast

        Returns:
            StationActionPlan with all recommendations
        """
        now = datetime.now(timezone.utc)

        # Get station forecasts
        demand = network_forecast.station_demand.get(station)
        supply = network_forecast.station_supply.get(station)
        imbalance = network_forecast.station_imbalance.get(station)

        # Get current inventory (from supply forecast)
        current_inventory = {}
        forecasted_demand = {}
        forecasted_supply = {}

        if supply:
            # Estimate current inventory from supply
            for uld_type in ULDType:
                if uld_type in supply.supply_by_type:
                    current_inventory[uld_type] = int(supply.supply_by_type[uld_type].q50)

        if demand:
            for uld_type in ULDType:
                if uld_type in demand.demand_by_type:
                    forecasted_demand[uld_type] = int(demand.demand_by_type[uld_type].q50)

        if supply:
            for uld_type in ULDType:
                if uld_type in supply.supply_by_type:
                    forecasted_supply[uld_type] = int(supply.supply_by_type[uld_type].q50)

        # Get repositioning recommendations affecting this station
        all_recs = await self.generate_repositioning_recommendations(network_forecast)

        repositioning_out = [r for r in all_recs if r.origin_station == station]
        repositioning_in = [r for r in all_recs if r.destination_station == station]

        # Calculate totals
        total_to_send = sum(r.quantity for r in repositioning_out)
        total_expected = sum(r.quantity for r in repositioning_in)

        # Estimate end inventory
        estimated_end = {}
        for uld_type, current in current_inventory.items():
            demand_qty = forecasted_demand.get(uld_type, 0)
            supply_qty = forecasted_supply.get(uld_type, current)
            estimated_end[uld_type] = max(0, supply_qty - demand_qty)

        # Total costs/benefits
        total_cost = sum(r.cost_benefit.total_cost for r in repositioning_out)
        total_benefit = sum(r.cost_benefit.total_benefit for r in repositioning_in)

        return StationActionPlan(
            station=station,
            generated_at=now,
            valid_until=now + timedelta(hours=4),
            current_inventory=current_inventory,
            forecasted_demand=forecasted_demand,
            forecasted_supply=forecasted_supply,
            repositioning_out=repositioning_out,
            repositioning_in=repositioning_in,
            allocations=[],  # Would populate from allocation recommendations
            total_ulds_to_send=total_to_send,
            total_ulds_expected=total_expected,
            estimated_end_inventory=estimated_end,
            total_cost=total_cost,
            total_benefit=total_benefit,
        )

    def _determine_priority(self, shortage_prob: float) -> RecommendationPriority:
        """Determine recommendation priority from shortage probability."""
        if shortage_prob >= self.SHORTAGE_PROB_CRITICAL:
            return RecommendationPriority.CRITICAL
        elif shortage_prob >= self.SHORTAGE_PROB_HIGH:
            return RecommendationPriority.HIGH
        elif shortage_prob >= self.SHORTAGE_PROB_MEDIUM:
            return RecommendationPriority.MEDIUM
        else:
            return RecommendationPriority.LOW

    def _calculate_cost_benefit(
        self,
        quantity: int,
        shortage_prob: float,
        transport_method: str,
    ) -> CostBenefit:
        """Calculate cost-benefit analysis for a repositioning move."""
        # Transportation cost
        if transport_method == "ground":
            transport_cost = quantity * self.COST_PARAMS["ground_transport_per_uld"]
        else:
            transport_cost = quantity * self.COST_PARAMS["flight_transport_per_uld"]

        # Handling cost (both ends)
        handling_cost = quantity * self.COST_PARAMS["handling_per_uld"] * 2

        # Avoided shortage cost
        avoided_shortage = (
            quantity * shortage_prob * self.COST_PARAMS["shortage_cost_per_uld"]
        )

        return CostBenefit(
            transportation_cost=transport_cost,
            handling_cost=handling_cost,
            opportunity_cost=0.0,  # Would calculate from alternative uses
            avoided_shortage_cost=avoided_shortage,
            revenue_protected=avoided_shortage * 0.5,  # Rough estimate
            avoided_delay_cost=shortage_prob * 30 * self.COST_PARAMS["delay_cost_per_minute"],
        )


class RecommendationPrioritizer:
    """
    Prioritizes and filters recommendations based on constraints.
    """

    def prioritize(
        self,
        recommendations: list[RepositioningRecommendation],
        max_moves: int = 10,
        min_roi: float = 0.0,
    ) -> list[RepositioningRecommendation]:
        """
        Prioritize and filter recommendations.

        Args:
            recommendations: All recommendations
            max_moves: Maximum number of moves to return
            min_roi: Minimum ROI threshold

        Returns:
            Prioritized list of recommendations
        """
        # Filter by ROI
        filtered = [
            r for r in recommendations
            if r.cost_benefit.roi is None or r.cost_benefit.roi >= min_roi
        ]

        # Sort by priority then by shortage probability
        priority_order = {
            RecommendationPriority.CRITICAL: 0,
            RecommendationPriority.HIGH: 1,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 3,
        }

        sorted_recs = sorted(
            filtered,
            key=lambda r: (
                priority_order.get(r.priority, 4),
                -r.shortage_probability_at_dest,
                -r.cost_benefit.net_benefit,
            ),
        )

        return sorted_recs[:max_moves]

    def deduplicate(
        self,
        recommendations: list[RepositioningRecommendation],
    ) -> list[RepositioningRecommendation]:
        """
        Remove duplicate recommendations for the same origin-destination pair.
        """
        seen = set()
        deduped = []

        for rec in recommendations:
            key = (rec.origin_station, rec.destination_station, rec.uld_type)
            if key not in seen:
                seen.add(key)
                deduped.append(rec)

        return deduped
