"""
Network Optimization Service.

Uses mathematical optimization to find optimal ULD repositioning across the network.
Implements a network flow model to minimize total cost while meeting demand.
"""

from datetime import datetime, timedelta, timezone
from typing import Any
import numpy as np

from src.domain import (
    ULDType,
    RepositioningRecommendation,
    StationActionPlan,
    NetworkOptimizationResult,
    CostBenefit,
    RecommendationPriority,
    NetworkForecast,
    DELTA_STATIONS,
)


class NetworkOptimizer:
    """
    Network-wide ULD optimization using mathematical programming.

    Solves a minimum cost network flow problem:
    - Nodes: Stations
    - Arcs: Possible repositioning routes
    - Supply: Surplus stations
    - Demand: Shortage stations
    - Objective: Minimize total repositioning + shortage cost

    For production, this would use OR-Tools or similar solver.
    This implementation provides a heuristic solution.

    Usage:
        optimizer = NetworkOptimizer()
        result = await optimizer.optimize(network_forecast)
    """

    # Cost parameters
    TRANSPORT_COST_PER_ULD_PER_KM = 0.05  # USD
    HANDLING_COST_PER_ULD = 25.0
    SHORTAGE_COST_PER_ULD = 500.0

    # Approximate distances between hubs (km)
    DISTANCES = {
        ("ATL", "DTW"): 1000,
        ("ATL", "MSP"): 1500,
        ("ATL", "SLC"): 2500,
        ("ATL", "JFK"): 1200,
        ("ATL", "LAX"): 3100,
        ("DTW", "MSP"): 900,
        ("DTW", "SLC"): 2200,
        ("DTW", "JFK"): 800,
        ("MSP", "SLC"): 1700,
        ("MSP", "SEA"): 2500,
        ("SLC", "LAX"): 1000,
        ("SLC", "SEA"): 1100,
        ("JFK", "BOS"): 300,
        ("JFK", "LAX"): 4000,
        ("LAX", "SEA"): 1500,
    }

    def __init__(self, max_iterations: int = 100):
        self.max_iterations = max_iterations
        self._rec_counter = 0

    def _get_distance(self, origin: str, destination: str) -> float:
        """Get distance between stations (symmetric)."""
        if origin == destination:
            return 0
        key = (origin, destination)
        reverse_key = (destination, origin)
        return self.DISTANCES.get(key, self.DISTANCES.get(reverse_key, 1500))

    def _generate_rec_id(self) -> str:
        """Generate unique recommendation ID."""
        self._rec_counter += 1
        return f"OPT-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{self._rec_counter:05d}"

    async def optimize(
        self,
        network_forecast: NetworkForecast,
        max_moves_per_station: int = 5,
        optimization_horizon_hours: int = 24,
    ) -> NetworkOptimizationResult:
        """
        Optimize ULD repositioning across the network.

        Args:
            network_forecast: Network forecast with imbalances
            max_moves_per_station: Maximum repositioning moves per station
            optimization_horizon_hours: Optimization horizon

        Returns:
            NetworkOptimizationResult with optimal repositioning moves
        """
        start_time = datetime.now(timezone.utc)

        # Extract imbalances
        shortages = {}  # station -> shortage quantity
        surpluses = {}  # station -> surplus quantity

        for station, imbalance in network_forecast.station_imbalance.items():
            if imbalance.total_imbalance.q50 < 0:  # Shortage
                shortages[station] = abs(int(imbalance.total_imbalance.q50))
            elif imbalance.total_imbalance.q50 > 5:  # Meaningful surplus
                surpluses[station] = int(imbalance.total_imbalance.q50)

        # Solve using greedy heuristic (would use OR-Tools in production)
        moves = self._solve_greedy(shortages, surpluses, max_moves_per_station)

        # Convert to recommendations
        recommendations = []
        for move in moves:
            origin, destination, quantity = move

            # Get imbalance info for priority
            dest_imbalance = network_forecast.station_imbalance.get(destination)
            shortage_prob = dest_imbalance.shortage_probability if dest_imbalance else 0.5

            priority = (
                RecommendationPriority.CRITICAL if shortage_prob > 0.8 else
                RecommendationPriority.HIGH if shortage_prob > 0.5 else
                RecommendationPriority.MEDIUM
            )

            # Calculate costs
            distance = self._get_distance(origin, destination)
            transport_cost = quantity * distance * self.TRANSPORT_COST_PER_ULD_PER_KM
            handling_cost = quantity * self.HANDLING_COST_PER_ULD * 2

            cost_benefit = CostBenefit(
                transportation_cost=transport_cost,
                handling_cost=handling_cost,
                avoided_shortage_cost=quantity * shortage_prob * self.SHORTAGE_COST_PER_ULD,
                revenue_protected=quantity * shortage_prob * 250,
            )

            rec = RepositioningRecommendation(
                recommendation_id=self._generate_rec_id(),
                priority=priority,
                uld_type=ULDType.AKE,
                quantity=quantity,
                origin_station=origin,
                destination_station=destination,
                transport_method="flight",
                recommended_departure=start_time + timedelta(hours=2),
                required_by=start_time + timedelta(hours=optimization_horizon_hours),
                estimated_duration_hours=distance / 800,  # ~800 km/h
                reason=f"Optimization: balance shortage at {destination}",
                shortage_probability_at_dest=shortage_prob,
                surplus_at_origin=surpluses.get(origin, 0),
                cost_benefit=cost_benefit,
            )
            recommendations.append(rec)

        # Build station plans
        station_plans = {}
        for station in network_forecast.station_imbalance.keys():
            plan = self._build_station_plan(
                station, recommendations, network_forecast
            )
            station_plans[station] = plan

        # Calculate totals
        total_shortage_cost = sum(
            abs(imb.total_imbalance.q05) * self.SHORTAGE_COST_PER_ULD * imb.shortage_probability
            for imb in network_forecast.station_imbalance.values()
            if imb.shortage_probability > 0.3
        )

        total_repositioning_cost = sum(
            r.cost_benefit.transportation_cost for r in recommendations
        )

        total_handling_cost = sum(
            r.cost_benefit.handling_cost for r in recommendations
        )

        solve_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        return NetworkOptimizationResult(
            generated_at=start_time,
            optimization_horizon_hours=optimization_horizon_hours,
            solver_status="feasible",  # Greedy always finds a solution
            repositioning_moves=recommendations,
            station_plans=station_plans,
            total_shortage_cost=total_shortage_cost,
            total_repositioning_cost=total_repositioning_cost,
            total_handling_cost=total_handling_cost,
            objective_value=total_shortage_cost + total_repositioning_cost + total_handling_cost,
            solve_time_seconds=solve_time,
            total_ulds_moved=sum(r.quantity for r in recommendations),
            max_moves_per_station=max_moves_per_station,
        )

    def _solve_greedy(
        self,
        shortages: dict[str, int],
        surpluses: dict[str, int],
        max_moves: int,
    ) -> list[tuple[str, str, int]]:
        """
        Greedy solution to the repositioning problem.

        Repeatedly matches largest shortage with nearest surplus.
        """
        moves = []
        remaining_shortages = dict(shortages)
        remaining_surpluses = dict(surpluses)

        for _ in range(self.max_iterations):
            if not remaining_shortages or not remaining_surpluses:
                break

            # Find largest shortage
            dest = max(remaining_shortages, key=lambda s: remaining_shortages[s])
            shortage_qty = remaining_shortages[dest]

            if shortage_qty <= 0:
                del remaining_shortages[dest]
                continue

            # Find nearest surplus with capacity
            best_origin = None
            best_distance = float("inf")

            for origin, surplus_qty in remaining_surpluses.items():
                if surplus_qty <= 0:
                    continue
                distance = self._get_distance(origin, dest)
                if distance < best_distance:
                    best_distance = distance
                    best_origin = origin

            if best_origin is None:
                break

            # Determine quantity to move
            surplus_qty = remaining_surpluses[best_origin]
            move_qty = min(shortage_qty, surplus_qty, 20)  # Cap at 20 per move

            if move_qty > 0:
                moves.append((best_origin, dest, move_qty))

                # Update remaining
                remaining_shortages[dest] -= move_qty
                remaining_surpluses[best_origin] -= move_qty

                if remaining_shortages[dest] <= 0:
                    del remaining_shortages[dest]
                if remaining_surpluses[best_origin] <= 0:
                    del remaining_surpluses[best_origin]

            if len(moves) >= max_moves * len(shortages):
                break

        return moves

    def _build_station_plan(
        self,
        station: str,
        moves: list[RepositioningRecommendation],
        forecast: NetworkForecast,
    ) -> StationActionPlan:
        """Build action plan for a station from optimization results."""
        now = datetime.now(timezone.utc)

        outbound = [m for m in moves if m.origin_station == station]
        inbound = [m for m in moves if m.destination_station == station]

        # Get forecast data
        demand = forecast.station_demand.get(station)
        supply = forecast.station_supply.get(station)

        current_inventory = {}
        forecasted_demand = {}
        forecasted_supply = {}

        if supply:
            for uld_type in ULDType:
                if uld_type in supply.supply_by_type:
                    current_inventory[uld_type] = int(supply.supply_by_type[uld_type].q50)
                    forecasted_supply[uld_type] = int(supply.supply_by_type[uld_type].q50)

        if demand:
            for uld_type in ULDType:
                if uld_type in demand.demand_by_type:
                    forecasted_demand[uld_type] = int(demand.demand_by_type[uld_type].q50)

        return StationActionPlan(
            station=station,
            generated_at=now,
            valid_until=now + timedelta(hours=4),
            current_inventory=current_inventory,
            forecasted_demand=forecasted_demand,
            forecasted_supply=forecasted_supply,
            repositioning_out=outbound,
            repositioning_in=inbound,
            total_ulds_to_send=sum(m.quantity for m in outbound),
            total_ulds_expected=sum(m.quantity for m in inbound),
            estimated_end_inventory={},
            total_cost=sum(m.cost_benefit.total_cost for m in outbound),
            total_benefit=sum(m.cost_benefit.total_benefit for m in inbound),
        )


class OptimizationConstraints:
    """Constraints for the optimization problem."""

    def __init__(
        self,
        max_total_moves: int = 50,
        max_moves_per_station: int = 10,
        min_move_quantity: int = 1,
        max_move_quantity: int = 30,
        available_transport_capacity: dict[tuple[str, str], int] | None = None,
    ):
        self.max_total_moves = max_total_moves
        self.max_moves_per_station = max_moves_per_station
        self.min_move_quantity = min_move_quantity
        self.max_move_quantity = max_move_quantity
        self.available_transport_capacity = available_transport_capacity or {}

    def validate_move(
        self,
        origin: str,
        destination: str,
        quantity: int,
    ) -> tuple[bool, str]:
        """Validate a proposed move against constraints."""
        if quantity < self.min_move_quantity:
            return False, f"Quantity {quantity} below minimum {self.min_move_quantity}"

        if quantity > self.max_move_quantity:
            return False, f"Quantity {quantity} exceeds maximum {self.max_move_quantity}"

        key = (origin, destination)
        if key in self.available_transport_capacity:
            if quantity > self.available_transport_capacity[key]:
                return False, f"Exceeds transport capacity on route {origin}-{destination}"

        return True, "OK"
