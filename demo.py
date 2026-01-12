#!/usr/bin/env python3
"""
ULD Forecasting System Demo.

Demonstrates the core capabilities:
1. Synthetic data generation
2. Demand/supply forecasting
3. Repositioning recommendations
4. Network optimization
"""

import asyncio
from datetime import datetime, timedelta, timezone

from src.data.synthetic import (
    FlightScheduleGenerator,
    ULDFleetGenerator,
    DemandPatternGenerator,
    ScenarioGenerator,
)
from src.services import (
    ForecastingService,
    RecommendationService,
    NetworkOptimizer,
)


def print_section(title: str):
    """Print a section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


async def main():
    print()
    print("*" * 60)
    print("*  Delta Airlines ULD Forecasting & Allocation System  *")
    print("*" * 60)

    # 1. Generate synthetic data
    print_section("1. Synthetic Data Generation")

    fg = FlightScheduleGenerator(seed=42)
    schedule = fg.generate_day_schedule(
        datetime.now(timezone.utc),
        stations=["ATL", "DTW", "MSP", "SLC", "JFK", "LAX"],
    )
    print(f"Generated {len(schedule.flights)} flights for today")
    print(f"Widebody flights: {len(schedule.widebody_flights())}")

    ug = ULDFleetGenerator(seed=42)
    print()
    print("Station Inventories:")
    for station in ["ATL", "DTW", "MSP", "SLC"]:
        inventory = ug.generate_inventory(station)
        print(
            f"  {station}: {inventory.total_count():3d} total, "
            f"{inventory.total_available():3d} available"
        )

    # 2. Demand Forecasting
    print_section("2. Demand Forecasting")

    fs = ForecastingService(seed=42)
    for station in ["ATL", "DTW", "JFK"]:
        forecasts = await fs.forecast_demand(station, hours_ahead=6)
        if forecasts:
            f = forecasts[0]
            print(
                f"{station}: Demand = {f.total_demand.q50:5.0f} ULDs "
                f"(90% CI: {f.total_demand.q05:.0f} - {f.total_demand.q95:.0f})"
            )

    # 3. Network Forecast
    print_section("3. Network Imbalance Analysis")

    network = await fs.forecast_network(
        stations=["ATL", "DTW", "MSP", "SLC", "JFK", "LAX", "SEA", "BOS"],
        hours_ahead=12,
    )

    print(
        f"Network Demand: {network.total_network_demand.q50:.0f} ULDs "
        f"(+/- {network.total_network_demand.iqr:.0f})"
    )
    print(
        f"Network Supply: {network.total_network_supply.q50:.0f} ULDs "
        f"(+/- {network.total_network_supply.iqr:.0f})"
    )
    print(f"Balance: {'Yes' if network.network_balanced else 'No'}")

    if network.shortage_stations:
        print(f"Shortage Stations: {', '.join(network.shortage_stations)}")
    if network.surplus_stations:
        print(f"Surplus Stations: {', '.join(network.surplus_stations)}")

    # 4. Recommendations
    print_section("4. Repositioning Recommendations")

    rs = RecommendationService()
    recs = await rs.generate_repositioning_recommendations(network)

    if recs:
        print(f"Generated {len(recs)} repositioning recommendations")
        print()
        for r in recs[:3]:  # Show top 3
            print(f"  [{r.priority.value.upper():8}] Move {r.quantity} ULDs")
            print(f"            From: {r.origin_station} -> To: {r.destination_station}")
            print(f"            Net Benefit: ${r.cost_benefit.net_benefit:,.0f}")
            print()
    else:
        print("No repositioning needed - network is balanced!")

    # 5. Optimization
    print_section("5. Network Optimization")

    optimizer = NetworkOptimizer()
    result = await optimizer.optimize(network, max_moves_per_station=3)

    print(f"Solver Status: {result.solver_status}")
    print(f"Solve Time: {result.solve_time_seconds:.3f} seconds")
    print(f"Total Moves: {len(result.repositioning_moves)}")
    print(f"Total ULDs to Move: {result.total_ulds_moved}")
    print(f"Total System Cost: ${result.total_system_cost:,.0f}")

    # 6. Scenario Analysis
    print_section("6. Scenario: Winter Storm")

    sg = ScenarioGenerator(seed=42)
    scenario = sg.generate_winter_storm_scenario(severity=0.8)

    print(f"Scenario: {scenario.name}")
    print(f"Duration: {(scenario.end_time - scenario.start_time).total_seconds() / 3600:.0f} hours")
    print(f"Affected Stations: {len(scenario.events)}")
    for event in scenario.events:
        print(f"  {event.station}: {event.description}")
    print(f"Impact: {scenario.total_impact_flights} flights, {scenario.total_impact_ulds} ULDs")

    print()
    print("=" * 60)
    print("  Demo Complete!")
    print("=" * 60)
    print()
    print("To run the API server:")
    print("  uvicorn src.api:app --reload")
    print()
    print("API Documentation at: http://localhost:8000/docs")
    print()


if __name__ == "__main__":
    asyncio.run(main())
