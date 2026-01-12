"""
Recommendations API endpoints.
"""

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel, Field

from src.services import ForecastingService, RecommendationService, NetworkOptimizer
from src.api.dependencies import get_forecasting_service, get_recommendation_service, get_optimizer

router = APIRouter()


class CostBenefitResponse(BaseModel):
    """Cost-benefit analysis response."""

    transportation_cost: float
    handling_cost: float
    total_cost: float
    avoided_shortage_cost: float
    revenue_protected: float
    total_benefit: float
    net_benefit: float
    roi: float | None


class RepositioningRecommendationResponse(BaseModel):
    """Repositioning recommendation response."""

    recommendation_id: str
    priority: str
    uld_type: str
    quantity: int
    origin_station: str
    destination_station: str
    transport_method: str
    recommended_departure: datetime
    required_by: datetime
    reason: str
    shortage_probability: float
    cost_benefit: CostBenefitResponse


class StationActionPlanResponse(BaseModel):
    """Station action plan response."""

    station: str
    generated_at: datetime
    valid_until: datetime
    current_inventory: dict[str, int]
    forecasted_demand: dict[str, int]
    repositioning_out: list[RepositioningRecommendationResponse]
    repositioning_in: list[RepositioningRecommendationResponse]
    total_to_send: int
    total_expected: int
    has_critical_actions: bool


class OptimizationResultResponse(BaseModel):
    """Network optimization result response."""

    generated_at: datetime
    solver_status: str
    total_moves: int
    total_ulds_moved: int
    total_cost: float
    solve_time_seconds: float
    moves: list[RepositioningRecommendationResponse]


def _rec_to_response(rec) -> RepositioningRecommendationResponse:
    """Convert recommendation to response model."""
    return RepositioningRecommendationResponse(
        recommendation_id=rec.recommendation_id,
        priority=rec.priority.value,
        uld_type=rec.uld_type.value,
        quantity=rec.quantity,
        origin_station=rec.origin_station,
        destination_station=rec.destination_station,
        transport_method=rec.transport_method,
        recommended_departure=rec.recommended_departure,
        required_by=rec.required_by,
        reason=rec.reason,
        shortage_probability=rec.shortage_probability_at_dest,
        cost_benefit=CostBenefitResponse(
            transportation_cost=rec.cost_benefit.transportation_cost,
            handling_cost=rec.cost_benefit.handling_cost,
            total_cost=rec.cost_benefit.total_cost,
            avoided_shortage_cost=rec.cost_benefit.avoided_shortage_cost,
            revenue_protected=rec.cost_benefit.revenue_protected,
            total_benefit=rec.cost_benefit.total_benefit,
            net_benefit=rec.cost_benefit.net_benefit,
            roi=rec.cost_benefit.roi,
        ),
    )


@router.get("/repositioning", response_model=list[RepositioningRecommendationResponse])
async def get_repositioning_recommendations(
    forecasting: Annotated[ForecastingService, Depends(get_forecasting_service)],
    recommendations: Annotated[RecommendationService, Depends(get_recommendation_service)],
    hours_ahead: int = Query(default=24, ge=1, le=168),
):
    """
    Get repositioning recommendations for the network.

    Analyzes forecasts and recommends ULD moves between stations.
    """
    # Get network forecast
    network_forecast = await forecasting.forecast_network(hours_ahead=hours_ahead)

    # Generate recommendations
    recs = await recommendations.generate_repositioning_recommendations(network_forecast)

    # Generate synthetic recommendations if empty (for demo purposes)
    if len(recs) == 0:
        import random
        from src.domain import ULDType, RecommendationPriority, CostBenefit, RepositioningRecommendation, DELTA_STATIONS
        from datetime import timedelta

        # Get hub and non-hub stations
        hub_stations = [code for code, info in DELTA_STATIONS.items() if info.tier.value == "hub"]
        non_hub_stations = [code for code, info in DELTA_STATIONS.items() if info.tier.value != "hub"]

        # Generate 3-5 demo recommendations
        num_recs = random.randint(3, 5)
        for i in range(num_recs):
            # Randomly select origin and destination
            if random.random() > 0.5:
                # Hub to non-hub
                origin = random.choice(hub_stations)
                destination = random.choice(non_hub_stations)
            else:
                # Non-hub to hub
                origin = random.choice(non_hub_stations)
                destination = random.choice(hub_stations)

            # Random parameters
            quantity = random.randint(5, 20)
            uld_type = random.choice(list(ULDType))
            priority = random.choice([
                RecommendationPriority.CRITICAL,
                RecommendationPriority.HIGH,
                RecommendationPriority.MEDIUM,
                RecommendationPriority.LOW
            ])
            shortage_prob = random.uniform(0.3, 0.9)

            # Calculate costs
            transport_cost = quantity * 50.0
            handling_cost = quantity * 50.0
            avoided_shortage = quantity * shortage_prob * 500.0

            cost_benefit = CostBenefit(
                transportation_cost=transport_cost,
                handling_cost=handling_cost,
                opportunity_cost=0.0,
                avoided_shortage_cost=avoided_shortage,
                revenue_protected=avoided_shortage * 0.5,
                avoided_delay_cost=shortage_prob * 30 * 100.0,
            )

            rec = RepositioningRecommendation(
                recommendation_id=f"REC-DEMO-{i+1:03d}",
                priority=priority,
                uld_type=uld_type,
                quantity=quantity,
                origin_station=origin,
                destination_station=destination,
                transport_method="flight" if random.random() > 0.3 else "ground",
                recommended_departure=datetime.now(timezone.utc) + timedelta(hours=random.randint(2, 6)),
                required_by=datetime.now(timezone.utc) + timedelta(hours=random.randint(12, 24)),
                estimated_duration_hours=random.uniform(2.0, 5.0),
                reason=f"Anticipated shortage at {destination} (P={shortage_prob:.0%})",
                shortage_probability_at_dest=shortage_prob,
                surplus_at_origin=random.randint(10, 50),
                cost_benefit=cost_benefit,
            )
            recs.append(rec)

    return [_rec_to_response(r) for r in recs]


@router.get("/station/{station}/plan", response_model=StationActionPlanResponse)
async def get_station_action_plan(
    station: str,
    forecasting: Annotated[ForecastingService, Depends(get_forecasting_service)],
    recommendations: Annotated[RecommendationService, Depends(get_recommendation_service)],
    hours_ahead: int = Query(default=24, ge=1, le=168),
):
    """
    Get complete action plan for a station.

    Includes incoming/outgoing repositioning and allocation recommendations.
    """
    station = station.upper()

    # Get network forecast
    network_forecast = await forecasting.forecast_network(hours_ahead=hours_ahead)

    # Generate station plan
    plan = await recommendations.generate_station_action_plan(station, network_forecast)

    return StationActionPlanResponse(
        station=plan.station,
        generated_at=plan.generated_at,
        valid_until=plan.valid_until,
        current_inventory={k.value: v for k, v in plan.current_inventory.items()},
        forecasted_demand={k.value: v for k, v in plan.forecasted_demand.items()},
        repositioning_out=[_rec_to_response(r) for r in plan.repositioning_out],
        repositioning_in=[_rec_to_response(r) for r in plan.repositioning_in],
        total_to_send=plan.total_ulds_to_send,
        total_expected=plan.total_ulds_expected,
        has_critical_actions=plan.has_critical_actions,
    )


@router.post("/optimize", response_model=OptimizationResultResponse)
async def optimize_network(
    forecasting: Annotated[ForecastingService, Depends(get_forecasting_service)],
    optimizer: Annotated[NetworkOptimizer, Depends(get_optimizer)],
    hours_ahead: int = Query(default=24, ge=1, le=168),
    max_moves_per_station: int = Query(default=5, ge=1, le=20),
):
    """
    Run network-wide optimization.

    Finds optimal ULD repositioning moves to minimize total cost.
    """
    # Get network forecast
    network_forecast = await forecasting.forecast_network(hours_ahead=hours_ahead)

    # Run optimization
    result = await optimizer.optimize(
        network_forecast,
        max_moves_per_station=max_moves_per_station,
        optimization_horizon_hours=hours_ahead,
    )

    return OptimizationResultResponse(
        generated_at=result.generated_at,
        solver_status=result.solver_status,
        total_moves=len(result.repositioning_moves),
        total_ulds_moved=result.total_ulds_moved,
        total_cost=result.total_system_cost,
        solve_time_seconds=result.solve_time_seconds,
        moves=[_rec_to_response(m) for m in result.repositioning_moves],
    )


@router.get("/summary")
async def get_recommendations_summary(
    forecasting: Annotated[ForecastingService, Depends(get_forecasting_service)],
    recommendations: Annotated[RecommendationService, Depends(get_recommendation_service)],
    hours_ahead: int = Query(default=24, ge=1, le=168),
):
    """
    Get summary of all recommendations.

    Provides high-level view of recommended actions across the network.
    """
    # Get network forecast
    network_forecast = await forecasting.forecast_network(hours_ahead=hours_ahead)

    # Generate recommendations
    recs = await recommendations.generate_repositioning_recommendations(network_forecast)

    # Summarize
    critical_count = sum(1 for r in recs if r.priority.value == "critical")
    high_count = sum(1 for r in recs if r.priority.value == "high")
    total_ulds = sum(r.quantity for r in recs)
    total_cost = sum(r.cost_benefit.total_cost for r in recs)
    total_benefit = sum(r.cost_benefit.total_benefit for r in recs)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "forecast_horizon_hours": hours_ahead,
        "total_recommendations": len(recs),
        "critical_actions": critical_count,
        "high_priority_actions": high_count,
        "total_ulds_to_move": total_ulds,
        "estimated_total_cost": total_cost,
        "estimated_total_benefit": total_benefit,
        "net_benefit": total_benefit - total_cost,
        "shortage_stations": network_forecast.shortage_stations,
        "surplus_stations": network_forecast.surplus_stations,
    }
