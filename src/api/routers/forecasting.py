"""
Forecasting API endpoints.
"""

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel, Field

from src.domain import ForecastGranularity, ULDType
from src.services import ForecastingService
from src.api.dependencies import get_forecasting_service

router = APIRouter()


class QuantileForecastResponse(BaseModel):
    """Quantile forecast response."""

    q05: float
    q25: float
    q50: float
    q75: float
    q95: float


class DemandForecastResponse(BaseModel):
    """Demand forecast response."""

    station: str
    forecast_time: datetime
    granularity: str
    total_demand: QuantileForecastResponse
    demand_by_type: dict[str, QuantileForecastResponse]
    confidence: str
    is_anomaly: bool


class SupplyForecastResponse(BaseModel):
    """Supply forecast response."""

    station: str
    forecast_time: datetime
    total_supply: QuantileForecastResponse
    expected_arrivals: int
    expected_departures: int
    confidence: str


class ImbalanceForecastResponse(BaseModel):
    """Imbalance forecast response."""

    station: str
    forecast_time: datetime
    total_imbalance: QuantileForecastResponse
    shortage_probability: float
    severity: str


class NetworkForecastSummary(BaseModel):
    """Network forecast summary."""

    generated_at: datetime
    forecast_horizon_hours: int
    total_network_demand: QuantileForecastResponse
    total_network_supply: QuantileForecastResponse
    shortage_stations: list[str]
    surplus_stations: list[str]
    network_balanced: bool


@router.get("/demand/{station}", response_model=list[DemandForecastResponse])
async def get_demand_forecast(
    station: str,
    service: Annotated[ForecastingService, Depends(get_forecasting_service)],
    hours_ahead: int = Query(default=24, ge=1, le=168),
    granularity: str = Query(default="hourly", pattern="^(hourly|daily|weekly)$"),
):
    """
    Get demand forecast for a station.

    Predicts ULD demand for the specified time horizon with uncertainty quantiles.
    """
    station = station.upper()

    gran = ForecastGranularity(granularity)
    forecasts = await service.forecast_demand(station, hours_ahead, gran)

    return [
        DemandForecastResponse(
            station=f.station,
            forecast_time=f.forecast_time,
            granularity=f.granularity.value,
            total_demand=QuantileForecastResponse(
                q05=f.total_demand.q05,
                q25=f.total_demand.q25,
                q50=f.total_demand.q50,
                q75=f.total_demand.q75,
                q95=f.total_demand.q95,
            ),
            demand_by_type={
                uld_type.value: QuantileForecastResponse(
                    q05=q.q05, q25=q.q25, q50=q.q50, q75=q.q75, q95=q.q95
                )
                for uld_type, q in f.demand_by_type.items()
            },
            confidence=f.confidence.value,
            is_anomaly=f.is_anomaly,
        )
        for f in forecasts
    ]


@router.get("/supply/{station}", response_model=list[SupplyForecastResponse])
async def get_supply_forecast(
    station: str,
    service: Annotated[ForecastingService, Depends(get_forecasting_service)],
    hours_ahead: int = Query(default=24, ge=1, le=168),
):
    """
    Get supply forecast for a station.

    Predicts ULD availability based on current inventory and expected movements.
    """
    station = station.upper()

    # Default inventory (would get from tracking service in production)
    current_inventory = {
        ULDType.AKE: 50,
        ULDType.AKH: 15,
        ULDType.PMC: 20,
        ULDType.AKN: 10,
        ULDType.AAP: 5,
    }

    forecasts = await service.forecast_supply(
        station, current_inventory, hours_ahead
    )

    return [
        SupplyForecastResponse(
            station=f.station,
            forecast_time=f.forecast_time,
            total_supply=QuantileForecastResponse(
                q05=f.total_supply.q05,
                q25=f.total_supply.q25,
                q50=f.total_supply.q50,
                q75=f.total_supply.q75,
                q95=f.total_supply.q95,
            ),
            expected_arrivals=f.expected_arrivals,
            expected_departures=f.expected_departures,
            confidence=f.confidence.value,
        )
        for f in forecasts
    ]


@router.get("/imbalance/{station}", response_model=list[ImbalanceForecastResponse])
async def get_imbalance_forecast(
    station: str,
    service: Annotated[ForecastingService, Depends(get_forecasting_service)],
    hours_ahead: int = Query(default=24, ge=1, le=168),
):
    """
    Get supply-demand imbalance forecast for a station.

    Identifies potential shortages or surpluses with probability estimates.
    """
    station = station.upper()

    # Get demand and supply forecasts
    demand = await service.forecast_demand(station, hours_ahead)
    current_inventory = {
        ULDType.AKE: 50,
        ULDType.AKH: 15,
        ULDType.PMC: 20,
        ULDType.AKN: 10,
        ULDType.AAP: 5,
    }
    supply = await service.forecast_supply(station, current_inventory, hours_ahead)

    # Calculate imbalances
    imbalances = await service.forecast_imbalance(station, demand, supply)

    return [
        ImbalanceForecastResponse(
            station=i.station,
            forecast_time=i.forecast_time,
            total_imbalance=QuantileForecastResponse(
                q05=i.total_imbalance.q05,
                q25=i.total_imbalance.q25,
                q50=i.total_imbalance.q50,
                q75=i.total_imbalance.q75,
                q95=i.total_imbalance.q95,
            ),
            shortage_probability=i.shortage_probability,
            severity=i.severity,
        )
        for i in imbalances
    ]


@router.get("/network", response_model=NetworkForecastSummary)
async def get_network_forecast(
    service: Annotated[ForecastingService, Depends(get_forecasting_service)],
    hours_ahead: int = Query(default=24, ge=1, le=168),
):
    """
    Get network-wide forecast summary.

    Provides aggregate forecasts and identifies stations needing attention.
    """
    forecast = await service.forecast_network(hours_ahead=hours_ahead)

    return NetworkForecastSummary(
        generated_at=forecast.generated_at,
        forecast_horizon_hours=forecast.forecast_horizon_hours,
        total_network_demand=QuantileForecastResponse(
            q05=forecast.total_network_demand.q05,
            q25=forecast.total_network_demand.q25,
            q50=forecast.total_network_demand.q50,
            q75=forecast.total_network_demand.q75,
            q95=forecast.total_network_demand.q95,
        ),
        total_network_supply=QuantileForecastResponse(
            q05=forecast.total_network_supply.q05,
            q25=forecast.total_network_supply.q25,
            q50=forecast.total_network_supply.q50,
            q75=forecast.total_network_supply.q75,
            q95=forecast.total_network_supply.q95,
        ),
        shortage_stations=forecast.shortage_stations,
        surplus_stations=forecast.surplus_stations,
        network_balanced=forecast.network_balanced,
    )


@router.get("/network/priority")
async def get_priority_stations(
    service: Annotated[ForecastingService, Depends(get_forecasting_service)],
    hours_ahead: int = Query(default=24, ge=1, le=168),
    top_n: int = Query(default=5, ge=1, le=20),
):
    """
    Get stations that need priority attention.

    Ranks stations by shortage probability and returns the top N.
    """
    forecast = await service.forecast_network(hours_ahead=hours_ahead)
    priority_stations = forecast.get_priority_stations(top_n)

    station_details = []
    for station in priority_stations:
        imbalance = forecast.station_imbalance.get(station)
        if imbalance:
            station_details.append({
                "station": station,
                "shortage_probability": imbalance.shortage_probability,
                "severity": imbalance.severity,
                "estimated_shortage": abs(imbalance.total_imbalance.q50)
                if imbalance.total_imbalance.q50 < 0 else 0,
            })

    return {
        "generated_at": forecast.generated_at.isoformat(),
        "priority_stations": station_details,
    }
