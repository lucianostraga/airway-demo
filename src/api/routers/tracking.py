"""
Tracking API endpoints.
"""

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel, Field

from src.domain import ULDType, ULDPosition, ULDInventory
from src.services import ULDTrackingService
from src.api.dependencies import get_tracking_service

router = APIRouter()


class PositionResponse(BaseModel):
    """ULD position response."""

    uld_id: str
    uld_type: str
    station: str
    timestamp: datetime
    source: str
    flight_number: str | None = None
    confidence: float


class InventorySummary(BaseModel):
    """Inventory summary for a station."""

    station: str
    timestamp: datetime
    total_count: int
    available_count: int
    availability_ratio: float
    by_type: dict[str, int]


class GeolocationRequest(BaseModel):
    """Request to record geolocation."""

    uld_id: str
    uld_type: str
    station: str = Field(..., pattern=r"^[A-Z]{3}$")
    confidence: float = Field(default=0.95, ge=0, le=1)


@router.get("/position/{uld_id}", response_model=PositionResponse | None)
async def get_uld_position(
    uld_id: str,
    service: Annotated[ULDTrackingService, Depends(get_tracking_service)],
):
    """
    Get current position of a ULD.

    Returns the most recent known position from geolocation or flight events.
    """
    position = await service.get_current_position(uld_id)
    if not position:
        raise HTTPException(status_code=404, detail=f"ULD {uld_id} not found")

    return PositionResponse(
        uld_id=position.uld_id,
        uld_type=position.uld_type.value,
        station=position.station,
        timestamp=position.timestamp,
        source=position.position_source,
        flight_number=position.flight_number,
        confidence=position.confidence,
    )


@router.get("/position/{uld_id}/history")
async def get_uld_history(
    uld_id: str,
    service: Annotated[ULDTrackingService, Depends(get_tracking_service)],
    days: int = Query(default=7, ge=1, le=90),
):
    """
    Get movement history for a ULD.

    Returns chronological list of positions over the specified number of days.
    """
    history = await service.get_movement_history(uld_id, days=days)

    return {
        "uld_id": uld_id,
        "days": days,
        "positions": [
            {
                "station": p.station,
                "timestamp": p.timestamp.isoformat(),
                "source": p.position_source,
                "flight_number": p.flight_number,
            }
            for p in history
        ],
    }


@router.post("/position/geolocation", response_model=PositionResponse)
async def record_geolocation(
    request: GeolocationRequest,
    service: Annotated[ULDTrackingService, Depends(get_tracking_service)],
):
    """
    Record a geolocation ping for a ULD.

    Used to update ULD positions from BLE tag readings.
    """
    try:
        uld_type = ULDType(request.uld_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ULD type: {request.uld_type}",
        )

    position = await service.record_geolocation(
        uld_id=request.uld_id,
        uld_type=uld_type,
        station=request.station,
        confidence=request.confidence,
    )

    return PositionResponse(
        uld_id=position.uld_id,
        uld_type=position.uld_type.value,
        station=position.station,
        timestamp=position.timestamp,
        source=position.position_source,
        flight_number=position.flight_number,
        confidence=position.confidence,
    )


@router.get("/inventory/{station}", response_model=InventorySummary)
async def get_station_inventory(
    station: str,
    service: Annotated[ULDTrackingService, Depends(get_tracking_service)],
):
    """
    Get current ULD inventory at a station.

    Returns inventory breakdown by type and availability status.
    """
    station = station.upper()
    inventory = await service.get_station_inventory(station)

    return InventorySummary(
        station=inventory.station,
        timestamp=inventory.timestamp,
        total_count=inventory.total_count(),
        available_count=inventory.total_available(),
        availability_ratio=inventory.availability_ratio(),
        by_type={k.value: v for k, v in inventory.inventory.items()},
    )


@router.get("/network/summary")
async def get_network_summary(
    service: Annotated[ULDTrackingService, Depends(get_tracking_service)],
):
    """
    Get network-wide ULD distribution summary.

    Returns inventory summary for all stations in the network.
    """
    summary = await service.get_network_summary()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stations": summary,
    }
