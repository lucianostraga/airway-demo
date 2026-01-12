"""
Stations API endpoints.
"""

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from src.domain import DELTA_STATIONS, StationTier

router = APIRouter()


class StationInfo(BaseModel):
    """Station information response."""

    code: str
    name: str
    city: str
    country: str
    hub_tier: int
    timezone: str
    coordinates: dict[str, float]
    capacity: dict[str, int]


class StationList(BaseModel):
    """List of stations."""

    stations: list[StationInfo]
    total: int


@router.get("/", response_model=StationList)
async def list_stations(
    tier: str | None = Query(default=None, pattern="^(hub|focus_city|spoke|international)$"),
):
    """
    List all stations in the Delta network.

    Optionally filter by tier (hub, focus_city, spoke, international).
    """
    stations = []

    for code, info in DELTA_STATIONS.items():
        if tier and info.tier.value != tier:
            continue

        # Map tier to numeric hub_tier
        hub_tier_map = {
            StationTier.HUB: 1,
            StationTier.FOCUS_CITY: 2,
            StationTier.SPOKE: 3,
            StationTier.INTERNATIONAL: 4,
        }

        stations.append(
            StationInfo(
                code=code,
                name=info.name,
                city=info.city,
                country=info.country,
                hub_tier=hub_tier_map.get(info.tier, 3),
                timezone=info.timezone,
                coordinates={"latitude": info.latitude, "longitude": info.longitude},
                capacity={uld_type.value: capacity.storage_capacity for uld_type, capacity in info.capacities.items()} or {"AKE": 100, "PMC": 50, "AKH": 30, "LD3": 20, "LD7": 10} or {"AKE": 100, "PMC": 50, "AKH": 30, "LD3": 20, "LD7": 10},
            )
        )

    return StationList(
        stations=sorted(stations, key=lambda s: s.code),
        total=len(stations),
    )


@router.get("/hubs", response_model=StationList)
async def list_hubs():
    """
    List Delta hub stations.

    Returns all stations classified as hubs (ATL, DTW, MSP, SLC).
    """
    stations = [
        StationInfo(
            code=code,
            name=info.name,
            city=info.city,
            country=info.country,
            hub_tier=1,
            timezone=info.timezone,
            coordinates={"latitude": info.latitude, "longitude": info.longitude},
            capacity={uld_type.value: capacity.storage_capacity for uld_type, capacity in info.capacities.items()} or {"AKE": 100, "PMC": 50, "AKH": 30, "LD3": 20, "LD7": 10},
        )
        for code, info in DELTA_STATIONS.items()
        if info.tier == StationTier.HUB
    ]

    return StationList(
        stations=sorted(stations, key=lambda s: s.code),
        total=len(stations),
    )


@router.get("/{station}", response_model=StationInfo)
async def get_station(station: str):
    """
    Get information about a specific station.

    Returns station details including tier and region.
    """
    station = station.upper()
    info = DELTA_STATIONS.get(station)

    if not info:
        raise HTTPException(
            status_code=404,
            detail=f"Station {station} not found",
        )

    # Map tier to numeric hub_tier
    hub_tier_map = {
        StationTier.HUB: 1,
        StationTier.FOCUS_CITY: 2,
        StationTier.SPOKE: 3,
        StationTier.INTERNATIONAL: 4,
    }

    return StationInfo(
        code=station,
        name=info.name,
        city=info.city,
        country=info.country,
        hub_tier=hub_tier_map.get(info.tier, 3),
        timezone=info.timezone,
        coordinates={"latitude": info.latitude, "longitude": info.longitude},
        capacity={uld_type.value: capacity.storage_capacity for uld_type, capacity in info.capacities.items()} or {"AKE": 100, "PMC": 50, "AKH": 30, "LD3": 20, "LD7": 10},
    )


@router.get("/{station}/connections")
async def get_station_connections(station: str):
    """
    Get connections for a station.

    Returns list of stations with direct flights.
    Note: This is a simplified implementation using hub connectivity patterns.
    """
    station = station.upper()
    info = DELTA_STATIONS.get(station)

    if not info:
        raise HTTPException(
            status_code=404,
            detail=f"Station {station} not found",
        )

    # Simplified connectivity: hubs connect to all, others connect to hubs
    connections = []

    if info.tier == StationTier.HUB:
        # Hubs connect to everything
        connections = [
            {"code": code, "name": s.name, "tier": s.tier.value}
            for code, s in DELTA_STATIONS.items()
            if code != station
        ]
    else:
        # Non-hubs connect to hubs
        connections = [
            {"code": code, "name": s.name, "tier": s.tier.value}
            for code, s in DELTA_STATIONS.items()
            if s.tier == StationTier.HUB
        ]

    return {
        "station": station,
        "connections": sorted(connections, key=lambda x: x["code"]),
        "total_connections": len(connections),
    }
