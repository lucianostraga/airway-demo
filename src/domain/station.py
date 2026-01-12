"""
Station (Airport) domain models.

Represents airports in the Delta network with their ULD handling capabilities.
"""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field

from .uld import ULDType


class StationTier(str, Enum):
    """
    Station classification in Delta's network.

    Tier affects ULD inventory policies, safety stock, and repositioning priority.
    """

    HUB = "hub"  # ATL, DTW, MSP, SLC - Highest volume, 24/7 operations
    FOCUS_CITY = "focus_city"  # BOS, LAX, JFK, SEA - High volume, priority routes
    SPOKE = "spoke"  # Regional stations - Lower volume, depends on hub supply
    INTERNATIONAL = "international"  # Non-US stations


# Delta hub and focus city definitions
DELTA_HUBS = {"ATL", "DTW", "MSP", "SLC"}
DELTA_FOCUS_CITIES = {"BOS", "JFK", "LAX", "SEA", "LGA", "SFO", "EWR"}


class StationCapacity(BaseModel):
    """ULD handling capacity at a station."""

    uld_type: ULDType
    storage_capacity: int  # Max ULDs that can be stored
    daily_throughput: int  # Average ULDs processed per day
    safety_stock: int  # Minimum inventory to maintain
    optimal_inventory: int  # Target inventory level

    model_config = {"frozen": True}


class Station(BaseModel):
    """
    Airport station in the Delta network.

    Contains operational characteristics relevant to ULD management.
    """

    code: Annotated[str, Field(pattern=r"^[A-Z]{3}$", description="IATA airport code")]
    icao: Annotated[str, Field(pattern=r"^[A-Z]{4}$", description="ICAO airport code")]
    name: str
    city: str
    country: str = "US"
    tier: StationTier
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str = "America/New_York"

    # Operational characteristics
    is_24_hour: bool = True
    has_cargo_facility: bool = True
    ground_handler: str | None = None  # Ground handling company

    # ULD capacities by type
    capacities: dict[ULDType, StationCapacity] = Field(default_factory=dict)

    # Average metrics (updated periodically)
    avg_daily_departures: int = 0
    avg_daily_arrivals: int = 0
    avg_dwell_time_hours: float = 8.0  # Average time ULD spends at station

    model_config = {"frozen": False}

    @classmethod
    def get_tier(cls, code: str) -> StationTier:
        """Determine station tier from code."""
        code = code.upper()
        if code in DELTA_HUBS:
            return StationTier.HUB
        elif code in DELTA_FOCUS_CITIES:
            return StationTier.FOCUS_CITY
        elif len(code) == 3 and code[0] != "K":  # International (rough heuristic)
            return StationTier.INTERNATIONAL
        else:
            return StationTier.SPOKE

    def get_safety_stock(self, uld_type: ULDType) -> int:
        """Get safety stock level for a ULD type."""
        if uld_type in self.capacities:
            return self.capacities[uld_type].safety_stock
        # Default safety stock based on tier
        defaults = {
            StationTier.HUB: 50,
            StationTier.FOCUS_CITY: 30,
            StationTier.SPOKE: 10,
            StationTier.INTERNATIONAL: 20,
        }
        return defaults.get(self.tier, 10)

    def distance_to(self, other: "Station") -> float:
        """Calculate great circle distance to another station in km."""
        import math

        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        return 6371 * c  # Earth radius in km


# Pre-defined Delta stations
DELTA_STATIONS: dict[str, Station] = {
    "ATL": Station(
        code="ATL",
        icao="KATL",
        name="Hartsfield-Jackson Atlanta International",
        city="Atlanta",
        country="US",
        tier=StationTier.HUB,
        latitude=33.6407,
        longitude=-84.4277,
        timezone="America/New_York",
        is_24_hour=True,
        avg_daily_departures=450,
        avg_daily_arrivals=450,
        avg_dwell_time_hours=6.0,
    ),
    "DTW": Station(
        code="DTW",
        icao="KDTW",
        name="Detroit Metropolitan Wayne County",
        city="Detroit",
        country="US",
        tier=StationTier.HUB,
        latitude=42.2124,
        longitude=-83.3534,
        timezone="America/Detroit",
        is_24_hour=True,
        avg_daily_departures=200,
        avg_daily_arrivals=200,
        avg_dwell_time_hours=8.0,
    ),
    "MSP": Station(
        code="MSP",
        icao="KMSP",
        name="Minneapolis-Saint Paul International",
        city="Minneapolis",
        country="US",
        tier=StationTier.HUB,
        latitude=44.8820,
        longitude=-93.2218,
        timezone="America/Chicago",
        is_24_hour=True,
        avg_daily_departures=180,
        avg_daily_arrivals=180,
        avg_dwell_time_hours=10.0,
    ),
    "SLC": Station(
        code="SLC",
        icao="KSLC",
        name="Salt Lake City International",
        city="Salt Lake City",
        country="US",
        tier=StationTier.HUB,
        latitude=40.7884,
        longitude=-111.9778,
        timezone="America/Denver",
        is_24_hour=True,
        avg_daily_departures=150,
        avg_daily_arrivals=150,
        avg_dwell_time_hours=8.0,
    ),
    "JFK": Station(
        code="JFK",
        icao="KJFK",
        name="John F. Kennedy International",
        city="New York",
        country="US",
        tier=StationTier.FOCUS_CITY,
        latitude=40.6413,
        longitude=-73.7781,
        timezone="America/New_York",
        is_24_hour=True,
        avg_daily_departures=120,
        avg_daily_arrivals=120,
        avg_dwell_time_hours=12.0,
    ),
    "LAX": Station(
        code="LAX",
        icao="KLAX",
        name="Los Angeles International",
        city="Los Angeles",
        country="US",
        tier=StationTier.FOCUS_CITY,
        latitude=33.9416,
        longitude=-118.4085,
        timezone="America/Los_Angeles",
        is_24_hour=True,
        avg_daily_departures=100,
        avg_daily_arrivals=100,
        avg_dwell_time_hours=14.0,
    ),
    "SEA": Station(
        code="SEA",
        icao="KSEA",
        name="Seattle-Tacoma International",
        city="Seattle",
        country="US",
        tier=StationTier.FOCUS_CITY,
        latitude=47.4502,
        longitude=-122.3088,
        timezone="America/Los_Angeles",
        is_24_hour=True,
        avg_daily_departures=80,
        avg_daily_arrivals=80,
        avg_dwell_time_hours=12.0,
    ),
    "BOS": Station(
        code="BOS",
        icao="KBOS",
        name="Boston Logan International",
        city="Boston",
        country="US",
        tier=StationTier.FOCUS_CITY,
        latitude=42.3656,
        longitude=-71.0096,
        timezone="America/New_York",
        is_24_hour=True,
        avg_daily_departures=70,
        avg_daily_arrivals=70,
        avg_dwell_time_hours=14.0,
    ),
}


def get_station(code: str) -> Station | None:
    """Get a station by IATA code."""
    return DELTA_STATIONS.get(code.upper())


def get_all_stations() -> list[Station]:
    """Get all defined stations."""
    return list(DELTA_STATIONS.values())
