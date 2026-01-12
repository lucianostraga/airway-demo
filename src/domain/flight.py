"""
Flight domain models.

Represents flight schedules and operations relevant to ULD movement.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, computed_field

from .uld import ULDType


class FlightStatus(str, Enum):
    """Flight operational status."""

    SCHEDULED = "scheduled"
    BOARDING = "boarding"
    DEPARTED = "departed"
    IN_AIR = "in_air"
    LANDED = "landed"
    ARRIVED = "arrived"
    CANCELLED = "cancelled"
    DIVERTED = "diverted"
    DELAYED = "delayed"


class AircraftType(BaseModel):
    """Aircraft type with ULD capacity information."""

    icao_code: str  # e.g., "B738", "A321", "B77W"
    iata_code: str  # e.g., "738", "321", "77W"
    name: str
    is_widebody: bool
    lower_deck_positions: int  # Number of ULD positions in cargo hold
    uld_types_compatible: list[ULDType]
    max_cargo_weight_kg: int

    model_config = {"frozen": True}


# Common Delta aircraft types
AIRCRAFT_TYPES: dict[str, AircraftType] = {
    "B738": AircraftType(
        icao_code="B738",
        iata_code="738",
        name="Boeing 737-800",
        is_widebody=False,
        lower_deck_positions=0,  # Narrow-body, bulk cargo only
        uld_types_compatible=[],
        max_cargo_weight_kg=2000,
    ),
    "A321": AircraftType(
        icao_code="A321",
        iata_code="321",
        name="Airbus A321",
        is_widebody=False,
        lower_deck_positions=0,
        uld_types_compatible=[],
        max_cargo_weight_kg=2500,
    ),
    "B763": AircraftType(
        icao_code="B763",
        iata_code="763",
        name="Boeing 767-300",
        is_widebody=True,
        lower_deck_positions=8,
        uld_types_compatible=[ULDType.AKE, ULDType.AKH, ULDType.PMC],
        max_cargo_weight_kg=15000,
    ),
    "A333": AircraftType(
        icao_code="A333",
        iata_code="333",
        name="Airbus A330-300",
        is_widebody=True,
        lower_deck_positions=10,
        uld_types_compatible=[ULDType.AKE, ULDType.AKH, ULDType.PMC, ULDType.AKN],
        max_cargo_weight_kg=18000,
    ),
    "A359": AircraftType(
        icao_code="A359",
        iata_code="359",
        name="Airbus A350-900",
        is_widebody=True,
        lower_deck_positions=12,
        uld_types_compatible=[ULDType.AKE, ULDType.AKH, ULDType.PMC, ULDType.AKN],
        max_cargo_weight_kg=20000,
    ),
    "B77W": AircraftType(
        icao_code="B77W",
        iata_code="77W",
        name="Boeing 777-300ER",
        is_widebody=True,
        lower_deck_positions=14,
        uld_types_compatible=[ULDType.AKE, ULDType.AKH, ULDType.PMC, ULDType.AKN],
        max_cargo_weight_kg=25000,
    ),
}


class Route(BaseModel):
    """Flight route between two stations."""

    origin: str = Field(..., pattern=r"^[A-Z]{3}$")
    destination: str = Field(..., pattern=r"^[A-Z]{3}$")
    distance_km: float = 0
    typical_duration_minutes: int = 0

    model_config = {"frozen": True}

    @computed_field
    @property
    def route_key(self) -> str:
        """Unique route identifier."""
        return f"{self.origin}-{self.destination}"

    def reverse(self) -> "Route":
        """Get the reverse route."""
        return Route(
            origin=self.destination,
            destination=self.origin,
            distance_km=self.distance_km,
            typical_duration_minutes=self.typical_duration_minutes,
        )


class Flight(BaseModel):
    """
    Flight entity representing a scheduled or operated flight.

    Contains information relevant to ULD planning and tracking.
    """

    flight_number: Annotated[str, Field(description="Flight number (e.g., DL123)")]
    airline: str = Field(default="DL", pattern=r"^[A-Z]{2}$")
    origin: str = Field(..., pattern=r"^[A-Z]{3}$")
    destination: str = Field(..., pattern=r"^[A-Z]{3}$")

    # Schedule
    scheduled_departure: datetime
    scheduled_arrival: datetime
    actual_departure: datetime | None = None
    actual_arrival: datetime | None = None

    # Status
    status: FlightStatus = FlightStatus.SCHEDULED
    delay_minutes: int = 0
    cancellation_reason: str | None = None

    # Aircraft
    aircraft_type: str | None = None  # ICAO code
    tail_number: str | None = None

    # Passenger/cargo info
    booked_passengers: int = 0
    estimated_bags: int = 0

    # ULD allocation
    uld_capacity: int = 0  # Number of ULD positions
    ulds_assigned: list[str] = Field(default_factory=list)  # ULD IDs

    model_config = {"frozen": False}

    @computed_field
    @property
    def route(self) -> Route:
        """Get the route for this flight."""
        duration = int((self.scheduled_arrival - self.scheduled_departure).total_seconds() / 60)
        return Route(
            origin=self.origin,
            destination=self.destination,
            typical_duration_minutes=duration,
        )

    @computed_field
    @property
    def is_widebody(self) -> bool:
        """Check if this is a widebody flight (can carry ULDs)."""
        if self.aircraft_type and self.aircraft_type in AIRCRAFT_TYPES:
            return AIRCRAFT_TYPES[self.aircraft_type].is_widebody
        # Estimate based on duration (>3 hours likely widebody for Delta domestic)
        duration = (self.scheduled_arrival - self.scheduled_departure).total_seconds() / 3600
        return duration > 3.0

    @property
    def flight_duration(self) -> timedelta:
        """Scheduled flight duration."""
        return self.scheduled_arrival - self.scheduled_departure

    @property
    def estimated_uld_demand(self) -> int:
        """Estimate ULD demand based on passengers."""
        if not self.is_widebody:
            return 0
        # Rough estimate: 1 ULD per 40 passengers for bags + cargo
        return max(1, self.booked_passengers // 40) if self.booked_passengers > 0 else 2

    def get_aircraft_spec(self) -> AircraftType | None:
        """Get aircraft specifications."""
        return AIRCRAFT_TYPES.get(self.aircraft_type) if self.aircraft_type else None


class FlightSchedule(BaseModel):
    """
    Collection of flights for a time period.

    Used for bulk operations and analysis.
    """

    flights: list[Flight]
    period_start: datetime
    period_end: datetime
    station: str | None = None  # If filtered by station

    model_config = {"frozen": True}

    def departures(self) -> list[Flight]:
        """Get departing flights."""
        if not self.station:
            return self.flights
        return [f for f in self.flights if f.origin == self.station]

    def arrivals(self) -> list[Flight]:
        """Get arriving flights."""
        if not self.station:
            return self.flights
        return [f for f in self.flights if f.destination == self.station]

    def widebody_flights(self) -> list[Flight]:
        """Get widebody flights only."""
        return [f for f in self.flights if f.is_widebody]

    def total_uld_demand(self) -> int:
        """Total estimated ULD demand across all widebody flights."""
        return sum(f.estimated_uld_demand for f in self.widebody_flights())

    def by_route(self) -> dict[str, list[Flight]]:
        """Group flights by route."""
        routes: dict[str, list[Flight]] = {}
        for f in self.flights:
            key = f.route.route_key
            if key not in routes:
                routes[key] = []
            routes[key].append(f)
        return routes
