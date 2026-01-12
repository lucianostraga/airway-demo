"""
Base classes for data clients.

Provides abstract interfaces for flight, weather, and events data.
Implementations can be swapped via configuration without changing business logic.
"""

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


# =============================================================================
# Data Models
# =============================================================================


class FlightInfo(BaseModel):
    """Standardized flight information across all providers."""

    flight_number: str
    airline_iata: str
    departure_airport: str = Field(..., pattern=r"^[A-Z]{3}$")
    arrival_airport: str = Field(..., pattern=r"^[A-Z]{3}$")
    scheduled_departure: datetime
    scheduled_arrival: datetime
    actual_departure: datetime | None = None
    actual_arrival: datetime | None = None
    status: str  # scheduled, active, landed, cancelled, diverted
    aircraft_type: str | None = None
    delay_minutes: int = 0

    model_config = {"frozen": True}


class WeatherObservation(BaseModel):
    """Standardized weather observation."""

    station: str = Field(..., pattern=r"^[A-Z]{4}$")  # ICAO code
    observed_at: datetime
    temperature_c: float | None = None
    wind_speed_kts: int | None = None
    wind_direction: int | None = None
    visibility_miles: float | None = None
    ceiling_ft: int | None = None
    flight_category: str | None = None  # VFR, MVFR, IFR, LIFR
    raw_metar: str | None = None

    model_config = {"frozen": True}


class WeatherForecast(BaseModel):
    """Standardized weather forecast."""

    station: str
    valid_from: datetime
    valid_to: datetime
    temperature_c: float | None = None
    precipitation_probability: float | None = None
    wind_speed_kts: int | None = None
    conditions: str | None = None  # clear, rain, snow, thunderstorm, etc.
    flight_category: str | None = None
    raw_taf: str | None = None

    model_config = {"frozen": True}


class Event(BaseModel):
    """Standardized event information."""

    event_id: str
    title: str
    category: str  # sports, concerts, conferences, holidays, etc.
    start_time: datetime
    end_time: datetime | None = None
    latitude: float
    longitude: float
    venue: str | None = None
    predicted_attendance: int | None = None
    impact_score: float | None = None  # 0-100 scale

    model_config = {"frozen": True}


# =============================================================================
# Abstract Base Clients
# =============================================================================


@runtime_checkable
class DataClient(Protocol):
    """Protocol for all data clients."""

    async def health_check(self) -> bool:
        """Check if the API is available."""
        ...


class FlightDataClient(ABC):
    """Abstract base class for flight data providers."""

    @abstractmethod
    async def get_flights_by_route(
        self,
        departure: str,
        arrival: str,
        date: date,
    ) -> list[FlightInfo]:
        """Get flights for a specific route on a date."""
        pass

    @abstractmethod
    async def get_flights_by_airport(
        self,
        airport: str,
        date: date,
        direction: str = "both",  # departure, arrival, both
    ) -> list[FlightInfo]:
        """Get all flights for an airport on a date."""
        pass

    @abstractmethod
    async def get_flight_status(
        self,
        flight_number: str,
        date: date,
    ) -> FlightInfo | None:
        """Get current status of a specific flight."""
        pass

    async def health_check(self) -> bool:
        """Check if the API is available."""
        return True


class WeatherDataClient(ABC):
    """Abstract base class for weather data providers."""

    @abstractmethod
    async def get_current_weather(
        self,
        station: str,  # ICAO code (e.g., KATL)
    ) -> WeatherObservation | None:
        """Get current weather observation for a station."""
        pass

    @abstractmethod
    async def get_forecast(
        self,
        station: str,
        hours_ahead: int = 24,
    ) -> list[WeatherForecast]:
        """Get weather forecast for a station."""
        pass

    @abstractmethod
    async def get_aviation_weather(
        self,
        station: str,
    ) -> tuple[WeatherObservation | None, list[WeatherForecast]]:
        """Get METAR (current) and TAF (forecast) for aviation."""
        pass

    async def health_check(self) -> bool:
        """Check if the API is available."""
        return True


class EventsDataClient(ABC):
    """Abstract base class for events data providers."""

    @abstractmethod
    async def get_events_near_location(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
        start_date: date,
        end_date: date,
        categories: list[str] | None = None,
    ) -> list[Event]:
        """Get events near a location within a date range."""
        pass

    @abstractmethod
    async def get_events_by_city(
        self,
        city: str,
        country: str,
        start_date: date,
        end_date: date,
    ) -> list[Event]:
        """Get events in a city within a date range."""
        pass

    async def health_check(self) -> bool:
        """Check if the API is available."""
        return True


# =============================================================================
# Client Factory
# =============================================================================


class DataClientFactory:
    """
    Factory for creating data clients based on configuration.

    Allows swapping between free and paid APIs without changing business logic.

    Example:
        factory = DataClientFactory()

        # Use free APIs
        flight_client = factory.get_flight_client("aviationstack")

        # Switch to paid API (same interface)
        flight_client = factory.get_flight_client("flightaware")
    """

    _flight_clients: dict[str, type[FlightDataClient]] = {}
    _weather_clients: dict[str, type[WeatherDataClient]] = {}
    _events_clients: dict[str, type[EventsDataClient]] = {}

    @classmethod
    def register_flight_client(cls, name: str, client_class: type[FlightDataClient]) -> None:
        """Register a flight data client implementation."""
        cls._flight_clients[name.lower()] = client_class

    @classmethod
    def register_weather_client(cls, name: str, client_class: type[WeatherDataClient]) -> None:
        """Register a weather data client implementation."""
        cls._weather_clients[name.lower()] = client_class

    @classmethod
    def register_events_client(cls, name: str, client_class: type[EventsDataClient]) -> None:
        """Register an events data client implementation."""
        cls._events_clients[name.lower()] = client_class

    @classmethod
    def get_flight_client(cls, name: str, **kwargs) -> FlightDataClient:
        """Get a flight data client by name."""
        if name.lower() not in cls._flight_clients:
            available = list(cls._flight_clients.keys())
            raise ValueError(f"Unknown flight client: {name}. Available: {available}")
        return cls._flight_clients[name.lower()](**kwargs)

    @classmethod
    def get_weather_client(cls, name: str, **kwargs) -> WeatherDataClient:
        """Get a weather data client by name."""
        if name.lower() not in cls._weather_clients:
            available = list(cls._weather_clients.keys())
            raise ValueError(f"Unknown weather client: {name}. Available: {available}")
        return cls._weather_clients[name.lower()](**kwargs)

    @classmethod
    def get_events_client(cls, name: str, **kwargs) -> EventsDataClient:
        """Get an events data client by name."""
        if name.lower() not in cls._events_clients:
            available = list(cls._events_clients.keys())
            raise ValueError(f"Unknown events client: {name}. Available: {available}")
        return cls._events_clients[name.lower()](**kwargs)
