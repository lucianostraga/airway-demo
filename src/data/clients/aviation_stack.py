"""
AviationStack API Client.

Free tier: 100 requests/month
Provides flight schedules, status, and airport information.

API Documentation: https://aviationstack.com/documentation

Note: Free tier is HTTPS-disabled and has limited features.
For production, upgrade to paid plan or switch to FlightAware.
"""

import os
from datetime import date, datetime, timezone
from typing import Any

import httpx

from .base import DataClientFactory, FlightDataClient, FlightInfo


class AviationStackClient(FlightDataClient):
    """
    AviationStack API client for flight data.

    Free tier limitations:
    - 100 requests/month
    - HTTP only (no HTTPS)
    - No historical data
    - Basic flight information only

    Example:
        client = AviationStackClient(api_key="your_key")
        flights = await client.get_flights_by_airport("ATL", date.today())
    """

    # Free tier uses HTTP only
    BASE_URL = "http://api.aviationstack.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("AVIATIONSTACK_API_KEY", "")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._request_count = 0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def health_check(self) -> bool:
        """Check if API is available and key is valid."""
        if not self.api_key:
            return False
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.BASE_URL}/flights",
                params={
                    "access_key": self.api_key,
                    "limit": 1,
                },
            )
            data = response.json()
            return "error" not in data
        except Exception:
            return False

    @property
    def requests_remaining(self) -> int:
        """Estimate of remaining requests (free tier: 100/month)."""
        return max(0, 100 - self._request_count)

    async def get_flights_by_route(
        self,
        departure: str,
        arrival: str,
        flight_date: date,
    ) -> list[FlightInfo]:
        """
        Get flights for a specific route.

        Args:
            departure: IATA departure airport code (e.g., "ATL")
            arrival: IATA arrival airport code (e.g., "JFK")
            flight_date: Date of flights

        Returns:
            List of FlightInfo objects
        """
        client = await self._get_client()
        self._request_count += 1

        try:
            response = await client.get(
                f"{self.BASE_URL}/flights",
                params={
                    "access_key": self.api_key,
                    "dep_iata": departure.upper(),
                    "arr_iata": arrival.upper(),
                    "flight_date": flight_date.isoformat(),
                    "limit": 100,
                },
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                return []

            return [
                self._parse_flight(f)
                for f in data.get("data", [])
                if self._parse_flight(f) is not None
            ]

        except httpx.HTTPError:
            return []

    async def get_flights_by_airport(
        self,
        airport: str,
        flight_date: date,
        direction: str = "both",
    ) -> list[FlightInfo]:
        """
        Get all flights for an airport.

        Args:
            airport: IATA airport code
            flight_date: Date of flights
            direction: "departure", "arrival", or "both"

        Returns:
            List of FlightInfo objects
        """
        flights = []

        if direction in ("departure", "both"):
            dep_flights = await self._get_departures(airport, flight_date)
            flights.extend(dep_flights)

        if direction in ("arrival", "both"):
            arr_flights = await self._get_arrivals(airport, flight_date)
            flights.extend(arr_flights)

        # Deduplicate by flight number + date
        seen = set()
        unique_flights = []
        for f in flights:
            key = (f.flight_number, f.scheduled_departure.date())
            if key not in seen:
                seen.add(key)
                unique_flights.append(f)

        return unique_flights

    async def _get_departures(
        self,
        airport: str,
        flight_date: date,
    ) -> list[FlightInfo]:
        """Get departing flights from an airport."""
        client = await self._get_client()
        self._request_count += 1

        try:
            response = await client.get(
                f"{self.BASE_URL}/flights",
                params={
                    "access_key": self.api_key,
                    "dep_iata": airport.upper(),
                    "flight_date": flight_date.isoformat(),
                    "limit": 100,
                },
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                return []

            return [
                f
                for f in (self._parse_flight(fl) for fl in data.get("data", []))
                if f is not None
            ]

        except httpx.HTTPError:
            return []

    async def _get_arrivals(
        self,
        airport: str,
        flight_date: date,
    ) -> list[FlightInfo]:
        """Get arriving flights to an airport."""
        client = await self._get_client()
        self._request_count += 1

        try:
            response = await client.get(
                f"{self.BASE_URL}/flights",
                params={
                    "access_key": self.api_key,
                    "arr_iata": airport.upper(),
                    "flight_date": flight_date.isoformat(),
                    "limit": 100,
                },
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                return []

            return [
                f
                for f in (self._parse_flight(fl) for fl in data.get("data", []))
                if f is not None
            ]

        except httpx.HTTPError:
            return []

    async def get_flight_status(
        self,
        flight_number: str,
        flight_date: date,
    ) -> FlightInfo | None:
        """
        Get status of a specific flight.

        Args:
            flight_number: Flight number (e.g., "DL123")
            flight_date: Date of flight

        Returns:
            FlightInfo or None
        """
        client = await self._get_client()
        self._request_count += 1

        # Parse airline and number
        airline = flight_number[:2].upper()
        number = flight_number[2:]

        try:
            response = await client.get(
                f"{self.BASE_URL}/flights",
                params={
                    "access_key": self.api_key,
                    "airline_iata": airline,
                    "flight_iata": flight_number.upper(),
                    "flight_date": flight_date.isoformat(),
                },
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data or not data.get("data"):
                return None

            return self._parse_flight(data["data"][0])

        except httpx.HTTPError:
            return None

    def _parse_flight(self, data: dict[str, Any]) -> FlightInfo | None:
        """Parse AviationStack flight data into FlightInfo."""
        try:
            flight = data.get("flight", {})
            departure = data.get("departure", {})
            arrival = data.get("arrival", {})
            airline = data.get("airline", {})

            # Parse times
            def parse_time(time_str: str | None) -> datetime | None:
                if not time_str:
                    return None
                try:
                    return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                except ValueError:
                    return None

            scheduled_dep = parse_time(departure.get("scheduled"))
            scheduled_arr = parse_time(arrival.get("scheduled"))

            if not scheduled_dep or not scheduled_arr:
                return None

            actual_dep = parse_time(departure.get("actual"))
            actual_arr = parse_time(arrival.get("actual"))

            # Calculate delay
            delay = 0
            if actual_dep and scheduled_dep:
                delay = int((actual_dep - scheduled_dep).total_seconds() / 60)

            # Map status
            status_map = {
                "scheduled": "scheduled",
                "active": "active",
                "landed": "landed",
                "cancelled": "cancelled",
                "diverted": "diverted",
            }
            raw_status = data.get("flight_status", "scheduled")
            status = status_map.get(raw_status, "scheduled")

            return FlightInfo(
                flight_number=flight.get("iata", ""),
                airline_iata=airline.get("iata", ""),
                departure_airport=departure.get("iata", ""),
                arrival_airport=arrival.get("iata", ""),
                scheduled_departure=scheduled_dep,
                scheduled_arrival=scheduled_arr,
                actual_departure=actual_dep,
                actual_arrival=actual_arr,
                status=status,
                aircraft_type=data.get("aircraft", {}).get("iata"),
                delay_minutes=max(0, delay),
            )

        except Exception:
            return None


# Register with factory
DataClientFactory.register_flight_client("aviationstack", AviationStackClient)


class MockFlightClient(FlightDataClient):
    """
    Mock flight client for development/testing without API calls.

    Generates realistic-looking flight data based on known Delta routes.
    """

    # Common Delta routes from ATL
    DELTA_ROUTES = [
        ("ATL", "JFK", 180),
        ("ATL", "LAX", 300),
        ("ATL", "ORD", 150),
        ("ATL", "DFW", 150),
        ("ATL", "DTW", 120),
        ("ATL", "MSP", 180),
        ("ATL", "SLC", 270),
        ("ATL", "SEA", 330),
        ("ATL", "BOS", 180),
        ("ATL", "MIA", 120),
        ("JFK", "LAX", 360),
        ("DTW", "MSP", 120),
        ("MSP", "SLC", 180),
    ]

    async def get_flights_by_route(
        self,
        departure: str,
        arrival: str,
        flight_date: date,
    ) -> list[FlightInfo]:
        """Generate mock flights for a route."""
        import random

        flights = []
        base_times = [6, 8, 10, 12, 14, 16, 18, 20]

        for hour in base_times:
            flight_num = f"DL{random.randint(100, 999)}"
            scheduled_dep = datetime(
                flight_date.year,
                flight_date.month,
                flight_date.day,
                hour,
                random.randint(0, 59),
                tzinfo=timezone.utc,
            )

            # Find duration or default
            duration = 180
            for dep, arr, dur in self.DELTA_ROUTES:
                if dep == departure.upper() and arr == arrival.upper():
                    duration = dur
                    break

            scheduled_arr = scheduled_dep.replace(
                hour=(scheduled_dep.hour + duration // 60) % 24,
                minute=(scheduled_dep.minute + duration % 60) % 60,
            )

            # Random delay (most on-time, some delayed)
            delay = 0 if random.random() < 0.7 else random.randint(5, 120)

            flights.append(
                FlightInfo(
                    flight_number=flight_num,
                    airline_iata="DL",
                    departure_airport=departure.upper(),
                    arrival_airport=arrival.upper(),
                    scheduled_departure=scheduled_dep,
                    scheduled_arrival=scheduled_arr,
                    actual_departure=None,
                    actual_arrival=None,
                    status="scheduled",
                    aircraft_type=random.choice(["B738", "A321", "B763", "A333"]),
                    delay_minutes=delay,
                )
            )

        return flights

    async def get_flights_by_airport(
        self,
        airport: str,
        flight_date: date,
        direction: str = "both",
    ) -> list[FlightInfo]:
        """Generate mock flights for an airport."""
        flights = []

        # Find routes involving this airport
        for dep, arr, dur in self.DELTA_ROUTES:
            if direction in ("departure", "both") and dep == airport.upper():
                route_flights = await self.get_flights_by_route(dep, arr, flight_date)
                flights.extend(route_flights)
            if direction in ("arrival", "both") and arr == airport.upper():
                route_flights = await self.get_flights_by_route(dep, arr, flight_date)
                flights.extend(route_flights)

        return flights

    async def get_flight_status(
        self,
        flight_number: str,
        flight_date: date,
    ) -> FlightInfo | None:
        """Return None for mock - would need to track generated flights."""
        return None


# Register mock client
DataClientFactory.register_flight_client("mock", MockFlightClient)
