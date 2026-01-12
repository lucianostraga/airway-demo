"""
Flight schedule generator for synthetic data.

Generates realistic flight schedules based on Delta's network structure:
- Hub-and-spoke patterns from ATL, DTW, MSP, SLC
- Time-of-day distributions (banks)
- Day-of-week variations
- Seasonal fluctuations
"""

import random
from datetime import datetime, timedelta, timezone
from typing import Iterator

import numpy as np

from src.domain import (
    Flight,
    FlightStatus,
    FlightSchedule,
    Route,
    DELTA_STATIONS,
    AIRCRAFT_TYPES,
)


class FlightScheduleGenerator:
    """
    Generate realistic synthetic flight schedules.

    Based on typical hub-and-spoke airline operations with:
    - Morning, midday, and evening banks
    - Higher frequency on major routes
    - Appropriate aircraft types for route length
    - Realistic passenger loads
    """

    # Delta hub stations
    HUBS = ["ATL", "DTW", "MSP", "SLC"]

    # Focus cities (higher frequency than spokes)
    FOCUS_CITIES = ["JFK", "LGA", "LAX", "SEA", "BOS"]

    # Time banks (UTC hours) - represents typical hub connection windows
    BANK_TIMES = {
        "morning_1": (6, 8),
        "morning_2": (9, 11),
        "midday": (12, 14),
        "afternoon": (15, 17),
        "evening": (18, 20),
        "night": (21, 23),
    }

    # Route frequency templates (flights per day)
    ROUTE_FREQUENCIES = {
        "hub_to_hub": 8,  # ATL-DTW, etc.
        "hub_to_focus": 6,  # ATL-JFK, etc.
        "hub_to_spoke": 3,  # ATL-smaller cities
        "focus_to_focus": 2,  # JFK-LAX, etc.
    }

    def __init__(self, seed: int | None = None):
        """Initialize generator with optional random seed."""
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        self._flight_counter = 100

    def generate_schedule(
        self,
        start_date: datetime,
        end_date: datetime,
        stations: list[str] | None = None,
    ) -> FlightSchedule:
        """
        Generate complete flight schedule for a date range.

        Args:
            start_date: Start of schedule period
            end_date: End of schedule period
            stations: Stations to include (default: all Delta stations)

        Returns:
            FlightSchedule with all generated flights
        """
        stations = stations or list(DELTA_STATIONS.keys())
        flights = list(self._generate_flights(start_date, end_date, stations))

        return FlightSchedule(
            flights=flights,
            period_start=start_date,
            period_end=end_date,
            station=None,
        )

    def generate_day_schedule(
        self,
        date: datetime,
        stations: list[str] | None = None,
    ) -> FlightSchedule:
        """Generate schedule for a single day."""
        end_date = date + timedelta(days=1)
        return self.generate_schedule(date, end_date, stations)

    def _generate_flights(
        self,
        start_date: datetime,
        end_date: datetime,
        stations: list[str],
    ) -> Iterator[Flight]:
        """Generate flights for all routes in the network."""
        current_date = start_date

        while current_date < end_date:
            # Day-of-week factor (weekends slightly lower)
            dow = current_date.weekday()
            dow_factor = 0.85 if dow >= 5 else 1.0

            # Generate flights for each route type
            for origin in stations:
                for destination in stations:
                    if origin == destination:
                        continue

                    route_type = self._classify_route(origin, destination)
                    if route_type is None:
                        continue

                    base_freq = self.ROUTE_FREQUENCIES[route_type]
                    daily_flights = int(base_freq * dow_factor)

                    # Add some randomness
                    daily_flights = max(1, daily_flights + self.rng.integers(-1, 2))

                    for _ in range(daily_flights):
                        flight = self._generate_single_flight(
                            origin, destination, current_date
                        )
                        if flight:
                            yield flight

            current_date += timedelta(days=1)

    def _classify_route(self, origin: str, destination: str) -> str | None:
        """Classify route type for frequency determination."""
        origin_is_hub = origin in self.HUBS
        dest_is_hub = destination in self.HUBS
        origin_is_focus = origin in self.FOCUS_CITIES
        dest_is_focus = destination in self.FOCUS_CITIES

        if origin_is_hub and dest_is_hub:
            return "hub_to_hub"
        elif origin_is_hub and dest_is_focus:
            return "hub_to_focus"
        elif origin_is_hub:
            return "hub_to_spoke"
        elif origin_is_focus and dest_is_focus:
            return "focus_to_focus"
        # Skip other combinations to keep schedule realistic
        return None

    def _generate_single_flight(
        self,
        origin: str,
        destination: str,
        date: datetime,
    ) -> Flight | None:
        """Generate a single flight."""
        # Select bank time
        bank = random.choice(list(self.BANK_TIMES.keys()))
        hour_range = self.BANK_TIMES[bank]
        departure_hour = self.rng.integers(hour_range[0], hour_range[1] + 1)
        departure_minute = self.rng.integers(0, 12) * 5  # 5-minute increments

        scheduled_departure = date.replace(
            hour=departure_hour,
            minute=departure_minute,
            second=0,
            microsecond=0,
            tzinfo=timezone.utc,
        )

        # Calculate flight duration based on distance estimate
        duration_minutes = self._estimate_duration(origin, destination)
        scheduled_arrival = scheduled_departure + timedelta(minutes=duration_minutes)

        # Select appropriate aircraft
        aircraft_type = self._select_aircraft(duration_minutes)
        aircraft_spec = AIRCRAFT_TYPES.get(aircraft_type)

        # Generate passenger load
        if aircraft_spec:
            # Estimate capacity based on aircraft
            capacity = 180 if aircraft_spec.is_widebody else 160
            load_factor = self.rng.uniform(0.65, 0.92)
            passengers = int(capacity * load_factor)
        else:
            passengers = self.rng.integers(100, 180)

        # Estimate bags per passenger
        bags = int(passengers * self.rng.uniform(0.8, 1.2))

        # ULD capacity
        uld_capacity = aircraft_spec.lower_deck_positions if aircraft_spec else 0

        # Generate flight number
        flight_number = f"DL{self._flight_counter}"
        self._flight_counter += 1

        return Flight(
            flight_number=flight_number,
            airline="DL",
            origin=origin,
            destination=destination,
            scheduled_departure=scheduled_departure,
            scheduled_arrival=scheduled_arrival,
            status=FlightStatus.SCHEDULED,
            aircraft_type=aircraft_type,
            booked_passengers=passengers,
            estimated_bags=bags,
            uld_capacity=uld_capacity,
        )

    def _estimate_duration(self, origin: str, destination: str) -> int:
        """Estimate flight duration in minutes."""
        # Simplified distance estimates based on station locations
        # In production, use actual great-circle distances

        # Hub distances from ATL (approximate)
        atl_distances = {
            "ATL": 0,
            "DTW": 594,
            "MSP": 907,
            "SLC": 1589,
            "JFK": 760,
            "LGA": 762,
            "LAX": 1946,
            "SEA": 2182,
            "BOS": 946,
            "ORD": 606,
            "DFW": 731,
            "MIA": 594,
            "DEN": 1208,
            "SFO": 2139,
            "PHX": 1587,
        }

        # Rough approximation using ATL as reference
        orig_dist = atl_distances.get(origin, 800)
        dest_dist = atl_distances.get(destination, 800)

        # Very rough estimate
        estimated_miles = abs(orig_dist - dest_dist) if orig_dist != dest_dist else 500

        # Convert to duration (500 mph cruise + 30 min taxi/climb/descent)
        duration = int(estimated_miles / 500 * 60 + 30)
        return max(60, min(duration, 420))  # 1-7 hours

    def _select_aircraft(self, duration_minutes: int) -> str:
        """Select appropriate aircraft for route length."""
        if duration_minutes > 300:  # >5 hours
            # Long haul - widebody
            return random.choice(["B77W", "A359", "A333"])
        elif duration_minutes > 180:  # 3-5 hours
            # Medium haul - mix
            return random.choice(["B763", "A333", "A321", "B738"])
        else:
            # Short haul - narrowbody
            return random.choice(["B738", "A321", "A321"])

    def add_delays(
        self,
        schedule: FlightSchedule,
        delay_rate: float = 0.15,
    ) -> FlightSchedule:
        """
        Add realistic delays to a schedule.

        Args:
            schedule: Original schedule
            delay_rate: Probability of delay (default 15%)

        Returns:
            New schedule with delays applied
        """
        delayed_flights = []

        for flight in schedule.flights:
            if self.rng.random() < delay_rate:
                # Generate delay (exponential distribution)
                delay_minutes = int(self.rng.exponential(30))
                delay_minutes = min(delay_minutes, 240)  # Cap at 4 hours

                # Create modified flight
                delayed_flight = Flight(
                    **flight.model_dump(exclude={"delay_minutes", "status", "actual_departure"})
                )
                delayed_flight.delay_minutes = delay_minutes
                delayed_flight.status = FlightStatus.DELAYED
                delayed_flight.actual_departure = (
                    flight.scheduled_departure + timedelta(minutes=delay_minutes)
                )
                delayed_flights.append(delayed_flight)
            else:
                delayed_flights.append(flight)

        return FlightSchedule(
            flights=delayed_flights,
            period_start=schedule.period_start,
            period_end=schedule.period_end,
            station=schedule.station,
        )

    def add_cancellations(
        self,
        schedule: FlightSchedule,
        cancel_rate: float = 0.02,
    ) -> FlightSchedule:
        """
        Add cancellations to a schedule.

        Args:
            schedule: Original schedule
            cancel_rate: Probability of cancellation (default 2%)

        Returns:
            New schedule with cancellations
        """
        updated_flights = []

        cancellation_reasons = [
            "Weather",
            "Mechanical",
            "Crew availability",
            "Air traffic control",
            "Operational requirement",
        ]

        for flight in schedule.flights:
            if self.rng.random() < cancel_rate:
                cancelled_flight = Flight(
                    **flight.model_dump(exclude={"status", "cancellation_reason"})
                )
                cancelled_flight.status = FlightStatus.CANCELLED
                cancelled_flight.cancellation_reason = random.choice(cancellation_reasons)
                updated_flights.append(cancelled_flight)
            else:
                updated_flights.append(flight)

        return FlightSchedule(
            flights=updated_flights,
            period_start=schedule.period_start,
            period_end=schedule.period_end,
            station=schedule.station,
        )
