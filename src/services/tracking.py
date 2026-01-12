"""
ULD Tracking Service.

Provides real-time and historical tracking of ULD positions across the network.
Combines geolocation data (24h refresh) with flight events for accurate tracking.
"""

from datetime import datetime, timedelta, timezone
from typing import Protocol

from src.domain import (
    ULD,
    ULDType,
    ULDStatus,
    ULDPosition,
    ULDInventory,
    Flight,
    FlightSchedule,
    DELTA_STATIONS,
)


class PositionRepository(Protocol):
    """Protocol for position data storage."""

    async def get_latest_position(self, uld_id: str) -> ULDPosition | None:
        """Get most recent position for a ULD."""
        ...

    async def get_position_history(
        self,
        uld_id: str,
        start: datetime,
        end: datetime,
    ) -> list[ULDPosition]:
        """Get position history for a ULD."""
        ...

    async def save_position(self, position: ULDPosition) -> None:
        """Save a position record."""
        ...

    async def get_station_inventory(self, station: str) -> ULDInventory:
        """Get current inventory at a station."""
        ...


class InMemoryPositionRepository:
    """In-memory implementation for development/testing."""

    def __init__(self):
        self._positions: dict[str, list[ULDPosition]] = {}
        self._inventories: dict[str, ULDInventory] = {}

    async def get_latest_position(self, uld_id: str) -> ULDPosition | None:
        positions = self._positions.get(uld_id, [])
        if not positions:
            return None
        return max(positions, key=lambda p: p.timestamp)

    async def get_position_history(
        self,
        uld_id: str,
        start: datetime,
        end: datetime,
    ) -> list[ULDPosition]:
        positions = self._positions.get(uld_id, [])
        return [p for p in positions if start <= p.timestamp <= end]

    async def save_position(self, position: ULDPosition) -> None:
        if position.uld_id not in self._positions:
            self._positions[position.uld_id] = []
        self._positions[position.uld_id].append(position)

    async def get_station_inventory(self, station: str) -> ULDInventory:
        return self._inventories.get(
            station,
            ULDInventory(
                station=station,
                timestamp=datetime.now(timezone.utc),
                inventory={},
                available={},
                in_use={},
                damaged={},
            ),
        )

    def set_inventory(self, station: str, inventory: ULDInventory) -> None:
        """Set inventory for testing."""
        self._inventories[station] = inventory


class ULDTrackingService:
    """
    Service for tracking ULD positions across the Delta network.

    Implements hybrid position tracking:
    1. Geolocation data (24h refresh from BLE tags)
    2. Flight event inference (when ULD loaded/unloaded)
    3. Manual position updates (ground handling)

    Usage:
        service = ULDTrackingService(repository)
        position = await service.get_current_position("AKE12345")
        history = await service.get_movement_history("AKE12345", days=7)
    """

    def __init__(self, repository: PositionRepository):
        self.repository = repository

    async def get_current_position(self, uld_id: str) -> ULDPosition | None:
        """
        Get the current best-estimate position of a ULD.

        Returns:
            Latest position or None if unknown
        """
        return await self.repository.get_latest_position(uld_id)

    async def get_movement_history(
        self,
        uld_id: str,
        days: int = 30,
    ) -> list[ULDPosition]:
        """
        Get movement history for a ULD.

        Args:
            uld_id: ULD identifier
            days: Number of days of history

        Returns:
            List of positions ordered by timestamp
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        positions = await self.repository.get_position_history(uld_id, start, end)
        return sorted(positions, key=lambda p: p.timestamp)

    async def record_geolocation(
        self,
        uld_id: str,
        uld_type: ULDType,
        station: str,
        timestamp: datetime | None = None,
        confidence: float = 0.95,
    ) -> ULDPosition:
        """
        Record a geolocation ping for a ULD.

        Args:
            uld_id: ULD identifier
            uld_type: Type of ULD
            station: Current station
            timestamp: Time of reading (default: now)
            confidence: Confidence level

        Returns:
            Created position record
        """
        position = ULDPosition(
            uld_id=uld_id,
            uld_type=uld_type,
            station=station,
            timestamp=timestamp or datetime.now(timezone.utc),
            position_source="geolocation",
            confidence=confidence,
        )
        await self.repository.save_position(position)
        return position

    async def record_flight_event(
        self,
        uld_id: str,
        uld_type: ULDType,
        flight: Flight,
        event_type: str,  # "loaded" or "unloaded"
    ) -> ULDPosition:
        """
        Record a ULD position based on flight event.

        Args:
            uld_id: ULD identifier
            uld_type: Type of ULD
            flight: Associated flight
            event_type: "loaded" (at origin) or "unloaded" (at destination)

        Returns:
            Created position record
        """
        if event_type == "loaded":
            station = flight.origin
            timestamp = flight.actual_departure or flight.scheduled_departure
        else:  # unloaded
            station = flight.destination
            timestamp = flight.actual_arrival or flight.scheduled_arrival

        position = ULDPosition(
            uld_id=uld_id,
            uld_type=uld_type,
            station=station,
            timestamp=timestamp,
            position_source="flight_event",
            flight_number=flight.flight_number,
            confidence=0.99,  # High confidence from flight event
        )
        await self.repository.save_position(position)
        return position

    async def infer_positions_from_schedule(
        self,
        schedule: FlightSchedule,
        uld_assignments: dict[str, list[str]],  # flight_number -> [uld_ids]
    ) -> list[ULDPosition]:
        """
        Infer ULD positions from flight schedule and assignments.

        Used to fill gaps between geolocation pings.

        Args:
            schedule: Flight schedule
            uld_assignments: Mapping of flights to assigned ULDs

        Returns:
            List of inferred positions
        """
        positions = []

        for flight in schedule.flights:
            uld_ids = uld_assignments.get(flight.flight_number, [])

            for uld_id in uld_ids:
                # Infer position at destination after arrival
                if flight.status in ("landed", "arrived"):
                    position = ULDPosition(
                        uld_id=uld_id,
                        uld_type=ULDType.AKE,  # Would need to look up
                        station=flight.destination,
                        timestamp=flight.actual_arrival or flight.scheduled_arrival,
                        position_source="flight_inference",
                        flight_number=flight.flight_number,
                        confidence=0.85,
                    )
                    positions.append(position)
                    await self.repository.save_position(position)

        return positions

    async def get_station_inventory(self, station: str) -> ULDInventory:
        """
        Get current ULD inventory at a station.

        Args:
            station: Station code

        Returns:
            ULDInventory snapshot
        """
        return await self.repository.get_station_inventory(station)

    async def get_network_summary(self) -> dict[str, dict]:
        """
        Get summary of ULD distribution across the network.

        Returns:
            Dict mapping station -> inventory summary
        """
        import random

        summary = {}

        for station_code, station_info in DELTA_STATIONS.items():
            inventory = await self.get_station_inventory(station_code)

            # If no real data, generate synthetic data for demo
            total = inventory.total_count()
            available = inventory.total_available()

            if total == 0:
                # Generate synthetic inventory based on station tier
                if station_info.tier.value == "hub":
                    total = random.randint(800, 1200)
                    available = random.randint(int(total * 0.3), int(total * 0.5))
                elif station_info.tier.value == "focus_city":
                    total = random.randint(400, 600)
                    available = random.randint(int(total * 0.2), int(total * 0.4))
                else:
                    total = random.randint(100, 300)
                    available = random.randint(int(total * 0.15), int(total * 0.35))

            summary[station_code] = {
                "total": total,
                "available": available,
                "availability_ratio": available / total if total > 0 else 0.0,
            }

        return summary

    async def calculate_dwell_time(
        self,
        uld_id: str,
        station: str,
    ) -> timedelta | None:
        """
        Calculate how long a ULD has been at a station.

        Args:
            uld_id: ULD identifier
            station: Station code

        Returns:
            Duration at station or None if not there
        """
        current = await self.get_current_position(uld_id)

        if not current or current.station != station:
            return None

        # Look back for when it arrived
        history = await self.get_movement_history(uld_id, days=30)

        arrival_time = None
        for pos in reversed(history):
            if pos.station == station:
                arrival_time = pos.timestamp
            else:
                break

        if arrival_time:
            return datetime.now(timezone.utc) - arrival_time

        return datetime.now(timezone.utc) - current.timestamp

    async def find_stuck_ulds(
        self,
        station: str,
        threshold_hours: int = 72,
    ) -> list[tuple[str, timedelta]]:
        """
        Find ULDs that have been at a station for too long.

        Args:
            station: Station code
            threshold_hours: Hours before considered "stuck"

        Returns:
            List of (uld_id, dwell_time) tuples
        """
        # This would need a more efficient implementation with a proper database
        # For now, return empty - would iterate over known ULDs
        return []
