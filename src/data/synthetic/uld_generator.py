"""
ULD fleet and inventory generator.

Generates realistic ULD fleet data:
- Fleet composition by type and owner
- Initial distribution across stations
- Movement history
- Maintenance states
"""

import random
from datetime import datetime, timedelta, timezone
from typing import Iterator
import uuid

import numpy as np

from src.domain import (
    ULD,
    ULDType,
    ULDStatus,
    ULDPosition,
    ULDInventory,
    DELTA_STATIONS,
    StationTier,
)


class ULDFleetGenerator:
    """
    Generate realistic synthetic ULD fleet data.

    Based on typical airline ULD operations:
    - Fleet size proportional to widebody operations
    - Mix of owned and leased units
    - Distribution weighted toward hubs
    - Realistic age and condition distribution
    """

    # Fleet composition by type (percentages)
    FLEET_COMPOSITION = {
        ULDType.AKE: 0.50,  # LD3 - most common
        ULDType.AKH: 0.15,  # LD3-45
        ULDType.PMC: 0.20,  # Standard pallets
        ULDType.AKN: 0.10,  # LD7 wider
        ULDType.AAP: 0.05,  # LD9 large
    }

    # Status distribution for serviceable fleet
    STATUS_DISTRIBUTION = {
        ULDStatus.SERVICEABLE: 0.25,  # Available at station
        ULDStatus.EMPTY: 0.10,  # At station, ready for loading
        ULDStatus.IN_USE: 0.40,  # Currently loaded
        ULDStatus.IN_TRANSIT: 0.15,  # On aircraft
        ULDStatus.OUT_OF_SERVICE: 0.05,  # At MRO
        ULDStatus.DAMAGED: 0.03,  # Awaiting inspection
        ULDStatus.CUSTOMS_HOLD: 0.02,  # Customs hold
    }

    # Ownership distribution
    OWNERSHIP_DISTRIBUTION = {
        "owned": 0.60,  # Delta-owned
        "leased": 0.25,  # Leased from pool
        "borrowed": 0.15,  # Borrowed from partners
    }

    def __init__(self, seed: int | None = None):
        """Initialize generator with optional random seed."""
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def generate_fleet(
        self,
        total_units: int = 5000,
        stations: list[str] | None = None,
    ) -> list[ULD]:
        """
        Generate complete ULD fleet.

        Args:
            total_units: Total number of ULDs to generate
            stations: Stations to distribute across (default: all)

        Returns:
            List of ULD objects
        """
        stations = stations or list(DELTA_STATIONS.keys())
        return list(self._generate_ulds(total_units, stations))

    def generate_inventory(
        self,
        station: str,
        timestamp: datetime | None = None,
    ) -> ULDInventory:
        """
        Generate current inventory snapshot for a station.

        Args:
            station: Station code
            timestamp: Time of snapshot (default: now)

        Returns:
            ULDInventory for the station
        """
        timestamp = timestamp or datetime.now(timezone.utc)

        # Get station tier for sizing
        station_info = DELTA_STATIONS.get(station)
        tier = station_info.tier if station_info else StationTier.SPOKE

        # Base units by tier
        tier_base = {
            StationTier.HUB: 300,  # High volume hub
            StationTier.FOCUS_CITY: 80,
            StationTier.SPOKE: 20,
            StationTier.INTERNATIONAL: 50,
        }

        base_units = tier_base.get(tier, 30)

        # Generate ULDs for this station
        ulds = list(self._generate_ulds(base_units, [station]))

        # Build inventory by type and status
        inventory: dict[ULDType, int] = {}
        available: dict[ULDType, int] = {}
        in_use: dict[ULDType, int] = {}
        damaged: dict[ULDType, int] = {}

        for uld in ulds:
            # Initialize type if not present
            if uld.uld_type not in inventory:
                inventory[uld.uld_type] = 0
                available[uld.uld_type] = 0
                in_use[uld.uld_type] = 0
                damaged[uld.uld_type] = 0

            inventory[uld.uld_type] += 1

            # Categorize by status
            if uld.status in (ULDStatus.SERVICEABLE, ULDStatus.EMPTY):
                available[uld.uld_type] += 1
            elif uld.status == ULDStatus.IN_USE:
                in_use[uld.uld_type] += 1
            elif uld.status == ULDStatus.DAMAGED:
                damaged[uld.uld_type] += 1

        return ULDInventory(
            station=station,
            timestamp=timestamp,
            inventory=inventory,
            available=available,
            in_use=in_use,
            damaged=damaged,
        )

    def _generate_ulds(
        self,
        count: int,
        stations: list[str],
    ) -> Iterator[ULD]:
        """Generate individual ULDs."""
        # Calculate station weights based on tier
        weights = []
        for station in stations:
            info = DELTA_STATIONS.get(station)
            tier = info.tier if info else StationTier.SPOKE

            tier_weight = {
                StationTier.HUB: 8,  # High volume
                StationTier.FOCUS_CITY: 3,
                StationTier.SPOKE: 1,
                StationTier.INTERNATIONAL: 2,
            }
            weights.append(tier_weight.get(tier, 1))

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        for i in range(count):
            # Select type based on composition
            uld_type = self._weighted_choice(
                list(self.FLEET_COMPOSITION.keys()),
                list(self.FLEET_COMPOSITION.values()),
            )

            # Select status
            status = self._weighted_choice(
                list(self.STATUS_DISTRIBUTION.keys()),
                list(self.STATUS_DISTRIBUTION.values()),
            )

            # Select current station
            current_station = self._weighted_choice(stations, weights)

            # Select ownership
            ownership = self._weighted_choice(
                list(self.OWNERSHIP_DISTRIBUTION.keys()),
                list(self.OWNERSHIP_DISTRIBUTION.values()),
            )

            # Generate ULD ID (airline prefix + type code + serial)
            owner_code = "DL" if ownership == "owned" else "UU"  # UU = universal pool
            serial = f"{i+1:05d}"
            uld_id = f"{uld_type.value}{owner_code}{serial}"

            # Generate manufacturing date (2-20 years old)
            age_years = float(self.rng.uniform(2, 20))
            manufactured = datetime.now(timezone.utc) - timedelta(days=int(age_years * 365))

            # Inspection dates
            last_inspection = datetime.now(timezone.utc) - timedelta(
                days=int(self.rng.integers(1, 180))
            )
            next_inspection = last_inspection + timedelta(days=180)

            # Condition rating (1-5)
            if age_years < 5:
                condition = int(self.rng.integers(4, 6))
            elif age_years < 10:
                condition = int(self.rng.integers(3, 5))
            else:
                condition = int(self.rng.integers(2, 4))

            yield ULD(
                uld_id=uld_id,
                uld_type=uld_type,
                status=status,
                current_station=current_station if status != ULDStatus.IN_TRANSIT else None,
                owner_airline="DL" if ownership == "owned" else "XX",
                is_leased=ownership != "owned",
                manufacturer="Nordisk" if self.rng.random() > 0.5 else "VRR",
                manufactured_date=manufactured,
                last_inspection_date=last_inspection,
                next_inspection_due=next_inspection,
                condition_rating=condition,
                total_flight_cycles=int(age_years * 365 * self.rng.uniform(0.5, 1.5)),
            )

    def _weighted_choice(self, items: list, weights: list):
        """Make weighted random choice."""
        return random.choices(items, weights=weights, k=1)[0]

    def generate_position_history(
        self,
        uld_id: str,
        uld_type: ULDType,
        start_date: datetime,
        end_date: datetime,
        stations: list[str] | None = None,
    ) -> list[ULDPosition]:
        """
        Generate position history for a ULD.

        Args:
            uld_id: ULD identifier
            uld_type: Type of ULD
            start_date: Start of history period
            end_date: End of history period
            stations: Possible stations (default: all)

        Returns:
            List of ULDPosition records
        """
        stations = stations or list(DELTA_STATIONS.keys())
        positions = []

        current_time = start_date
        current_station = random.choice(stations)

        while current_time < end_date:
            # Create position record
            positions.append(
                ULDPosition(
                    uld_id=uld_id,
                    uld_type=uld_type,
                    station=current_station,
                    timestamp=current_time,
                    position_source="geolocation",
                    flight_number=None,
                    confidence=0.95,
                )
            )

            # Simulate movement
            # Average dwell time at station: 4-48 hours
            dwell_hours = self.rng.exponential(12)
            dwell_hours = min(dwell_hours, 72)  # Cap at 3 days

            current_time += timedelta(hours=dwell_hours)

            # Move to new station with some probability
            if self.rng.random() > 0.3:  # 70% chance of movement
                # Prefer hub-spoke patterns
                available_hubs = [s for s in ["ATL", "DTW", "MSP", "SLC"] if s in stations]
                if current_station in available_hubs:
                    # Hub -> spoke more likely
                    current_station = random.choice(stations)
                else:
                    # Spoke -> hub more likely
                    if self.rng.random() > 0.3 and available_hubs:
                        current_station = random.choice(available_hubs)
                    else:
                        current_station = random.choice(stations)

        return positions

    def simulate_day_movements(
        self,
        inventory: ULDInventory,
        departure_flights: int,
        arrival_flights: int,
    ) -> tuple[ULDInventory, int, int]:
        """
        Simulate one day of ULD movements at a station.

        Args:
            inventory: Starting inventory
            departure_flights: Number of departing widebody flights
            arrival_flights: Number of arriving widebody flights

        Returns:
            Tuple of (new_inventory, ulds_departed, ulds_arrived)
        """
        # Average ULDs per widebody flight
        ulds_per_flight = 8

        departures = departure_flights * ulds_per_flight
        arrivals = arrival_flights * ulds_per_flight

        # Add some randomness
        departures = max(0, departures + self.rng.integers(-5, 6))
        arrivals = max(0, arrivals + self.rng.integers(-5, 6))

        # Update serviceable count
        new_serviceable = inventory.serviceable_count - departures + arrivals
        new_serviceable = max(0, new_serviceable)

        # Create new inventory (simplified)
        new_inventory = ULDInventory(
            station=inventory.station,
            timestamp=inventory.timestamp + timedelta(days=1),
            by_type=inventory.by_type,
            by_status=inventory.by_status,
            total_count=inventory.total_count - departures + arrivals,
            serviceable_count=new_serviceable,
            ulds=[],  # Would need to track individual ULDs
        )

        return new_inventory, departures, arrivals
