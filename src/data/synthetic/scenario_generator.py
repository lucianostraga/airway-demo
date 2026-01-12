"""
Scenario generator for disruption and what-if analysis.

Generates realistic operational scenarios:
- Weather disruptions (snow, thunderstorms, fog)
- Capacity constraints
- Demand spikes (events, holidays)
- Equipment failures
"""

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from dataclasses import dataclass

import numpy as np

from src.domain import DELTA_STATIONS, StationTier


class ScenarioType(str, Enum):
    """Types of operational scenarios."""

    WEATHER_SNOW = "weather_snow"
    WEATHER_THUNDERSTORM = "weather_thunderstorm"
    WEATHER_FOG = "weather_fog"
    WEATHER_HURRICANE = "weather_hurricane"

    DEMAND_SPIKE = "demand_spike"
    DEMAND_DROP = "demand_drop"

    CAPACITY_GROUND_STOP = "capacity_ground_stop"
    CAPACITY_REDUCED = "capacity_reduced"

    EQUIPMENT_SHORTAGE = "equipment_shortage"
    EQUIPMENT_EXCESS = "equipment_excess"

    EVENT_SPORTS = "event_sports"
    EVENT_CONVENTION = "event_convention"
    EVENT_CONCERT = "event_concert"


@dataclass
class ScenarioEvent:
    """A single scenario event."""

    event_type: ScenarioType
    station: str
    start_time: datetime
    end_time: datetime
    severity: float  # 0-1 scale
    impact_factor: float  # Multiplier on operations
    description: str
    affected_flights: int = 0
    affected_ulds: int = 0


@dataclass
class Scenario:
    """Complete scenario with multiple events."""

    scenario_id: str
    name: str
    description: str
    events: list[ScenarioEvent]
    start_time: datetime
    end_time: datetime
    total_impact_flights: int = 0
    total_impact_ulds: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "events": [
                {
                    "event_type": e.event_type.value,
                    "station": e.station,
                    "start_time": e.start_time.isoformat(),
                    "end_time": e.end_time.isoformat(),
                    "severity": e.severity,
                    "impact_factor": e.impact_factor,
                    "description": e.description,
                }
                for e in self.events
            ],
        }


class ScenarioGenerator:
    """
    Generate realistic operational scenarios for testing and analysis.

    Creates scenarios based on:
    - Historical disruption patterns
    - Seasonal weather patterns
    - Event calendars
    - Operational constraints
    """

    # Weather patterns by season and region
    WEATHER_PATTERNS = {
        "winter": {
            "northeast": [(ScenarioType.WEATHER_SNOW, 0.15)],
            "midwest": [(ScenarioType.WEATHER_SNOW, 0.12)],
            "southeast": [(ScenarioType.WEATHER_FOG, 0.05)],
            "west": [(ScenarioType.WEATHER_FOG, 0.03)],
        },
        "summer": {
            "southeast": [(ScenarioType.WEATHER_THUNDERSTORM, 0.20)],
            "midwest": [(ScenarioType.WEATHER_THUNDERSTORM, 0.15)],
            "gulf": [(ScenarioType.WEATHER_HURRICANE, 0.05)],
        },
    }

    # Station regions
    STATION_REGIONS = {
        "northeast": ["JFK", "LGA", "BOS", "DCA", "EWR"],
        "midwest": ["DTW", "MSP", "ORD"],
        "southeast": ["ATL", "MIA"],
        "west": ["LAX", "SFO", "SEA", "PHX"],
        "mountain": ["SLC", "DEN"],
        "gulf": ["MIA", "ATL"],  # Hurricane risk
    }

    def __init__(self, seed: int | None = None):
        """Initialize generator."""
        self.rng = np.random.default_rng(seed)
        self._scenario_counter = 0

    def generate_scenario(
        self,
        scenario_type: ScenarioType,
        station: str | None = None,
        start_time: datetime | None = None,
        duration_hours: float | None = None,
        severity: float | None = None,
    ) -> Scenario:
        """
        Generate a single scenario.

        Args:
            scenario_type: Type of scenario to generate
            station: Affected station (random if not specified)
            start_time: Start time (random if not specified)
            duration_hours: Duration (type-appropriate default if not specified)
            severity: Severity 0-1 (random if not specified)

        Returns:
            Generated Scenario
        """
        self._scenario_counter += 1

        # Default values
        station = station or self._random_station()
        start_time = start_time or self._random_time()
        severity = severity if severity is not None else self.rng.uniform(0.3, 1.0)

        # Type-specific defaults
        if duration_hours is None:
            duration_hours = self._default_duration(scenario_type)

        end_time = start_time + timedelta(hours=duration_hours)

        # Calculate impact
        impact_factor = self._calculate_impact(scenario_type, severity)

        event = ScenarioEvent(
            event_type=scenario_type,
            station=station,
            start_time=start_time,
            end_time=end_time,
            severity=severity,
            impact_factor=impact_factor,
            description=self._generate_description(scenario_type, station, severity),
            affected_flights=int(duration_hours * 5 * severity),  # Rough estimate
            affected_ulds=int(duration_hours * 40 * severity),
        )

        return Scenario(
            scenario_id=f"SCN-{self._scenario_counter:05d}",
            name=f"{scenario_type.value.replace('_', ' ').title()} at {station}",
            description=event.description,
            events=[event],
            start_time=start_time,
            end_time=end_time,
            total_impact_flights=event.affected_flights,
            total_impact_ulds=event.affected_ulds,
        )

    def generate_winter_storm_scenario(
        self,
        severity: float = 0.7,
    ) -> Scenario:
        """Generate a winter storm affecting multiple Northeast stations."""
        self._scenario_counter += 1

        start_time = self._random_time(season="winter")
        stations = ["JFK", "LGA", "BOS", "EWR"]

        events = []
        for station in stations:
            # Stagger impact times slightly
            station_start = start_time + timedelta(hours=self.rng.uniform(0, 4))
            duration = self.rng.uniform(8, 24)

            event = ScenarioEvent(
                event_type=ScenarioType.WEATHER_SNOW,
                station=station,
                start_time=station_start,
                end_time=station_start + timedelta(hours=duration),
                severity=severity * self.rng.uniform(0.8, 1.0),
                impact_factor=0.3,  # Severe reduction
                description=f"Heavy snowfall affecting operations at {station}",
                affected_flights=int(duration * 8),
                affected_ulds=int(duration * 60),
            )
            events.append(event)

        end_time = max(e.end_time for e in events)

        return Scenario(
            scenario_id=f"SCN-{self._scenario_counter:05d}",
            name="Northeast Winter Storm",
            description="Major winter storm affecting Northeast operations",
            events=events,
            start_time=start_time,
            end_time=end_time,
            total_impact_flights=sum(e.affected_flights for e in events),
            total_impact_ulds=sum(e.affected_ulds for e in events),
        )

    def generate_summer_thunderstorm_scenario(
        self,
        station: str = "ATL",
        severity: float = 0.6,
    ) -> Scenario:
        """Generate afternoon thunderstorm scenario."""
        self._scenario_counter += 1

        # Thunderstorms typically afternoon/evening
        base_date = self._random_time(season="summer").date()
        start_time = datetime.combine(
            base_date, datetime.min.time()
        ).replace(hour=15, tzinfo=timezone.utc)

        duration = self.rng.uniform(2, 6)  # 2-6 hours

        event = ScenarioEvent(
            event_type=ScenarioType.WEATHER_THUNDERSTORM,
            station=station,
            start_time=start_time,
            end_time=start_time + timedelta(hours=duration),
            severity=severity,
            impact_factor=0.5,  # Moderate reduction
            description=f"Severe thunderstorms causing ground stops at {station}",
            affected_flights=int(duration * 15),
            affected_ulds=int(duration * 100),
        )

        return Scenario(
            scenario_id=f"SCN-{self._scenario_counter:05d}",
            name=f"Thunderstorm at {station}",
            description=event.description,
            events=[event],
            start_time=start_time,
            end_time=event.end_time,
            total_impact_flights=event.affected_flights,
            total_impact_ulds=event.affected_ulds,
        )

    def generate_demand_spike_scenario(
        self,
        station: str = "ATL",
        event_name: str = "Major Event",
        severity: float = 0.5,
    ) -> Scenario:
        """Generate demand spike from major event."""
        self._scenario_counter += 1

        start_time = self._random_time()
        duration = 72  # 3 days for event

        event = ScenarioEvent(
            event_type=ScenarioType.EVENT_SPORTS,
            station=station,
            start_time=start_time,
            end_time=start_time + timedelta(hours=duration),
            severity=severity,
            impact_factor=1.5,  # 50% increase
            description=f"{event_name} causing increased demand at {station}",
            affected_flights=0,  # No cancellations
            affected_ulds=int(duration * 10 * severity),  # Additional ULDs needed
        )

        return Scenario(
            scenario_id=f"SCN-{self._scenario_counter:05d}",
            name=f"{event_name} at {station}",
            description=event.description,
            events=[event],
            start_time=start_time,
            end_time=event.end_time,
            total_impact_flights=0,
            total_impact_ulds=event.affected_ulds,
        )

    def generate_random_scenarios(
        self,
        count: int = 10,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[Scenario]:
        """
        Generate multiple random scenarios.

        Args:
            count: Number of scenarios to generate
            start_date: Start of period
            end_date: End of period

        Returns:
            List of Scenario objects
        """
        scenarios = []

        # Scenario type weights
        types = list(ScenarioType)
        weights = [
            0.15,  # snow
            0.20,  # thunderstorm
            0.10,  # fog
            0.02,  # hurricane
            0.15,  # demand spike
            0.08,  # demand drop
            0.05,  # ground stop
            0.05,  # reduced capacity
            0.05,  # equipment shortage
            0.05,  # equipment excess
            0.05,  # sports
            0.03,  # convention
            0.02,  # concert
        ]

        for _ in range(count):
            type_idx = self.rng.choice(len(types), p=weights)
            scenario_type = types[type_idx]
            start_time = start_date if start_date else self._random_time()

            scenario = self.generate_scenario(
                scenario_type=scenario_type,
                start_time=start_time,
            )
            scenarios.append(scenario)

        return scenarios

    def _random_station(self) -> str:
        """Select random station weighted by tier."""
        stations = list(DELTA_STATIONS.keys())
        weights = []

        for station in stations:
            info = DELTA_STATIONS[station]
            tier_weight = {
                StationTier.HUB: 5,  # High volume
                StationTier.FOCUS_CITY: 3,
                StationTier.SPOKE: 1,
                StationTier.INTERNATIONAL: 2,
            }
            weights.append(tier_weight.get(info.tier, 1))

        weights = np.array(weights) / sum(weights)
        return self.rng.choice(stations, p=weights)

    def _random_time(
        self,
        season: str | None = None,
    ) -> datetime:
        """Generate random datetime, optionally in a specific season."""
        base = datetime.now(timezone.utc)

        if season == "winter":
            # December-February
            month = self.rng.choice([12, 1, 2])
            if month == 12:
                year = base.year
            else:
                year = base.year + 1
        elif season == "summer":
            # June-August
            month = self.rng.choice([6, 7, 8])
            year = base.year
        else:
            # Random
            days_ahead = int(self.rng.integers(0, 90))
            return base + timedelta(days=days_ahead)

        day = self.rng.integers(1, 29)
        hour = self.rng.integers(0, 24)

        return datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)

    def _default_duration(self, scenario_type: ScenarioType) -> float:
        """Get default duration in hours for scenario type."""
        durations = {
            ScenarioType.WEATHER_SNOW: 18,
            ScenarioType.WEATHER_THUNDERSTORM: 4,
            ScenarioType.WEATHER_FOG: 6,
            ScenarioType.WEATHER_HURRICANE: 48,
            ScenarioType.DEMAND_SPIKE: 72,
            ScenarioType.DEMAND_DROP: 48,
            ScenarioType.CAPACITY_GROUND_STOP: 3,
            ScenarioType.CAPACITY_REDUCED: 8,
            ScenarioType.EQUIPMENT_SHORTAGE: 24,
            ScenarioType.EQUIPMENT_EXCESS: 24,
            ScenarioType.EVENT_SPORTS: 48,
            ScenarioType.EVENT_CONVENTION: 96,
            ScenarioType.EVENT_CONCERT: 24,
        }
        return durations.get(scenario_type, 12)

    def _calculate_impact(
        self,
        scenario_type: ScenarioType,
        severity: float,
    ) -> float:
        """Calculate operational impact factor."""
        # Base impact by type
        base_impacts = {
            ScenarioType.WEATHER_SNOW: 0.4,
            ScenarioType.WEATHER_THUNDERSTORM: 0.5,
            ScenarioType.WEATHER_FOG: 0.6,
            ScenarioType.WEATHER_HURRICANE: 0.1,
            ScenarioType.DEMAND_SPIKE: 1.5,
            ScenarioType.DEMAND_DROP: 0.7,
            ScenarioType.CAPACITY_GROUND_STOP: 0.0,
            ScenarioType.CAPACITY_REDUCED: 0.5,
            ScenarioType.EQUIPMENT_SHORTAGE: 0.8,
            ScenarioType.EQUIPMENT_EXCESS: 1.0,
            ScenarioType.EVENT_SPORTS: 1.3,
            ScenarioType.EVENT_CONVENTION: 1.2,
            ScenarioType.EVENT_CONCERT: 1.1,
        }

        base = base_impacts.get(scenario_type, 1.0)

        # Adjust by severity
        if base < 1.0:  # Reduction scenarios
            return base + (1.0 - base) * (1.0 - severity)
        else:  # Increase scenarios
            return 1.0 + (base - 1.0) * severity

    def _generate_description(
        self,
        scenario_type: ScenarioType,
        station: str,
        severity: float,
    ) -> str:
        """Generate human-readable description."""
        severity_text = "severe" if severity > 0.7 else "moderate" if severity > 0.4 else "minor"

        descriptions = {
            ScenarioType.WEATHER_SNOW: f"{severity_text.title()} snowfall affecting operations at {station}",
            ScenarioType.WEATHER_THUNDERSTORM: f"{severity_text.title()} thunderstorms causing delays at {station}",
            ScenarioType.WEATHER_FOG: f"Dense fog reducing visibility at {station}",
            ScenarioType.WEATHER_HURRICANE: f"Hurricane preparation/impact at {station}",
            ScenarioType.DEMAND_SPIKE: f"Unusual demand increase at {station}",
            ScenarioType.DEMAND_DROP: f"Reduced demand at {station}",
            ScenarioType.CAPACITY_GROUND_STOP: f"Ground stop in effect at {station}",
            ScenarioType.CAPACITY_REDUCED: f"Reduced airport capacity at {station}",
            ScenarioType.EQUIPMENT_SHORTAGE: f"ULD equipment shortage at {station}",
            ScenarioType.EQUIPMENT_EXCESS: f"ULD equipment surplus at {station}",
            ScenarioType.EVENT_SPORTS: f"Major sporting event at {station}",
            ScenarioType.EVENT_CONVENTION: f"Large convention at {station}",
            ScenarioType.EVENT_CONCERT: f"Major concert event at {station}",
        }

        return descriptions.get(scenario_type, f"Operational event at {station}")
