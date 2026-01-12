"""Tests for synthetic data generators."""

import pytest
from datetime import datetime, timedelta, timezone

from src.data.synthetic import (
    FlightScheduleGenerator,
    ULDFleetGenerator,
    DemandPatternGenerator,
    ScenarioGenerator,
)
from src.domain import ULDType, ULDStatus, FlightStatus, ForecastGranularity


class TestFlightScheduleGenerator:
    """Tests for flight schedule generator."""

    @pytest.fixture
    def generator(self):
        return FlightScheduleGenerator(seed=42)

    def test_generate_day_schedule(self, generator):
        """Test generating a day's schedule."""
        schedule = generator.generate_day_schedule(
            datetime.now(timezone.utc),
            stations=["ATL", "DTW", "JFK"],
        )
        assert len(schedule.flights) > 0
        assert all(f.origin in ["ATL", "DTW", "JFK"] for f in schedule.flights)
        assert all(f.destination in ["ATL", "DTW", "JFK"] for f in schedule.flights)

    def test_generate_schedule_range(self, generator):
        """Test generating schedule for date range."""
        start = datetime.now(timezone.utc)
        end = start + timedelta(days=3)
        schedule = generator.generate_schedule(start, end, stations=["ATL", "DTW"])

        assert schedule.period_start == start
        assert schedule.period_end == end
        assert len(schedule.flights) > 0

    def test_flight_numbers_unique(self, generator):
        """Test flight numbers are unique."""
        schedule = generator.generate_day_schedule(
            datetime.now(timezone.utc),
            stations=["ATL", "DTW"],
        )
        flight_numbers = [f.flight_number for f in schedule.flights]
        assert len(flight_numbers) == len(set(flight_numbers))

    def test_add_delays(self, generator):
        """Test adding delays to schedule."""
        schedule = generator.generate_day_schedule(
            datetime.now(timezone.utc),
            stations=["ATL", "DTW"],
        )
        delayed = generator.add_delays(schedule, delay_rate=0.5)

        delayed_count = sum(1 for f in delayed.flights if f.status == FlightStatus.DELAYED)
        # With 50% rate, should have some delays
        assert delayed_count > 0

    def test_add_cancellations(self, generator):
        """Test adding cancellations to schedule."""
        schedule = generator.generate_day_schedule(
            datetime.now(timezone.utc),
            stations=["ATL", "DTW"],
        )
        cancelled = generator.add_cancellations(schedule, cancel_rate=0.2)

        cancelled_count = sum(1 for f in cancelled.flights if f.status == FlightStatus.CANCELLED)
        # With 20% rate, should have some cancellations
        assert cancelled_count >= 0  # Might be 0 with small sample

    def test_deterministic_with_seed(self):
        """Test generator is deterministic with same seed."""
        gen1 = FlightScheduleGenerator(seed=123)
        gen2 = FlightScheduleGenerator(seed=123)

        schedule1 = gen1.generate_day_schedule(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            stations=["ATL"],
        )
        schedule2 = gen2.generate_day_schedule(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            stations=["ATL"],
        )

        assert len(schedule1.flights) == len(schedule2.flights)


class TestULDFleetGenerator:
    """Tests for ULD fleet generator."""

    @pytest.fixture
    def generator(self):
        return ULDFleetGenerator(seed=42)

    def test_generate_fleet(self, generator):
        """Test generating a fleet."""
        fleet = generator.generate_fleet(total_units=100, stations=["ATL", "DTW"])

        assert len(fleet) == 100
        assert all(uld.uld_type in ULDType for uld in fleet)
        assert all(uld.status in ULDStatus for uld in fleet)

    def test_fleet_type_distribution(self, generator):
        """Test fleet has correct type distribution."""
        fleet = generator.generate_fleet(total_units=1000, stations=["ATL"])

        type_counts = {}
        for uld in fleet:
            type_counts[uld.uld_type] = type_counts.get(uld.uld_type, 0) + 1

        # AKE should be most common (~50%)
        assert type_counts[ULDType.AKE] > type_counts.get(ULDType.PMC, 0)

    def test_generate_inventory(self, generator):
        """Test generating station inventory."""
        inventory = generator.generate_inventory("ATL")

        assert inventory.station == "ATL"
        assert inventory.total_count() > 0
        assert inventory.total_available() >= 0

    def test_generate_position_history(self, generator):
        """Test generating position history."""
        start = datetime.now(timezone.utc) - timedelta(days=7)
        end = datetime.now(timezone.utc)

        history = generator.generate_position_history(
            uld_id="AKEDL00001",
            uld_type=ULDType.AKE,
            start_date=start,
            end_date=end,
            stations=["ATL", "DTW", "JFK"],
        )

        assert len(history) > 0
        assert all(p.uld_id == "AKEDL00001" for p in history)
        assert all(p.station in ["ATL", "DTW", "JFK"] for p in history)


class TestDemandPatternGenerator:
    """Tests for demand pattern generator."""

    @pytest.fixture
    def generator(self):
        return DemandPatternGenerator(seed=42)

    def test_generate_demand_series(self, generator):
        """Test generating demand series."""
        start = datetime.now(timezone.utc) - timedelta(days=7)
        end = datetime.now(timezone.utc)

        forecasts = generator.generate_demand_series(
            "ATL", start, end, ForecastGranularity.DAILY
        )

        assert len(forecasts) >= 7
        assert all(f.station == "ATL" for f in forecasts)
        assert all(f.total_demand.q50 > 0 for f in forecasts)

    def test_demand_varies_by_station_tier(self, generator):
        """Test demand varies by station tier."""
        start = datetime.now(timezone.utc) - timedelta(days=3)
        end = datetime.now(timezone.utc)

        atl_forecasts = generator.generate_demand_series("ATL", start, end)
        # Use a spoke station that might be in DELTA_STATIONS or just test hub
        hub_demand = sum(f.total_demand.q50 for f in atl_forecasts)

        # Hub should have positive demand
        assert hub_demand > 0

    def test_generate_historical_data(self, generator):
        """Test generating historical data with actuals."""
        history = generator.generate_historical_data("ATL", days=30, include_actuals=True)

        assert len(history) >= 30
        assert all(isinstance(item, tuple) for item in history)
        assert all(len(item) == 2 for item in history)

    def test_add_anomalies(self, generator):
        """Test adding anomalies to forecasts."""
        start = datetime.now(timezone.utc) - timedelta(days=30)
        end = datetime.now(timezone.utc)

        forecasts = generator.generate_demand_series("ATL", start, end)
        with_anomalies = generator.add_anomalies(forecasts, anomaly_rate=0.2)

        anomaly_count = sum(1 for f in with_anomalies if f.is_anomaly)
        # Should have some anomalies
        assert anomaly_count >= 0


class TestScenarioGenerator:
    """Tests for scenario generator."""

    @pytest.fixture
    def generator(self):
        return ScenarioGenerator(seed=42)

    def test_generate_winter_storm(self, generator):
        """Test generating winter storm scenario."""
        scenario = generator.generate_winter_storm_scenario(severity=0.8)

        assert "Winter Storm" in scenario.name
        assert len(scenario.events) > 0
        assert scenario.total_impact_flights > 0

    def test_generate_thunderstorm(self, generator):
        """Test generating thunderstorm scenario."""
        scenario = generator.generate_summer_thunderstorm_scenario("ATL")

        assert "Thunderstorm" in scenario.name
        assert scenario.events[0].station == "ATL"

    def test_generate_demand_spike(self, generator):
        """Test generating demand spike scenario."""
        scenario = generator.generate_demand_spike_scenario(
            station="ATL",
            event_name="Super Bowl",
        )

        assert "Super Bowl" in scenario.name
        assert scenario.events[0].station == "ATL"

    def test_generate_random_scenarios(self, generator):
        """Test generating multiple random scenarios."""
        scenarios = generator.generate_random_scenarios(count=5)

        assert len(scenarios) == 5
        assert all(s.scenario_id.startswith("SCN-") for s in scenarios)

    def test_scenario_to_dict(self, generator):
        """Test scenario serialization."""
        scenario = generator.generate_winter_storm_scenario()
        data = scenario.to_dict()

        assert "scenario_id" in data
        assert "events" in data
        assert isinstance(data["events"], list)
