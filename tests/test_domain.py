"""Tests for domain models."""

import pytest
from datetime import datetime, timedelta, timezone

from src.domain import (
    ULD,
    ULDType,
    ULDStatus,
    ULDPosition,
    ULDInventory,
    Station,
    StationTier,
    Flight,
    FlightStatus,
    FlightSchedule,
    Route,
    DemandForecast,
    QuantileForecast,
    ForecastGranularity,
    ForecastConfidence,
    RepositioningRecommendation,
    CostBenefit,
    RecommendationPriority,
    DELTA_STATIONS,
    AIRCRAFT_TYPES,
)


class TestULDModels:
    """Tests for ULD domain models."""

    def test_uld_creation(self, sample_uld):
        """Test ULD can be created with valid data."""
        assert sample_uld.uld_id == "AKEDL00001"
        assert sample_uld.uld_type == ULDType.AKE
        assert sample_uld.status == ULDStatus.SERVICEABLE
        assert sample_uld.current_station == "ATL"

    def test_uld_types(self):
        """Test all ULD types are defined."""
        assert ULDType.AKE.value == "AKE"
        assert ULDType.PMC.value == "PMC"
        assert len(ULDType) >= 5

    def test_uld_statuses(self):
        """Test all ULD statuses are defined."""
        assert ULDStatus.SERVICEABLE.value == "serviceable"
        assert ULDStatus.IN_USE.value == "in_use"
        assert ULDStatus.DAMAGED.value == "damaged"

    def test_uld_position(self):
        """Test ULDPosition model."""
        position = ULDPosition(
            uld_id="AKEDL00001",
            uld_type=ULDType.AKE,
            station="ATL",
            timestamp=datetime.now(timezone.utc),
            position_source="geolocation",
            confidence=0.95,
        )
        assert position.station == "ATL"
        assert position.confidence == 0.95

    def test_uld_inventory(self, sample_inventory):
        """Test ULDInventory model."""
        assert sample_inventory.station == "ATL"
        assert sample_inventory.total_count() == 85
        assert sample_inventory.total_available() == 55
        assert sample_inventory.availability_ratio() == pytest.approx(55 / 85, rel=0.01)


class TestStationModels:
    """Tests for station domain models."""

    def test_delta_stations_defined(self):
        """Test Delta stations are defined."""
        assert "ATL" in DELTA_STATIONS
        assert "DTW" in DELTA_STATIONS
        assert "MSP" in DELTA_STATIONS
        assert "SLC" in DELTA_STATIONS

    def test_station_tiers(self):
        """Test station tier classification."""
        atl = DELTA_STATIONS["ATL"]
        assert atl.tier == StationTier.HUB

        jfk = DELTA_STATIONS["JFK"]
        assert jfk.tier == StationTier.FOCUS_CITY

    def test_hub_count(self):
        """Test we have the expected number of hubs."""
        hubs = [s for s, info in DELTA_STATIONS.items() if info.tier == StationTier.HUB]
        assert len(hubs) == 4  # ATL, DTW, MSP, SLC


class TestFlightModels:
    """Tests for flight domain models."""

    def test_flight_creation(self, sample_flight):
        """Test Flight can be created."""
        assert sample_flight.flight_number == "DL123"
        assert sample_flight.origin == "ATL"
        assert sample_flight.destination == "JFK"

    def test_flight_is_widebody(self, sample_flight):
        """Test widebody detection."""
        assert sample_flight.is_widebody is True  # B763 is widebody

    def test_flight_route(self, sample_flight):
        """Test route property."""
        route = sample_flight.route
        assert route.origin == "ATL"
        assert route.destination == "JFK"
        assert route.route_key == "ATL-JFK"

    def test_aircraft_types_defined(self):
        """Test aircraft types are defined."""
        assert "B738" in AIRCRAFT_TYPES
        assert "B763" in AIRCRAFT_TYPES
        assert "A359" in AIRCRAFT_TYPES

    def test_aircraft_uld_capacity(self):
        """Test aircraft ULD capacity."""
        b763 = AIRCRAFT_TYPES["B763"]
        assert b763.is_widebody is True
        assert b763.lower_deck_positions > 0

        b738 = AIRCRAFT_TYPES["B738"]
        assert b738.is_widebody is False
        assert b738.lower_deck_positions == 0

    def test_flight_schedule(self, sample_flight):
        """Test FlightSchedule model."""
        now = datetime.now(timezone.utc)
        schedule = FlightSchedule(
            flights=[sample_flight],
            period_start=now,
            period_end=now + timedelta(days=1),
        )
        assert len(schedule.flights) == 1
        assert len(schedule.widebody_flights()) == 1


class TestForecastModels:
    """Tests for forecast domain models."""

    def test_quantile_forecast(self):
        """Test QuantileForecast model."""
        qf = QuantileForecast(
            q05=10, q25=20, q50=30, q75=40, q95=50
        )
        assert qf.point_estimate == 30
        assert qf.uncertainty_range == 40  # 50 - 10
        assert qf.iqr == 20  # 40 - 20

    def test_demand_forecast(self):
        """Test DemandForecast model."""
        now = datetime.now(timezone.utc)
        forecast = DemandForecast(
            station="ATL",
            forecast_time=now,
            generated_at=now,
            granularity=ForecastGranularity.HOURLY,
            demand_by_type={
                ULDType.AKE: QuantileForecast(q05=10, q25=15, q50=20, q75=25, q95=30)
            },
            total_demand=QuantileForecast(q05=20, q25=30, q50=40, q75=50, q95=60),
            confidence=ForecastConfidence.MEDIUM,
        )
        assert forecast.station == "ATL"
        assert forecast.total_point_estimate == 40


class TestRecommendationModels:
    """Tests for recommendation domain models."""

    def test_cost_benefit(self):
        """Test CostBenefit model."""
        cb = CostBenefit(
            transportation_cost=100,
            handling_cost=50,
            avoided_shortage_cost=300,
            revenue_protected=100,
        )
        assert cb.total_cost == 150
        assert cb.total_benefit == 400
        assert cb.net_benefit == 250
        assert cb.roi == pytest.approx(250 / 150, rel=0.01)

    def test_repositioning_recommendation(self):
        """Test RepositioningRecommendation model."""
        now = datetime.now(timezone.utc)
        rec = RepositioningRecommendation(
            recommendation_id="REC-001",
            priority=RecommendationPriority.HIGH,
            uld_type=ULDType.AKE,
            quantity=10,
            origin_station="DTW",
            destination_station="ATL",
            transport_method="flight",
            recommended_departure=now,
            required_by=now + timedelta(hours=12),
            estimated_duration_hours=2.0,
            reason="Shortage at ATL",
            shortage_probability_at_dest=0.7,
            cost_benefit=CostBenefit(
                transportation_cost=100,
                handling_cost=50,
                avoided_shortage_cost=500,
            ),
        )
        assert rec.is_economic is True
        assert rec.quantity == 10
