"""Pytest fixtures for ULD Forecasting tests."""

import pytest
from datetime import datetime, timezone, timedelta

from src.domain import (
    ULD,
    ULDType,
    ULDStatus,
    ULDInventory,
    Station,
    StationTier,
    Flight,
    FlightStatus,
    DELTA_STATIONS,
)
from src.services import (
    ULDTrackingService,
    ForecastingService,
    RecommendationService,
    NetworkOptimizer,
)
from src.services.tracking import InMemoryPositionRepository


@pytest.fixture
def sample_uld() -> ULD:
    """Create a sample ULD for testing."""
    return ULD(
        uld_id="AKEDL00001",
        uld_type=ULDType.AKE,
        status=ULDStatus.SERVICEABLE,
        current_station="ATL",
        owner_airline="DL",
        is_leased=False,
    )


@pytest.fixture
def sample_inventory() -> ULDInventory:
    """Create a sample inventory for testing."""
    return ULDInventory(
        station="ATL",
        timestamp=datetime.now(timezone.utc),
        inventory={
            ULDType.AKE: 50,
            ULDType.AKH: 15,
            ULDType.PMC: 20,
        },
        available={
            ULDType.AKE: 30,
            ULDType.AKH: 10,
            ULDType.PMC: 15,
        },
        in_use={
            ULDType.AKE: 18,
            ULDType.AKH: 4,
            ULDType.PMC: 4,
        },
        damaged={
            ULDType.AKE: 2,
            ULDType.AKH: 1,
            ULDType.PMC: 1,
        },
    )


@pytest.fixture
def sample_flight() -> Flight:
    """Create a sample flight for testing."""
    now = datetime.now(timezone.utc)
    return Flight(
        flight_number="DL123",
        airline="DL",
        origin="ATL",
        destination="JFK",
        scheduled_departure=now,
        scheduled_arrival=now + timedelta(hours=2),
        status=FlightStatus.SCHEDULED,
        aircraft_type="B763",
        booked_passengers=200,
        uld_capacity=8,
    )


@pytest.fixture
def position_repository() -> InMemoryPositionRepository:
    """Create an in-memory repository for testing."""
    return InMemoryPositionRepository()


@pytest.fixture
def tracking_service(position_repository) -> ULDTrackingService:
    """Create a tracking service for testing."""
    return ULDTrackingService(position_repository)


@pytest.fixture
def forecasting_service() -> ForecastingService:
    """Create a forecasting service for testing."""
    return ForecastingService(seed=42)


@pytest.fixture
def recommendation_service() -> RecommendationService:
    """Create a recommendation service for testing."""
    return RecommendationService()


@pytest.fixture
def optimizer() -> NetworkOptimizer:
    """Create a network optimizer for testing."""
    return NetworkOptimizer()
