"""Tests for database layer."""

import pytest
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from src.database.models import (
    Base,
    ULDPositionRecord,
    ULDInventoryRecord,
    ForecastRecord,
)
from src.database.repository import (
    ULDPositionRepository,
    ULDInventoryRepository,
    ForecastRepository,
)


@pytest.fixture
async def async_engine():
    """Create an in-memory SQLite engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def async_session(async_engine):
    """Create an async session for testing."""
    session_factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        yield session


class TestULDPositionRepository:
    """Tests for ULD position repository."""

    @pytest.mark.asyncio
    async def test_add_position(self, async_session):
        """Test adding a position record."""
        repo = ULDPositionRepository(async_session)

        record = ULDPositionRecord(
            uld_id="AKEDL00001",
            uld_type="AKE",
            station="ATL",
            timestamp=datetime.now(timezone.utc),
            position_source="geolocation",
            confidence=0.95,
        )

        result = await repo.add(record)
        await async_session.commit()

        assert result.id is not None
        assert result.uld_id == "AKEDL00001"

    @pytest.mark.asyncio
    async def test_get_latest_position(self, async_session):
        """Test getting latest position for a ULD."""
        repo = ULDPositionRepository(async_session)

        # Add multiple positions
        now = datetime.now(timezone.utc)
        positions = [
            ULDPositionRecord(
                uld_id="AKEDL00001",
                uld_type="AKE",
                station="ATL",
                timestamp=now - timedelta(hours=2),
                position_source="geolocation",
            ),
            ULDPositionRecord(
                uld_id="AKEDL00001",
                uld_type="AKE",
                station="DTW",
                timestamp=now - timedelta(hours=1),
                position_source="flight_event",
            ),
            ULDPositionRecord(
                uld_id="AKEDL00001",
                uld_type="AKE",
                station="JFK",
                timestamp=now,
                position_source="geolocation",
            ),
        ]
        await repo.add_many(positions)
        await async_session.commit()

        latest = await repo.get_latest_position("AKEDL00001")

        assert latest is not None
        assert latest.station == "JFK"

    @pytest.mark.asyncio
    async def test_get_position_history(self, async_session):
        """Test getting position history."""
        repo = ULDPositionRepository(async_session)

        now = datetime.now(timezone.utc)
        start = now - timedelta(days=7)

        # Add positions
        for i in range(5):
            record = ULDPositionRecord(
                uld_id="AKEDL00001",
                uld_type="AKE",
                station="ATL",
                timestamp=start + timedelta(days=i),
                position_source="geolocation",
            )
            await repo.add(record)
        await async_session.commit()

        history = await repo.get_position_history(
            "AKEDL00001",
            start=start,
            end=now,
        )

        assert len(history) == 5

    @pytest.mark.asyncio
    async def test_get_latest_position_not_found(self, async_session):
        """Test getting latest position for unknown ULD."""
        repo = ULDPositionRepository(async_session)

        latest = await repo.get_latest_position("UNKNOWN")

        assert latest is None


class TestULDInventoryRepository:
    """Tests for ULD inventory repository."""

    @pytest.mark.asyncio
    async def test_add_inventory(self, async_session):
        """Test adding an inventory record."""
        repo = ULDInventoryRepository(async_session)

        record = ULDInventoryRecord(
            station="ATL",
            timestamp=datetime.now(timezone.utc),
            inventory={"AKE": 50, "PMC": 20},
            available={"AKE": 30, "PMC": 15},
            in_use={"AKE": 18, "PMC": 4},
            damaged={"AKE": 2, "PMC": 1},
            total_count=70,
            total_available=45,
            availability_ratio=45 / 70,
        )

        result = await repo.add(record)
        await async_session.commit()

        assert result.id is not None
        assert result.station == "ATL"
        assert result.inventory["AKE"] == 50

    @pytest.mark.asyncio
    async def test_get_latest_inventory(self, async_session):
        """Test getting latest inventory for a station."""
        repo = ULDInventoryRepository(async_session)

        now = datetime.now(timezone.utc)

        # Add multiple inventory snapshots
        for i in range(3):
            record = ULDInventoryRecord(
                station="ATL",
                timestamp=now - timedelta(hours=i),
                inventory={"AKE": 50 + i},
                available={"AKE": 30},
                in_use={"AKE": 20 + i},
                damaged={"AKE": 0},
                total_count=50 + i,
                total_available=30,
                availability_ratio=0.6,
            )
            await repo.add(record)
        await async_session.commit()

        latest = await repo.get_latest_inventory("ATL")

        assert latest is not None
        assert latest.inventory["AKE"] == 50  # Most recent (i=0)


class TestForecastRepository:
    """Tests for forecast repository."""

    @pytest.mark.asyncio
    async def test_save_forecasts(self, async_session):
        """Test saving forecast records."""
        repo = ForecastRepository(async_session)

        now = datetime.now(timezone.utc)
        forecasts = [
            {
                "forecast_time": now + timedelta(hours=i),
                "generated_at": now,
                "q05": 10.0 + i,
                "q25": 15.0 + i,
                "q50": 20.0 + i,
                "q75": 25.0 + i,
                "q95": 30.0 + i,
            }
            for i in range(6)
        ]

        results = await repo.save_forecasts(forecasts, "ATL", "demand")
        await async_session.commit()

        assert len(results) == 6
        assert all(r.station == "ATL" for r in results)
        assert all(r.forecast_type == "demand" for r in results)

    @pytest.mark.asyncio
    async def test_get_forecasts(self, async_session):
        """Test getting forecasts for a time range."""
        repo = ForecastRepository(async_session)

        now = datetime.now(timezone.utc)

        # Add forecasts
        for i in range(10):
            record = ForecastRecord(
                station="ATL",
                forecast_type="demand",
                forecast_time=now + timedelta(hours=i),
                generated_at=now,
                q05=10.0,
                q25=15.0,
                q50=20.0,
                q75=25.0,
                q95=30.0,
            )
            await repo.add(record)
        await async_session.commit()

        forecasts = await repo.get_forecasts(
            "ATL",
            "demand",
            start=now,
            end=now + timedelta(hours=5),
        )

        assert len(forecasts) == 6  # hours 0-5 inclusive


class TestDatabaseModels:
    """Tests for database model constraints."""

    @pytest.mark.asyncio
    async def test_uld_position_indices(self, async_session):
        """Test that ULD position indices work correctly."""
        repo = ULDPositionRepository(async_session)

        # Add many records
        now = datetime.now(timezone.utc)
        for i in range(100):
            record = ULDPositionRecord(
                uld_id=f"AKEDL{i:05d}",
                uld_type="AKE",
                station=["ATL", "DTW", "JFK", "LAX"][i % 4],
                timestamp=now - timedelta(hours=i),
                position_source="geolocation",
            )
            await repo.add(record)
        await async_session.commit()

        # Query by station should use index
        atl_positions = await repo.get_station_ulds("ATL")
        assert len(atl_positions) == 25  # Every 4th record

    @pytest.mark.asyncio
    async def test_forecast_record_by_type(self, async_session):
        """Test forecast by_type JSON field."""
        repo = ForecastRepository(async_session)

        record = ForecastRecord(
            station="ATL",
            forecast_type="demand",
            forecast_time=datetime.now(timezone.utc),
            generated_at=datetime.now(timezone.utc),
            q05=10.0,
            q25=15.0,
            q50=20.0,
            q75=25.0,
            q95=30.0,
            by_type={
                "AKE": {"q05": 5, "q50": 10, "q95": 15},
                "PMC": {"q05": 3, "q50": 6, "q95": 9},
            },
        )

        result = await repo.add(record)
        await async_session.commit()

        # Refresh to get from DB
        await async_session.refresh(result)

        assert result.by_type["AKE"]["q50"] == 10
        assert result.by_type["PMC"]["q95"] == 9
