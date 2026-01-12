"""
Repository pattern for database access.

Provides clean abstraction over SQLAlchemy queries with async support.
"""

from datetime import datetime, timedelta, timezone
from typing import Generic, TypeVar, Optional, List

from sqlalchemy import select, delete, update, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    Base,
    ULDPositionRecord,
    ULDInventoryRecord,
    ForecastRecord,
    RecommendationRecord,
    FlightRecord,
    OptimizationRunRecord,
)
from src.domain import ULDType, ULDPosition, ULDInventory


T = TypeVar("T", bound=Base)


class DatabaseRepository(Generic[T]):
    """
    Generic repository with common CRUD operations.
    """

    def __init__(self, session: AsyncSession, model_class: type[T]):
        self.session = session
        self.model_class = model_class

    async def get_by_id(self, id: int) -> Optional[T]:
        """Get a record by ID."""
        result = await self.session.execute(
            select(self.model_class).where(self.model_class.id == id)
        )
        return result.scalar_one_or_none()

    async def get_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """Get all records with pagination."""
        result = await self.session.execute(
            select(self.model_class).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def add(self, entity: T) -> T:
        """Add a new record."""
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def add_many(self, entities: List[T]) -> List[T]:
        """Add multiple records."""
        self.session.add_all(entities)
        await self.session.flush()
        return entities

    async def delete(self, entity: T) -> None:
        """Delete a record."""
        await self.session.delete(entity)

    async def delete_by_id(self, id: int) -> bool:
        """Delete a record by ID."""
        result = await self.session.execute(
            delete(self.model_class).where(self.model_class.id == id)
        )
        return result.rowcount > 0


class ULDPositionRepository(DatabaseRepository[ULDPositionRecord]):
    """
    Repository for ULD position records.
    """

    def __init__(self, session: AsyncSession):
        super().__init__(session, ULDPositionRecord)

    async def get_latest_position(self, uld_id: str) -> Optional[ULDPositionRecord]:
        """Get the most recent position for a ULD."""
        result = await self.session.execute(
            select(ULDPositionRecord)
            .where(ULDPositionRecord.uld_id == uld_id)
            .order_by(ULDPositionRecord.timestamp.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_position_history(
        self,
        uld_id: str,
        start: datetime,
        end: datetime,
    ) -> List[ULDPositionRecord]:
        """Get position history for a ULD in time range."""
        result = await self.session.execute(
            select(ULDPositionRecord)
            .where(
                and_(
                    ULDPositionRecord.uld_id == uld_id,
                    ULDPositionRecord.timestamp >= start,
                    ULDPositionRecord.timestamp <= end,
                )
            )
            .order_by(ULDPositionRecord.timestamp)
        )
        return list(result.scalars().all())

    async def get_station_ulds(
        self,
        station: str,
        as_of: Optional[datetime] = None,
    ) -> List[ULDPositionRecord]:
        """Get all ULDs currently at a station."""
        # Get latest position for each ULD
        subquery = (
            select(
                ULDPositionRecord.uld_id,
                func.max(ULDPositionRecord.timestamp).label("max_timestamp"),
            )
            .group_by(ULDPositionRecord.uld_id)
            .subquery()
        )

        query = (
            select(ULDPositionRecord)
            .join(
                subquery,
                and_(
                    ULDPositionRecord.uld_id == subquery.c.uld_id,
                    ULDPositionRecord.timestamp == subquery.c.max_timestamp,
                ),
            )
            .where(ULDPositionRecord.station == station)
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def save_position(self, position: ULDPosition) -> ULDPositionRecord:
        """Save a position record from domain model."""
        record = ULDPositionRecord(
            uld_id=position.uld_id,
            uld_type=position.uld_type.value if hasattr(position.uld_type, 'value') else position.uld_type,
            station=position.station,
            timestamp=position.timestamp,
            position_source=position.position_source,
            flight_number=position.flight_number,
            latitude=position.latitude,
            longitude=position.longitude,
            location_type=position.location_type,
            confidence=position.confidence,
        )
        return await self.add(record)

    async def delete_old_records(self, days: int = 90) -> int:
        """Delete position records older than specified days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        result = await self.session.execute(
            delete(ULDPositionRecord).where(ULDPositionRecord.timestamp < cutoff)
        )
        return result.rowcount


class ULDInventoryRepository(DatabaseRepository[ULDInventoryRecord]):
    """
    Repository for ULD inventory snapshots.
    """

    def __init__(self, session: AsyncSession):
        super().__init__(session, ULDInventoryRecord)

    async def get_latest_inventory(self, station: str) -> Optional[ULDInventoryRecord]:
        """Get the most recent inventory snapshot for a station."""
        result = await self.session.execute(
            select(ULDInventoryRecord)
            .where(ULDInventoryRecord.station == station)
            .order_by(ULDInventoryRecord.timestamp.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_inventory_history(
        self,
        station: str,
        start: datetime,
        end: datetime,
    ) -> List[ULDInventoryRecord]:
        """Get inventory history for a station."""
        result = await self.session.execute(
            select(ULDInventoryRecord)
            .where(
                and_(
                    ULDInventoryRecord.station == station,
                    ULDInventoryRecord.timestamp >= start,
                    ULDInventoryRecord.timestamp <= end,
                )
            )
            .order_by(ULDInventoryRecord.timestamp)
        )
        return list(result.scalars().all())

    async def save_inventory(self, inventory: ULDInventory) -> ULDInventoryRecord:
        """Save an inventory snapshot from domain model."""
        record = ULDInventoryRecord(
            station=inventory.station,
            timestamp=inventory.timestamp,
            inventory={k.value if hasattr(k, 'value') else k: v for k, v in inventory.inventory.items()},
            available={k.value if hasattr(k, 'value') else k: v for k, v in inventory.available.items()},
            in_use={k.value if hasattr(k, 'value') else k: v for k, v in inventory.in_use.items()},
            damaged={k.value if hasattr(k, 'value') else k: v for k, v in inventory.damaged.items()},
            total_count=inventory.total_count(),
            total_available=inventory.total_available(),
            availability_ratio=inventory.availability_ratio(),
        )
        return await self.add(record)


class ForecastRepository(DatabaseRepository[ForecastRecord]):
    """
    Repository for forecast records.
    """

    def __init__(self, session: AsyncSession):
        super().__init__(session, ForecastRecord)

    async def get_forecasts(
        self,
        station: str,
        forecast_type: str,
        start: datetime,
        end: datetime,
    ) -> List[ForecastRecord]:
        """Get forecasts for a station in time range."""
        result = await self.session.execute(
            select(ForecastRecord)
            .where(
                and_(
                    ForecastRecord.station == station,
                    ForecastRecord.forecast_type == forecast_type,
                    ForecastRecord.forecast_time >= start,
                    ForecastRecord.forecast_time <= end,
                )
            )
            .order_by(ForecastRecord.forecast_time)
        )
        return list(result.scalars().all())

    async def get_latest_forecasts(
        self,
        station: str,
        forecast_type: str,
        hours_ahead: int = 24,
    ) -> List[ForecastRecord]:
        """Get the most recent forecast series for a station."""
        # Find the latest generation time
        subquery = (
            select(func.max(ForecastRecord.generated_at))
            .where(
                and_(
                    ForecastRecord.station == station,
                    ForecastRecord.forecast_type == forecast_type,
                )
            )
            .scalar_subquery()
        )

        result = await self.session.execute(
            select(ForecastRecord)
            .where(
                and_(
                    ForecastRecord.station == station,
                    ForecastRecord.forecast_type == forecast_type,
                    ForecastRecord.generated_at == subquery,
                )
            )
            .order_by(ForecastRecord.forecast_time)
            .limit(hours_ahead)
        )
        return list(result.scalars().all())

    async def save_forecasts(
        self,
        forecasts: List[dict],
        station: str,
        forecast_type: str,
    ) -> List[ForecastRecord]:
        """Save a batch of forecasts."""
        records = []
        for f in forecasts:
            record = ForecastRecord(
                station=station,
                forecast_type=forecast_type,
                forecast_time=f["forecast_time"],
                generated_at=f["generated_at"],
                granularity=f.get("granularity", "hourly"),
                q05=f["q05"],
                q25=f["q25"],
                q50=f["q50"],
                q75=f["q75"],
                q95=f["q95"],
                by_type=f.get("by_type"),
                confidence=f.get("confidence", "medium"),
                is_anomaly=f.get("is_anomaly", False),
                model_version=f.get("model_version"),
            )
            records.append(record)
        return await self.add_many(records)


class RecommendationRepository(DatabaseRepository[RecommendationRecord]):
    """
    Repository for repositioning recommendations.
    """

    def __init__(self, session: AsyncSession):
        super().__init__(session, RecommendationRecord)

    async def get_pending_recommendations(
        self,
        station: Optional[str] = None,
    ) -> List[RecommendationRecord]:
        """Get pending recommendations, optionally filtered by station."""
        query = select(RecommendationRecord).where(
            RecommendationRecord.status == "pending"
        )

        if station:
            query = query.where(
                or_(
                    RecommendationRecord.origin_station == station,
                    RecommendationRecord.destination_station == station,
                )
            )

        query = query.order_by(
            RecommendationRecord.priority,
            RecommendationRecord.required_by,
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def update_status(
        self,
        recommendation_id: str,
        status: str,
        executed_at: Optional[datetime] = None,
    ) -> bool:
        """Update recommendation status."""
        values = {"status": status}
        if executed_at:
            values["executed_at"] = executed_at

        result = await self.session.execute(
            update(RecommendationRecord)
            .where(RecommendationRecord.recommendation_id == recommendation_id)
            .values(**values)
        )
        return result.rowcount > 0


class FlightRepository(DatabaseRepository[FlightRecord]):
    """
    Repository for flight records.
    """

    def __init__(self, session: AsyncSession):
        super().__init__(session, FlightRecord)

    async def get_flights_by_route(
        self,
        origin: str,
        destination: str,
        start: datetime,
        end: datetime,
    ) -> List[FlightRecord]:
        """Get flights for a route in time range."""
        result = await self.session.execute(
            select(FlightRecord)
            .where(
                and_(
                    FlightRecord.origin == origin,
                    FlightRecord.destination == destination,
                    FlightRecord.scheduled_departure >= start,
                    FlightRecord.scheduled_departure <= end,
                )
            )
            .order_by(FlightRecord.scheduled_departure)
        )
        return list(result.scalars().all())

    async def get_station_departures(
        self,
        station: str,
        start: datetime,
        end: datetime,
    ) -> List[FlightRecord]:
        """Get departures from a station."""
        result = await self.session.execute(
            select(FlightRecord)
            .where(
                and_(
                    FlightRecord.origin == station,
                    FlightRecord.scheduled_departure >= start,
                    FlightRecord.scheduled_departure <= end,
                )
            )
            .order_by(FlightRecord.scheduled_departure)
        )
        return list(result.scalars().all())

    async def get_station_arrivals(
        self,
        station: str,
        start: datetime,
        end: datetime,
    ) -> List[FlightRecord]:
        """Get arrivals at a station."""
        result = await self.session.execute(
            select(FlightRecord)
            .where(
                and_(
                    FlightRecord.destination == station,
                    FlightRecord.scheduled_arrival >= start,
                    FlightRecord.scheduled_arrival <= end,
                )
            )
            .order_by(FlightRecord.scheduled_arrival)
        )
        return list(result.scalars().all())


class OptimizationRepository(DatabaseRepository[OptimizationRunRecord]):
    """
    Repository for optimization run records.
    """

    def __init__(self, session: AsyncSession):
        super().__init__(session, OptimizationRunRecord)

    async def get_latest_run(self) -> Optional[OptimizationRunRecord]:
        """Get the most recent optimization run."""
        result = await self.session.execute(
            select(OptimizationRunRecord)
            .order_by(OptimizationRunRecord.generated_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_run_by_id(self, run_id: str) -> Optional[OptimizationRunRecord]:
        """Get an optimization run by UUID."""
        result = await self.session.execute(
            select(OptimizationRunRecord)
            .where(OptimizationRunRecord.run_id == run_id)
        )
        return result.scalar_one_or_none()
