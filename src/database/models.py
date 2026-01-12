"""
SQLAlchemy database models for ULD Forecasting System.

Provides persistent storage for:
- ULD positions and movement history
- Station inventory snapshots
- Forecast results
- Recommendations
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Boolean,
    JSON,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class ULDPositionRecord(Base):
    """
    ULD position tracking record.

    Stores all position updates from geolocation pings and flight events.
    """
    __tablename__ = "uld_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uld_id = Column(String(20), nullable=False, index=True)
    uld_type = Column(String(10), nullable=False)
    station = Column(String(3), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    position_source = Column(String(20), default="geolocation")
    flight_number = Column(String(10), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    location_type = Column(String(20), default="ground")
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_uld_positions_uld_timestamp", "uld_id", "timestamp"),
        Index("ix_uld_positions_station_timestamp", "station", "timestamp"),
    )


class ULDInventoryRecord(Base):
    """
    Station inventory snapshot.

    Captures ULD inventory state at a point in time.
    """
    __tablename__ = "uld_inventories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    station = Column(String(3), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Inventory counts by type (stored as JSON)
    inventory = Column(JSON, nullable=False)  # {type: count}
    available = Column(JSON, nullable=False)  # {type: count}
    in_use = Column(JSON, nullable=False)     # {type: count}
    damaged = Column(JSON, nullable=False)    # {type: count}

    total_count = Column(Integer, nullable=False)
    total_available = Column(Integer, nullable=False)
    availability_ratio = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_inventory_station_timestamp", "station", "timestamp"),
    )


class ForecastRecord(Base):
    """
    Forecast result record.

    Stores demand, supply, and imbalance forecasts with uncertainty.
    """
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    station = Column(String(3), nullable=False, index=True)
    forecast_type = Column(String(20), nullable=False)  # demand, supply, imbalance
    forecast_time = Column(DateTime, nullable=False, index=True)
    generated_at = Column(DateTime, nullable=False)
    granularity = Column(String(10), default="hourly")

    # Quantile forecasts
    q05 = Column(Float, nullable=False)
    q25 = Column(Float, nullable=False)
    q50 = Column(Float, nullable=False)
    q75 = Column(Float, nullable=False)
    q95 = Column(Float, nullable=False)

    # Breakdown by type (stored as JSON)
    by_type = Column(JSON, nullable=True)

    # Metadata
    confidence = Column(String(10), default="medium")
    is_anomaly = Column(Boolean, default=False)
    model_version = Column(String(20), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_forecast_station_time", "station", "forecast_time"),
        Index("ix_forecast_type_time", "forecast_type", "forecast_time"),
    )


class RecommendationRecord(Base):
    """
    Repositioning recommendation record.

    Stores generated recommendations for ULD movements.
    """
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    recommendation_id = Column(String(20), unique=True, nullable=False)
    priority = Column(String(10), nullable=False)
    uld_type = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    origin_station = Column(String(3), nullable=False, index=True)
    destination_station = Column(String(3), nullable=False, index=True)
    transport_method = Column(String(20), nullable=False)
    recommended_departure = Column(DateTime, nullable=False)
    required_by = Column(DateTime, nullable=False)
    reason = Column(String(500), nullable=True)
    shortage_probability = Column(Float, nullable=False)

    # Cost-benefit analysis (stored as JSON)
    cost_benefit = Column(JSON, nullable=False)

    # Status tracking
    status = Column(String(20), default="pending")  # pending, approved, executed, cancelled
    executed_at = Column(DateTime, nullable=True)

    generated_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_rec_origin_dest", "origin_station", "destination_station"),
        Index("ix_rec_status_priority", "status", "priority"),
    )


class FlightRecord(Base):
    """
    Flight schedule record.

    Stores flight information for tracking and forecasting.
    """
    __tablename__ = "flights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    flight_number = Column(String(10), nullable=False, index=True)
    airline = Column(String(3), nullable=False)
    origin = Column(String(3), nullable=False, index=True)
    destination = Column(String(3), nullable=False, index=True)
    scheduled_departure = Column(DateTime, nullable=False, index=True)
    scheduled_arrival = Column(DateTime, nullable=False)
    actual_departure = Column(DateTime, nullable=True)
    actual_arrival = Column(DateTime, nullable=True)
    status = Column(String(20), nullable=False)
    aircraft_type = Column(String(10), nullable=True)
    uld_capacity = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_flight_route", "origin", "destination"),
        Index("ix_flight_departure", "scheduled_departure"),
    )


class OptimizationRunRecord(Base):
    """
    Network optimization run record.

    Stores results of optimization runs.
    """
    __tablename__ = "optimization_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), unique=True, nullable=False)  # UUID
    generated_at = Column(DateTime, nullable=False)
    solver_status = Column(String(20), nullable=False)
    total_moves = Column(Integer, nullable=False)
    total_ulds_moved = Column(Integer, nullable=False)
    total_cost = Column(Float, nullable=False)
    solve_time_seconds = Column(Float, nullable=False)

    # Configuration
    optimization_horizon_hours = Column(Integer, nullable=False)
    max_moves_per_station = Column(Integer, nullable=False)

    # Results stored as JSON (list of moves)
    moves = Column(JSON, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
