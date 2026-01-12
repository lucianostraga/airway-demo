"""
Database session management.

Provides async SQLAlchemy session handling with connection pooling.
Supports SQLite (development) and PostgreSQL (production).
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.pool import NullPool, StaticPool

from .models import Base


# Global engine instance
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_database_url() -> str:
    """
    Get database URL from environment or use default SQLite.

    Supports:
    - SQLite: sqlite+aiosqlite:///path/to/db.sqlite
    - PostgreSQL: postgresql+asyncpg://user:pass@host:port/dbname
    """
    db_url = os.getenv("DATABASE_URL")

    if db_url:
        # Convert postgres:// to postgresql+asyncpg://
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return db_url

    # Default: SQLite in project root
    return "sqlite+aiosqlite:///./uld_forecasting.db"


def get_engine() -> AsyncEngine:
    """Get or create the async database engine."""
    global _engine

    if _engine is None:
        db_url = get_database_url()

        # Configure engine based on database type
        if "sqlite" in db_url:
            # SQLite: use StaticPool for in-memory, NullPool for file
            if ":memory:" in db_url:
                _engine = create_async_engine(
                    db_url,
                    echo=False,
                    poolclass=StaticPool,
                    connect_args={"check_same_thread": False},
                )
            else:
                _engine = create_async_engine(
                    db_url,
                    echo=False,
                    connect_args={"check_same_thread": False},
                )
        else:
            # PostgreSQL: use connection pooling
            _engine = create_async_engine(
                db_url,
                echo=False,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
            )

    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the session factory."""
    global _session_factory

    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

    return _session_factory


async def init_db() -> None:
    """
    Initialize the database (create all tables).

    Should be called once at application startup.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """
    Close database connections.

    Should be called at application shutdown.
    """
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an async database session.

    Usage:
        async with get_session() as session:
            result = await session.execute(query)
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.

    Usage:
        @app.get("/")
        async def endpoint(session: AsyncSession = Depends(get_session_dependency)):
            ...
    """
    async with get_session() as session:
        yield session
