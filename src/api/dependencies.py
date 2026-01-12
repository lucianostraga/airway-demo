"""
API dependencies.

Provides dependency injection for FastAPI routes.
"""

from src.services import (
    ULDTrackingService,
    ForecastingService,
    RecommendationService,
    NetworkOptimizer,
)
from src.services.tracking import InMemoryPositionRepository


class AppState:
    """Application state container."""

    _instance: "AppState | None" = None

    def __init__(self):
        self.position_repository = InMemoryPositionRepository()
        self.tracking_service = ULDTrackingService(self.position_repository)
        self.forecasting_service = ForecastingService(seed=42)
        self.recommendation_service = RecommendationService()
        self.optimizer = NetworkOptimizer()

    @classmethod
    def get_instance(cls) -> "AppState":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = AppState()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None


def get_tracking_service() -> ULDTrackingService:
    """Dependency for tracking service."""
    return AppState.get_instance().tracking_service


def get_forecasting_service() -> ForecastingService:
    """Dependency for forecasting service."""
    return AppState.get_instance().forecasting_service


def get_recommendation_service() -> RecommendationService:
    """Dependency for recommendation service."""
    return AppState.get_instance().recommendation_service


def get_optimizer() -> NetworkOptimizer:
    """Dependency for optimizer."""
    return AppState.get_instance().optimizer
