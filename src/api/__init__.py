"""
FastAPI application for ULD Forecasting System.

Provides REST endpoints for:
- ULD tracking and inventory
- Demand/supply forecasting
- Repositioning recommendations
- Network optimization
"""

from .main import app, create_app

__all__ = ["app", "create_app"]
