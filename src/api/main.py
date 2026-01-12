"""
FastAPI main application.
"""

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .routers import tracking, forecasting, recommendations, stations

# Static files directory
STATIC_DIR = Path(__file__).parent.parent.parent / "static"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="ULD Forecasting API",
        description="""
        Delta Airlines ULD Forecasting & Allocation System.

        Provides real-time forecasting and optimization for Unit Load Device (ULD)
        management across the Delta network.

        ## Features

        - **Tracking**: Real-time ULD position tracking
        - **Forecasting**: Demand and supply predictions with uncertainty quantification
        - **Recommendations**: Actionable repositioning suggestions
        - **Optimization**: Network-wide ULD allocation optimization
        """,
        version="1.0.0",
    )

    # CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    application.include_router(
        tracking.router,
        prefix="/api/v1/tracking",
        tags=["Tracking"],
    )
    application.include_router(
        forecasting.router,
        prefix="/api/v1/forecasting",
        tags=["Forecasting"],
    )
    application.include_router(
        recommendations.router,
        prefix="/api/v1/recommendations",
        tags=["Recommendations"],
    )
    application.include_router(
        stations.router,
        prefix="/api/v1/stations",
        tags=["Stations"],
    )

    @application.get("/", tags=["Health"])
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "ULD Forecasting API",
            "version": "1.0.0",
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @application.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": {
                "tracking": "available",
                "forecasting": "available",
                "recommendations": "available",
                "optimization": "available",
            },
        }

    @application.get("/dashboard", tags=["Dashboard"], include_in_schema=False)
    async def dashboard():
        """Serve the dashboard UI."""
        return FileResponse(STATIC_DIR / "index.html")

    # Mount static files if directory exists
    if STATIC_DIR.exists():
        application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return application


# Create default app instance
app = create_app()
