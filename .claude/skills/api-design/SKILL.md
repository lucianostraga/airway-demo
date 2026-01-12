---
name: api-design
description: API design patterns for FastAPI endpoints. Activates when designing or implementing REST APIs.
allowed-tools: Read, Grep, Glob, Edit, Write
---

# API Design Patterns

## Endpoint Structure

```python
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Annotated

router = APIRouter(prefix="/api/v1", tags=["forecasting"])

@router.get(
    "/stations/{station_id}/forecast",
    response_model=ForecastResponse,
    summary="Get demand forecast for a station",
    responses={
        404: {"description": "Station not found"},
        422: {"description": "Invalid parameters"},
    },
)
async def get_station_forecast(
    station_id: Annotated[str, Path(pattern=r"^[A-Z]{3}$")],
    horizon_days: Annotated[int, Query(ge=1, le=30)] = 7,
    uld_type: str | None = None,
    service: ForecastService = Depends(get_forecast_service),
) -> ForecastResponse:
    """
    Retrieve ULD demand forecast for a specific station.

    - **station_id**: 3-letter IATA airport code
    - **horizon_days**: Number of days to forecast (1-30)
    - **uld_type**: Optional filter by ULD type
    """
    try:
        forecast = await service.get_forecast(station_id, horizon_days, uld_type)
        return ForecastResponse.from_domain(forecast)
    except StationNotFoundError:
        raise HTTPException(status_code=404, detail=f"Station {station_id} not found")
```

## Request/Response Models

```python
from pydantic import BaseModel, Field
from datetime import date

class ForecastRequest(BaseModel):
    station_id: str = Field(..., pattern=r"^[A-Z]{3}$")
    target_date: date
    uld_types: list[str] | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "station_id": "ATL",
                    "target_date": "2024-03-15",
                    "uld_types": ["AKE", "PMC"],
                }
            ]
        }
    }

class ForecastItem(BaseModel):
    date: date
    uld_type: str
    predicted_demand: int = Field(..., ge=0)
    confidence_lower: int = Field(..., ge=0)
    confidence_upper: int = Field(..., ge=0)

class ForecastResponse(BaseModel):
    station_id: str
    generated_at: datetime
    horizon_days: int
    forecasts: list[ForecastItem]

    @classmethod
    def from_domain(cls, forecast: DomainForecast) -> "ForecastResponse":
        return cls(
            station_id=forecast.station_id,
            generated_at=datetime.utcnow(),
            horizon_days=len(forecast.predictions),
            forecasts=[
                ForecastItem(
                    date=p.date,
                    uld_type=p.uld_type,
                    predicted_demand=p.demand,
                    confidence_lower=p.ci_lower,
                    confidence_upper=p.ci_upper,
                )
                for p in forecast.predictions
            ],
        )
```

## Error Responses

```python
from fastapi import Request
from fastapi.responses import JSONResponse

class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str | None = None

@app.exception_handler(ULDError)
async def uld_error_handler(request: Request, exc: ULDError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            detail=str(exc),
            request_id=request.state.request_id,
        ).model_dump(),
    )
```

## API Versioning

```python
from fastapi import FastAPI

app = FastAPI(title="ULD Forecasting API", version="1.0.0")

# Mount versioned routers
app.include_router(v1_router, prefix="/api/v1")
app.include_router(v2_router, prefix="/api/v2")  # Future version
```

## Standard Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/stations` | List all stations |
| GET | `/stations/{id}` | Get station details |
| GET | `/stations/{id}/inventory` | Current ULD inventory |
| GET | `/stations/{id}/forecast` | Demand forecast |
| GET | `/forecasts/network` | Network-wide forecast |
| POST | `/recommendations/repositioning` | Generate repositioning plan |
| POST | `/simulations` | Run what-if simulation |
