---
name: python-standards
description: Python coding standards for the ULD Forecasting project. Auto-activates when writing Python code.
allowed-tools: Read, Grep, Glob, Edit, Write
---

# Python Standards

## Type Hints

All functions must have complete type annotations:

```python
from datetime import date, datetime
from typing import Optional

def forecast_demand(
    station_id: str,
    target_date: date,
    uld_type: str | None = None,
) -> DemandForecast:
    """Forecast ULD demand for a station."""
    ...
```

## Pydantic Models

Use Pydantic v2 for all data structures:

```python
from pydantic import BaseModel, Field
from enum import Enum

class ULDStatus(str, Enum):
    SERVICEABLE = "serviceable"
    IN_USE = "in_use"
    EMPTY = "empty"
    DAMAGED = "damaged"
    OUT_OF_SERVICE = "out_of_service"

class ULD(BaseModel):
    uld_id: str = Field(..., description="Unique ULD identifier")
    uld_type: str = Field(..., description="IATA ULD type code")
    status: ULDStatus
    station: str = Field(..., pattern=r"^[A-Z]{3}$")
    last_update: datetime

    model_config = {"frozen": True}
```

## Error Handling

Define domain-specific exceptions:

```python
class ULDError(Exception):
    """Base exception for ULD operations."""
    pass

class ULDNotFoundError(ULDError):
    """ULD not found in inventory."""
    def __init__(self, uld_id: str):
        self.uld_id = uld_id
        super().__init__(f"ULD {uld_id} not found")

class InsufficientInventoryError(ULDError):
    """Not enough ULDs at station."""
    def __init__(self, station: str, required: int, available: int):
        self.station = station
        self.required = required
        self.available = available
        super().__init__(
            f"Station {station}: need {required}, have {available}"
        )
```

## Async Patterns

Use async for I/O operations:

```python
async def fetch_station_inventory(
    station_ids: list[str],
) -> dict[str, StationInventory]:
    async with httpx.AsyncClient() as client:
        tasks = [
            client.get(f"/api/stations/{sid}/inventory")
            for sid in station_ids
        ]
        responses = await asyncio.gather(*tasks)
        return {
            sid: StationInventory.model_validate(r.json())
            for sid, r in zip(station_ids, responses)
        }
```

## Project Structure

```
src/
├── api/           # FastAPI endpoints
├── domain/        # Core business entities (Pydantic models)
├── services/      # Business logic
├── ml/            # ML models and pipelines
├── data/          # Data access layer
└── synthetic/     # Synthetic data generation
```

## Testing Pattern

```python
import pytest
from hypothesis import given, strategies as st

class TestDemandForecast:
    def test_forecast_returns_valid_prediction(self):
        result = forecast_demand("ATL", date(2024, 1, 15))
        assert result.predicted_demand >= 0
        assert result.confidence_interval[0] <= result.predicted_demand

    @given(st.integers(min_value=0, max_value=1000))
    def test_demand_never_negative(self, passenger_count: int):
        demand = calculate_uld_demand(passenger_count)
        assert demand >= 0
```
