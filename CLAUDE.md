# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ULD Forecasting & Allocation System** for Delta Airlines - a system to optimize Unit Load Device (ULD) utilization through forecasting and allocation recommendations.

**Tech Stack**: Python 3.11+, FastAPI, scikit-learn, XGBoost, Prophet, Pandas, OR-Tools

## Domain Context

ULDs (Unit Load Devices) are standardized containers/pallets used to load luggage, cargo, and mail on aircraft. This system aims to:
- Forecast ULD locations using geolocation (daily updates), flight schedules, and luggage demand
- Recommend ULD allocation strategies including empty repositioning (deadheading)
- Prevent shortages that cause flight delays

## System Components

1. **Tracking Module**: Ingest daily geolocation data, maintain ULD inventory state
2. **Forecasting Engine**: ML-based demand prediction using historical + external signals
3. **Recommendation Engine**: Optimization for allocation and repositioning
4. **Simulation Module**: What-if scenarios with configurable parameters
5. **API Layer**: FastAPI endpoints for all operations

## Skills

Skills provide auto-activated knowledge and standards. Located in `.claude/skills/`.

| Skill | Purpose | Auto-activates When |
|-------|---------|---------------------|
| `uld-domain` | ULD types, statuses, metrics, terminology | Discussing ULDs, containers, airline ops |
| `python-standards` | Type hints, Pydantic, error handling, testing | Writing Python code |
| `ml-patterns` | Forecasting pipelines, features, evaluation | Working on ML models |
| `api-design` | FastAPI patterns, request/response models | Designing/implementing APIs |
| `decision-log` | ADR format, decision documentation | Making architecture/design decisions |

## Sub-Agents

Sub-agents run in isolated contexts with specific tools and skills. Located in `.claude/agents/`.

### Domain Experts
| Agent | Skills | Tools | Use For |
|-------|--------|-------|---------|
| `ops-sme` | uld-domain | Read, Grep, Glob | ULD operations, ground handling, constraints |
| `network-planner` | uld-domain | Read, Grep, Glob | Repositioning costs, hub flows, capacity |
| `demand-analyst` | uld-domain | Read, Grep, Glob | Seasonality, booking patterns, demand drivers |

### Technical Roles
| Agent | Skills | Tools | Use For |
|-------|--------|-------|---------|
| `architect` | uld-domain, python-standards | Read, Grep, Glob, WebFetch | System design, tech choices, trade-offs |
| `data-engineer` | uld-domain, python-standards | Read, Grep, Glob, Bash, Edit, Write | Schemas, pipelines, data quality |
| `ml-engineer` | uld-domain, python-standards, ml-patterns | Read, Grep, Glob, Bash, Edit, Write | Models, training, evaluation |
| `backend-developer` | python-standards, api-design | Read, Grep, Glob, Bash, Edit, Write | API implementation, services |
| `qa-engineer` | python-standards | Read, Grep, Glob, Bash | Test strategy, edge cases, validation |

### How Skills + Agents Work Together

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Conversation                         │
│  Skills auto-activate based on context (always available)   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  ops-sme     │  │  ml-engineer │  │  backend-dev │       │
│  │  (isolated)  │  │  (isolated)  │  │  (isolated)  │       │
│  │  +uld-domain │  │  +ml-patterns│  │  +api-design │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│       Subagents spawn with their own skills loaded          │
└─────────────────────────────────────────────────────────────┘
```

- **Skills** = Knowledge that auto-loads in current context
- **Subagents** = Isolated specialists that spawn for deep work

### Orchestration Guidelines
- **Design phase**: `architect` first → domain experts for validation
- **Data modeling**: `data-engineer` → `ops-sme` for domain accuracy
- **ML development**: `demand-analyst` → `ml-engineer` → `qa-engineer`
- **Implementation**: `backend-developer` → `qa-engineer` for tests

## Build Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run single test
pytest tests/test_forecasting.py::test_demand_prediction -v

# Type checking
mypy src

# Linting
ruff check src tests

# Start API server
uvicorn src.api.main:app --reload
```

## Key Data Entities

- **ULD**: ID, type (AKE/AKH/PMC/LD3/LD7), status, location, last_update
- **Station**: Airport code, hub_tier, capacity by ULD type
- **Flight**: Number, route, schedule, aircraft_type, uld_positions
- **Forecast**: Station, date, uld_type, predicted_demand, confidence_interval
- **Repositioning**: Origin, destination, flight, cost, benefit_score

## Success Metrics

- Forecast MAPE < 15% for 7-day horizon
- Reduction in ULD-related delays
- Lower deadheading cost per ton-km
- On-time departure % improvement

## Architecture Decisions

All significant decisions are documented in [docs/decisions/](docs/decisions/). Key decisions:

| ADR | Title | Status |
|-----|-------|--------|
| [API-001](docs/decisions/API-001-flight-data-apis.md) | Flight Data API Selection | Proposed |
| [API-002](docs/decisions/API-002-weather-events-apis.md) | Weather & Events APIs | Proposed |
| [DATA-001](docs/decisions/DATA-001-datasets-strategy.md) | Datasets Strategy | Proposed |
| [ARCH-001](docs/decisions/ARCH-001-geolocation-frequency.md) | Geolocation Frequency | Accepted |
| [ML-001](docs/decisions/ML-001-forecasting-approach.md) | Forecasting Mathematical Approach | Accepted |

## Project Structure

```
src/
├── api/                    # FastAPI application
│   ├── main.py            # App creation, health endpoints
│   ├── dependencies.py    # DI container for services
│   └── routers/           # API endpoint routers
│       ├── tracking.py    # ULD position/inventory endpoints
│       ├── forecasting.py # Demand/supply forecast endpoints
│       ├── recommendations.py # Repositioning recommendations
│       └── stations.py    # Station information endpoints
├── domain/                 # Core business entities (Pydantic models)
│   ├── uld.py             # ULD, ULDType, ULDStatus, ULDInventory
│   ├── station.py         # Station, StationTier, DELTA_STATIONS
│   ├── flight.py          # Flight, FlightSchedule, AircraftType
│   ├── forecast.py        # DemandForecast, SupplyForecast, QuantileForecast
│   └── recommendation.py  # RepositioningRecommendation, CostBenefit
├── services/              # Business logic layer
│   ├── tracking.py        # ULD position tracking service
│   ├── forecasting.py     # Demand/supply forecasting service
│   ├── recommendations.py # Repositioning recommendations
│   └── optimization.py    # Network-wide optimization (greedy solver)
├── data/
│   ├── clients/           # External API clients
│   │   ├── base.py        # Abstract base + factory pattern
│   │   ├── aviation_stack.py  # AviationStack flight API
│   │   ├── noaa_weather.py    # NOAA aviation weather API
│   │   └── open_meteo.py      # Open-Meteo weather API
│   ├── downloaders/       # Dataset downloaders
│   │   └── bts_downloader.py  # BTS TranStats downloader
│   └── synthetic/         # Synthetic data generators
│       ├── flight_generator.py   # Flight schedule generation
│       ├── uld_generator.py      # ULD fleet/inventory generation
│       ├── demand_generator.py   # Demand pattern generation
│       └── scenario_generator.py # Disruption scenarios
├── forecasting/           # ML pipeline
│   ├── features.py        # Feature engineering (temporal, lag, rolling, Fourier)
│   ├── models.py          # LightGBM, Prophet, Ensemble forecasters
│   ├── uncertainty.py     # Conformalized quantile regression (CQR)
│   └── hierarchy.py       # Hierarchical forecast reconciliation
└── database/              # Persistence layer
    ├── models.py          # SQLAlchemy models
    ├── repository.py      # Repository pattern for data access
    └── session.py         # Async session management
```

## Running the System

### Local Development
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run demo
python demo.py

# Start API server
uvicorn src.api:app --reload --port 8000

# API docs available at http://localhost:8000/docs
```

### Docker
```bash
# Build and run production image
docker build -t uld-forecasting .
docker run -p 8000:8000 uld-forecasting

# Or use docker-compose
docker-compose up api              # Production API
docker-compose --profile dev up    # Development with hot-reload
docker-compose --profile test up   # Run tests
docker-compose --profile postgres up  # With PostgreSQL
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | sqlite+aiosqlite:///./uld_forecasting.db | Database connection |
| `PORT` | 8000 | API server port |
| `LOG_LEVEL` | INFO | Logging level |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/stations/` | GET | List all stations |
| `/api/v1/stations/hubs` | GET | List hub stations |
| `/api/v1/tracking/position/{uld_id}` | GET | Get ULD position |
| `/api/v1/tracking/inventory/{station}` | GET | Station inventory |
| `/api/v1/forecasting/demand/{station}` | GET | Demand forecast |
| `/api/v1/forecasting/supply/{station}` | GET | Supply forecast |
| `/api/v1/forecasting/network` | GET | Network-wide forecast |
| `/api/v1/recommendations/repositioning` | GET | Repositioning recommendations |
| `/api/v1/recommendations/optimize` | POST | Run network optimization |

## Reference Documents

- [ULD_Forecasting_Canvas.pdf](ULD_Forecasting_Canvas.pdf) - Requirements canvas from Team International
- [docs/ML-FRAMEWORK-uld-forecasting.md](docs/ML-FRAMEWORK-uld-forecasting.md) - Mathematical framework specification
