# API-002: Weather & Events API Selection

**Date**: 2026-01-10
**Status**: Proposed
**Deciders**: Architecture Team

## Context

External signals significantly impact ULD demand:
- **Weather**: Storms cause delays, diversions, and demand spikes
- **Events**: Concerts, sports, conferences increase passenger/baggage volume
- **Disruptions**: Strikes, airport closures affect ULD flow

## Decision

### Weather APIs

**Primary**: NOAA Aviation Weather API (free, authoritative)
**Supplement**: Open-Meteo (free, global forecasts)

### Events API

**Primary**: PredictHQ (purpose-built for demand forecasting)

## Weather API Evaluation

| API | Aviation-Specific | Cost | Forecast Range | METAR/TAF |
|-----|-------------------|------|----------------|-----------|
| **NOAA Aviation Weather** | ✅ Yes | Free | 7 days | ✅ Yes |
| **Open-Meteo** | ❌ General | Free | 16 days | ❌ No |
| **OpenWeatherMap** | ❌ General | Freemium | 8 days | ❌ No |
| **Tomorrow.io** | ⚠️ Partial | $$$ | 15 days | ✅ Yes |

### NOAA Aviation Weather Selected Because:
- **Aviation-specific**: METAR, TAF, SIGMET, AIRMET data
- **Authoritative**: Official source for US aviation weather
- **Free**: No cost, reasonable rate limits
- **JSON API**: Easy integration at `aviationweather.gov/api/data/`

## Events API Evaluation

| API | Demand Scoring | Coverage | Event Types | ML-Ready |
|-----|----------------|----------|-------------|----------|
| **PredictHQ** | ✅ Built-in | 300K+ cities | 19 categories | ✅ Features API |
| **Ticketmaster** | ❌ No | Ticketed only | Limited | ❌ No |
| **Eventbrite** | ❌ No | Limited | Limited | ❌ No |

### PredictHQ Selected Because:
- **Purpose-built for forecasting**: Includes predicted attendance, impact scores
- **Comprehensive**: Concerts, sports, conferences, holidays, severe weather
- **Features API**: Pre-aggregated features ready for ML models
- **Proven**: Used by airlines, hotels, retailers for demand forecasting

## Consequences

**Benefits**:
- Weather data improves short-term forecast accuracy (0-72 hours)
- Event data captures demand spikes not visible in historical patterns
- Both APIs provide ML-ready features

**Challenges**:
- PredictHQ requires paid subscription
- Need to correlate events with specific stations
- Weather impact varies by ULD type (temperature-sensitive cargo)

## Integration Strategy

```
External Signals Pipeline:
├── Weather (hourly refresh)
│   ├── NOAA Aviation → METAR, TAF for each station
│   └── Open-Meteo → Extended forecast, global coverage
│
└── Events (daily refresh)
    └── PredictHQ → Events within radius of each station
        ├── Attendance predictions
        ├── Category (sports, concerts, conferences)
        └── Impact score
```

## Alternatives Considered

1. **Tomorrow.io**: Good aviation features but expensive
2. **Manual event tracking**: Not scalable
3. **Social media signals**: Noisy, hard to quantify

## References

- [NOAA Aviation Weather API](https://aviationweather.gov/data/api/)
- [Open-Meteo](https://open-meteo.com/)
- [PredictHQ Events API](https://www.predicthq.com/apis)
- [PredictHQ Features API](https://www.predicthq.com/apis/features-api)
