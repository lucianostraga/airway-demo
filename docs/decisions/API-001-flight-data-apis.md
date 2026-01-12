# API-001: Flight Data API Selection

**Date**: 2026-01-10
**Status**: Proposed
**Deciders**: Architecture Team

## Context

The ULD Forecasting system requires real-time flight data including:
- Flight schedules (planned departures/arrivals)
- Real-time flight status (delays, cancellations, diversions)
- Historical flight data for pattern analysis
- Aircraft type information (affects ULD capacity)

## Decision

**Recommended Primary API**: FlightAware AeroAPI

**Recommended Backup/Supplement**: Aviationstack (free tier for development)

## Evaluation Matrix

| API | Real-time | Historical | Price | Coverage | Delta Support |
|-----|-----------|------------|-------|----------|---------------|
| **FlightAware AeroAPI** | ✅ Excellent | ✅ Since 2011 | $$$ | Global | ✅ Full |
| **Aviationstack** | ✅ Good | ✅ Yes | $ | Global | ✅ Yes |
| **OAG Flight Info** | ✅ Excellent | ✅ Yes | $$$$ | Global | ✅ Yes |
| **AirLabs** | ✅ Good | ⚠️ Limited | $$ | Global | ✅ Yes |
| **FlightLabs** | ✅ Good | ⚠️ Limited | Free tier | Global | ✅ Yes |

## Rationale

1. **FlightAware AeroAPI** selected because:
   - Deepest historical data (2011+) for ML training
   - Predictive ETAs powered by FlightAware Foresight
   - Firehose option for real-time streaming
   - Industry-standard in aviation applications
   - REST API with comprehensive documentation

2. **Aviationstack** as backup because:
   - Free tier (100 requests/month) for development
   - Good for prototyping before production API costs
   - Simple REST interface

## Consequences

**Benefits**:
- Rich historical data improves forecasting model accuracy
- Real-time updates enable responsive allocation
- Predictive ETAs help anticipate ULD arrivals

**Challenges**:
- Cost: FlightAware requires paid subscription ($$$)
- Rate limits may require caching strategy
- Need to handle API failures gracefully

## Alternatives Considered

1. **OAG Flight Info**: Superior data but highest cost
2. **Direct Delta feed**: Would require internal integration (longer timeline)
3. **ADS-B data (FlightRadar24)**: Real-time only, no schedules

## References

- [FlightAware AeroAPI](https://www.flightaware.com/commercial/aeroapi/)
- [Aviationstack](https://aviationstack.com/)
- [OAG Flight Info API](https://www.oag.com/flight-info-api)
