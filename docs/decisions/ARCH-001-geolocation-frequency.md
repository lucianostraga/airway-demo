# ARCH-001: ULD Geolocation Frequency Constraint

**Date**: 2026-01-10
**Status**: Accepted
**Deciders**: Architecture Team, Operations SME

## Context

Delta's ULD tracking devices provide geolocation updates **once every 24 hours**. This constraint raises concerns about forecasting accuracy and operational responsiveness.

**Key Question**: Is 24-hour update frequency sufficient for ULD forecasting?

## Decision

**Finding**: 24-hour geolocation updates are **insufficient for day-of-operations** but can be supplemented with alternative data sources to achieve effective 4-6 hour position accuracy.

**Approach**: Implement a **Hybrid Position Tracking Model** that combines:
1. 24-hour geolocation as baseline anchor
2. Flight event data for transit tracking
3. Inferred positions based on historical patterns
4. Confidence decay for stale data

## Analysis by Station Type

| Station Type | ULD Movements/Day | Avg Dwell Time | 24-hr Adequate? |
|--------------|-------------------|----------------|-----------------|
| Major Hubs (ATL, DTW) | 400-600 | 4-8 hours | ❌ No |
| Focus Cities (BOS, LAX) | 80-200 | 12-36 hours | ⚠️ Marginal |
| Spoke Stations | 5-40 | 24-72 hours | ✅ Acceptable |

## Operational Impact of 24-Hour Constraint

### Without Mitigation
- 16-32 ULD handling cycles missed per day at hubs
- Cannot detect diversions, delays, or customs holds in time
- Requires 20-30% larger fleet to buffer uncertainty
- Reactive instead of proactive repositioning

### With Hybrid Approach
- Effective 4-6 hour accuracy at major stations
- Event-driven exception detection
- Confidence-scored positions for decision-making

## Hybrid Position Tracking Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Position Certainty Engine                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Inputs:                                                     │
│  ├── Geolocation (24-hour baseline)                         │
│  ├── Flight Manifests (FWB/FHL messages)                    │
│  ├── Departure/Arrival Events (ACARS)                       │
│  ├── Ground Handler Scans (where available)                 │
│  └── Historical Patterns (ML-inferred)                      │
│                                                              │
│  Output:                                                     │
│  ├── Estimated Position                                      │
│  ├── Confidence Score (0-100%)                              │
│  ├── Time Since Last Confirmed Position                     │
│  └── Next Expected Update Time                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Position Confidence Algorithm

```python
def calculate_position_confidence(
    last_geolocation: datetime,
    last_flight_event: datetime | None,
    last_scan_event: datetime | None,
    station_type: StationType,
) -> float:
    """
    Confidence decays over time since last confirmed position.
    Different data sources have different confidence weights.
    """
    now = datetime.utcnow()

    # Base confidence from geolocation (decays over 24 hours)
    geo_age_hours = (now - last_geolocation).total_seconds() / 3600
    geo_confidence = max(0, 100 - (geo_age_hours * 4))  # Loses 4%/hour

    # Boost from flight events (if more recent)
    if last_flight_event and last_flight_event > last_geolocation:
        flight_age_hours = (now - last_flight_event).total_seconds() / 3600
        flight_confidence = max(0, 95 - (flight_age_hours * 8))
        return max(geo_confidence, flight_confidence)

    # Boost from scan events (highest confidence when fresh)
    if last_scan_event and last_scan_event > last_geolocation:
        scan_age_hours = (now - last_scan_event).total_seconds() / 3600
        scan_confidence = max(0, 100 - (scan_age_hours * 10))
        return max(geo_confidence, scan_confidence)

    return geo_confidence
```

## Required Frequency by Use Case

| Use Case | Required Accuracy | Achievable with Hybrid |
|----------|-------------------|------------------------|
| Day-of-ops load planning | 15-30 min | ⚠️ 2-4 hours |
| Capacity allocation | 1-2 hours | ✅ Yes |
| Repositioning planning | 4-6 hours | ✅ Yes |
| 7-day forecast | 12-24 hours | ✅ Yes |
| Fleet planning | 24-48 hours | ✅ Yes |

## Consequences

**Benefits**:
- No hardware changes required (uses existing tracking)
- Supplements position data with readily available flight events
- Confidence scoring enables risk-aware decision making
- Provides foundation for future tracking upgrades

**Challenges**:
- Day-of-operations accuracy limited to 2-4 hours (vs ideal 15-30 min)
- Requires integration with flight event systems
- Position inference adds model complexity

**Accepted Trade-offs**:
- Accept reduced accuracy for day-of-ops in exchange for faster project delivery
- Build architecture that can leverage higher-frequency tracking if upgraded later

## Future Recommendations

1. **Short-term**: Integrate flight manifest and departure/arrival events
2. **Medium-term**: Negotiate scan data feeds from ground handlers at top 20 stations
3. **Long-term**: Build business case for cellular IoT or BLE beacon upgrades

## References

- Operations SME Assessment (internal consultation)
- [IATA ULD Management Standards](https://www.iata.org/en/programs/cargo/cargo-operations/unit-load-devices/)
