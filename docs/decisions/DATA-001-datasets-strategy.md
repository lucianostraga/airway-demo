# DATA-001: Datasets Strategy

**Date**: 2026-01-10
**Status**: Proposed
**Deciders**: Data Engineering, ML Engineering

## Context

The ULD Forecasting system requires training data for:
- Demand forecasting models (passenger/baggage → ULD requirements)
- Network flow patterns (ULD movement between stations)
- Seasonal and event-driven demand patterns

**Key Finding**: No public ULD-specific datasets exist. Airline operational data is proprietary.

## Decision

**Approach**: Hybrid strategy combining:
1. Public flight/delay datasets for pattern learning
2. BTS government data for official statistics
3. Synthetic ULD data generation calibrated to realistic parameters

## Available Public Datasets

### Flight Delay Datasets (Kaggle)

| Dataset | Period | Records | Use Case |
|---------|--------|---------|----------|
| [Flight Delay 2018-2024](https://www.kaggle.com/datasets/shubhamsingh42/flight-delay-dataset-2018-2024) | 6 years | Millions | Delay patterns, seasonality |
| [Flight Delay & Cancellation 2019-2023](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023) | 5 years | Millions | Cancellation impact on ULD flow |
| [US DOT 2015 Flight Delays](https://www.kaggle.com/datasets/usdot/flight-delays) | 2015 | ~6M | Official DOT baseline |

### BTS Government Data

| Dataset | Source | Update Frequency | Key Metrics |
|---------|--------|------------------|-------------|
| T-100 Domestic/International | transtats.bts.gov | Monthly | Passengers, freight, load factor |
| DB1B Survey | transtats.bts.gov | Quarterly | Origin-destination patterns |
| Air Travel Consumer Report | bts.gov | Monthly | Delays, mishandled baggage |
| On-Time Performance | transtats.bts.gov | Monthly | Delay causes, cancellations |

### What's Missing (Requires Synthesis)

- ULD inventory levels by station
- ULD type distribution by route
- Baggage-to-ULD conversion ratios
- ULD dwell times and cycle patterns
- Repositioning (deadheading) events

## Synthetic Data Generation Strategy

### Calibration Sources

| Parameter | Calibration Source |
|-----------|-------------------|
| Passenger volumes | BTS T-100 data |
| Flight schedules | Public timetables, FlightAware |
| Seasonal patterns | Historical delay datasets |
| Baggage ratios | Industry averages (0.8-1.2 bags/pax) |
| ULD capacities | IATA ULD specifications |
| Station characteristics | Delta hub information |

### Generation Approach

```python
# Synthetic ULD Data Model
class SyntheticDataGenerator:
    """
    Generate realistic ULD operational data calibrated to:
    - Real flight schedules (from APIs)
    - Real passenger volumes (from BTS)
    - Industry-standard ULD specifications
    - Seasonal/event patterns (from historical data)
    """

    def generate_uld_demand(self, station, date):
        # Base demand from passenger forecast
        passengers = self.forecast_passengers(station, date)
        bags_per_pax = self.sample_baggage_ratio(route_type)
        total_bags = passengers * bags_per_pax

        # Convert to ULD requirements
        uld_demand = self.bags_to_ulds(total_bags, aircraft_mix)

        # Add noise and variation
        return self.add_realistic_variation(uld_demand)
```

### Validation Approach

1. **Statistical validation**: Generated data matches known distributions
2. **Operational validation**: SME review of patterns and edge cases
3. **Backtesting**: Apply models to historical periods with known outcomes

## Consequences

**Benefits**:
- Can start development immediately without waiting for proprietary data
- Full control over data characteristics for testing edge cases
- Models learn transferable patterns (seasonality, delays, weather impact)

**Challenges**:
- Synthetic data may miss real-world anomalies
- ULD-specific parameters require SME calibration
- Production system will need retraining on actual Delta data

**Mitigation**:
- Design models to be retrained easily
- Document all synthetic data assumptions
- Build validation pipeline for when real data becomes available

## Alternatives Considered

1. **Wait for Delta data**: Delays project significantly
2. **Use only public data**: Missing ULD-specific patterns
3. **Partner with ULD provider (Jettainer, CHEP)**: Long negotiation, NDA issues

## References

- [BTS TranStats](https://www.transtats.bts.gov/)
- [Kaggle Flight Delay Datasets](https://www.kaggle.com/search?q=flight+delay)
- [IATA ULD Specifications](https://www.iata.org/en/programs/cargo/cargo-operations/unit-load-devices/)
