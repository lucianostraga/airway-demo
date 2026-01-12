---
name: uld-domain
description: ULD (Unit Load Device) domain knowledge for airline operations. Activates when discussing ULDs, containers, baggage handling, or airline logistics.
---

# ULD Domain Knowledge

## ULD Types and Specifications

| Type | IATA Code | Description | Typical Aircraft |
|------|-----------|-------------|------------------|
| LD3 | AKE | Lower deck container | A320, A330, A350, B777, B787 |
| LD7 | AKH | Lower deck container (large) | B747, B777, A380 |
| LD26 | AAF | Lower deck container | A300, A310 |
| PMC | P6P | Pallet with net | All wide-body |
| PLA | P1P | Pallet (20ft) | B747F, B777F |

## ULD Status Lifecycle

```
SERVICEABLE → IN_USE → EMPTY → SERVICEABLE
     ↓                           ↑
DAMAGED → OUT_OF_SERVICE → REPAIRED
```

**Status Definitions:**
- `SERVICEABLE`: Available for use, passed inspection
- `IN_USE`: Currently loaded on aircraft or in transit
- `EMPTY`: At station, available for loading
- `DAMAGED`: Requires inspection/repair
- `OUT_OF_SERVICE`: At MRO facility for repair

## Station Tiers

| Tier | Examples | Characteristics |
|------|----------|-----------------|
| Hub | ATL, DTW, MSP, SLC | High throughput, 24/7 ops, large inventory |
| Focus City | BOS, LAX, JFK, SEA | Medium throughput, priority routes |
| Spoke | Regional stations | Lower volume, depends on hub supply |

## Key Metrics

- **Dwell Time**: Hours a ULD spends at a station (target: < 24h at hubs)
- **Utilization Rate**: % of ULDs in active use (target: > 70%)
- **Turn Time**: Aircraft arrival to departure ULD swap (target: < 90 min)
- **Deadhead Ratio**: Empty repositioning moves / total moves (target: < 15%)

## Demand Drivers

1. **Passenger Count**: Primary driver, ~0.8 bags/pax domestic, ~1.2 bags/pax international
2. **Route Type**: Leisure routes have higher bag ratios
3. **Season**: Holiday peaks increase demand 40-80%
4. **Day of Week**: Friday/Sunday peaks for leisure, Mon/Thu for business
5. **Weather**: Storms cause irregular ops (IROP), disrupting ULD flow

## Terminology

- **Deadheading**: Moving empty ULDs to rebalance network
- **IROP**: Irregular Operations (delays, cancellations, diversions)
- **MRO**: Maintenance, Repair, Overhaul facility
- **Interline**: ULD transfer between airlines
- **Build-up**: Loading cargo/bags into ULD before flight
- **Break-down**: Unloading ULD after flight arrival
