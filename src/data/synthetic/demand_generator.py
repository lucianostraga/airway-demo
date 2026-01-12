"""
Demand pattern generator for synthetic data.

Generates realistic ULD demand patterns with:
- Seasonality (annual, weekly, daily)
- Holiday effects
- Weather disruptions
- Trend components
- Station-specific patterns
"""

from datetime import datetime, timedelta, timezone
from typing import Iterator
import math

import numpy as np

from src.domain import (
    DemandForecast,
    QuantileForecast,
    ForecastGranularity,
    ForecastConfidence,
    ULDType,
    DELTA_STATIONS,
    StationTier,
)


class DemandPatternGenerator:
    """
    Generate realistic synthetic demand patterns.

    Uses statistical models to create demand time series with:
    - Multiple seasonality components
    - Calendar effects (holidays, events)
    - Weather impacts
    - Random noise with realistic distributions
    """

    # US Federal holidays (month, day)
    US_HOLIDAYS = [
        (1, 1),    # New Year
        (1, 15),   # MLK Day (approx)
        (2, 19),   # Presidents Day (approx)
        (5, 27),   # Memorial Day (approx)
        (7, 4),    # Independence Day
        (9, 2),    # Labor Day (approx)
        (10, 14),  # Columbus Day (approx)
        (11, 11),  # Veterans Day
        (11, 28),  # Thanksgiving (approx)
        (12, 25),  # Christmas
    ]

    # High travel periods
    HIGH_TRAVEL_PERIODS = [
        ((12, 20), (1, 3)),    # Christmas/New Year
        ((3, 10), (3, 20)),    # Spring Break
        ((6, 15), (8, 20)),    # Summer
        ((11, 20), (11, 30)),  # Thanksgiving
    ]

    def __init__(self, seed: int | None = None):
        """Initialize generator with optional random seed."""
        self.rng = np.random.default_rng(seed)

    def generate_demand_series(
        self,
        station: str,
        start_date: datetime,
        end_date: datetime,
        granularity: ForecastGranularity = ForecastGranularity.DAILY,
    ) -> list[DemandForecast]:
        """
        Generate demand time series for a station.

        Args:
            station: Station code
            start_date: Start of series
            end_date: End of series
            granularity: Time granularity

        Returns:
            List of DemandForecast objects
        """
        forecasts = []

        # Get station base demand
        base_demand = self._get_base_demand(station)

        # Generate for each time period
        current = start_date
        step = self._get_step(granularity)

        while current < end_date:
            demand = self._calculate_demand(station, current, base_demand)

            # Generate quantile forecast
            forecast = self._create_forecast(
                station=station,
                timestamp=current,
                demand=demand,
                granularity=granularity,
            )
            forecasts.append(forecast)

            current += step

        return forecasts

    def generate_historical_data(
        self,
        station: str,
        days: int = 365,
        include_actuals: bool = True,
    ) -> list[tuple[DemandForecast, int]]:
        """
        Generate historical demand data with actuals.

        Args:
            station: Station code
            days: Number of days of history
            include_actuals: Include actual values (for training)

        Returns:
            List of (forecast, actual) tuples
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        forecasts = self.generate_demand_series(
            station, start_date, end_date, ForecastGranularity.DAILY
        )

        results = []
        for forecast in forecasts:
            if include_actuals:
                # Actual is drawn from the forecast distribution
                actual = int(
                    self.rng.normal(
                        forecast.total_demand.q50,
                        forecast.total_demand.iqr / 1.35,  # Approximate std
                    )
                )
                actual = max(0, actual)
            else:
                actual = 0

            results.append((forecast, actual))

        return results

    def _get_base_demand(self, station: str) -> float:
        """Get base demand level for a station."""
        station_info = DELTA_STATIONS.get(station)
        tier = station_info.tier if station_info else StationTier.SPOKE

        # Base ULDs per day by tier
        tier_base = {
            StationTier.HUB: 150,  # High volume hub
            StationTier.FOCUS_CITY: 50,
            StationTier.SPOKE: 15,
            StationTier.INTERNATIONAL: 35,
        }

        return tier_base.get(tier, 15.0)

    def _get_step(self, granularity: ForecastGranularity) -> timedelta:
        """Get time step for granularity."""
        if granularity == ForecastGranularity.HOURLY:
            return timedelta(hours=1)
        elif granularity == ForecastGranularity.DAILY:
            return timedelta(days=1)
        else:  # WEEKLY
            return timedelta(days=7)

    def _calculate_demand(
        self,
        station: str,
        timestamp: datetime,
        base_demand: float,
    ) -> dict:
        """
        Calculate demand components.

        Returns dict with point estimate and components.
        """
        demand = base_demand

        # 1. Annual seasonality (peak summer, holiday)
        day_of_year = timestamp.timetuple().tm_yday
        annual_factor = 1.0 + 0.15 * math.sin(2 * math.pi * (day_of_year - 172) / 365)
        demand *= annual_factor

        # 2. Day of week (lower weekends)
        dow = timestamp.weekday()
        dow_factors = [1.0, 1.05, 1.0, 0.95, 1.1, 0.85, 0.80]
        demand *= dow_factors[dow]

        # 3. Hour of day (if hourly)
        hour = timestamp.hour
        hour_factor = 0.7 + 0.6 * math.sin(math.pi * (hour - 6) / 12) if 6 <= hour <= 22 else 0.3
        # Only apply if actually hourly
        if hour != 0:
            demand *= hour_factor

        # 4. Holiday effects
        if self._is_holiday_period(timestamp):
            demand *= 1.25

        if self._is_day_before_holiday(timestamp):
            demand *= 1.15

        # 5. Weather disruption (random)
        if self.rng.random() < 0.05:  # 5% chance of disruption
            disruption_factor = self.rng.uniform(0.5, 0.8)
            demand *= disruption_factor

        # 6. Random noise (log-normal for positive values)
        noise_factor = self.rng.lognormal(0, 0.1)
        demand *= noise_factor

        return {
            "point": max(1, int(demand)),
            "annual_factor": annual_factor,
            "dow_factor": dow_factors[dow],
            "is_holiday": self._is_holiday_period(timestamp),
        }

    def _is_holiday_period(self, timestamp: datetime) -> bool:
        """Check if date is in a high travel period."""
        month = timestamp.month
        day = timestamp.day

        for (start_m, start_d), (end_m, end_d) in self.HIGH_TRAVEL_PERIODS:
            if start_m <= end_m:
                # Same year
                if (start_m, start_d) <= (month, day) <= (end_m, end_d):
                    return True
            else:
                # Crosses year boundary
                if (month, day) >= (start_m, start_d) or (month, day) <= (end_m, end_d):
                    return True

        return False

    def _is_day_before_holiday(self, timestamp: datetime) -> bool:
        """Check if date is day before a holiday."""
        next_day = timestamp + timedelta(days=1)
        return (next_day.month, next_day.day) in self.US_HOLIDAYS

    def _create_forecast(
        self,
        station: str,
        timestamp: datetime,
        demand: dict,
        granularity: ForecastGranularity,
    ) -> DemandForecast:
        """Create a DemandForecast object."""
        point = demand["point"]

        # Generate uncertainty based on point estimate
        # Higher uncertainty for higher demand
        std = max(1, point * 0.15)

        # Quantile forecast
        q05 = max(0, int(point - 1.645 * std))
        q25 = max(0, int(point - 0.675 * std))
        q50 = point
        q75 = int(point + 0.675 * std)
        q95 = int(point + 1.645 * std)

        total_quantile = QuantileForecast(
            q05=q05, q25=q25, q50=q50, q75=q75, q95=q95
        )

        # Demand by ULD type (proportional to total)
        type_distribution = {
            ULDType.AKE: 0.50,
            ULDType.AKH: 0.15,
            ULDType.PMC: 0.20,
            ULDType.AKN: 0.10,
            ULDType.AAP: 0.05,
        }

        demand_by_type = {}
        for uld_type, proportion in type_distribution.items():
            type_point = int(point * proportion)
            type_std = max(1, type_point * 0.2)

            demand_by_type[uld_type] = QuantileForecast(
                q05=max(0, int(type_point - 1.645 * type_std)),
                q25=max(0, int(type_point - 0.675 * type_std)),
                q50=type_point,
                q75=int(type_point + 0.675 * type_std),
                q95=int(type_point + 1.645 * type_std),
            )

        # Confidence based on uncertainty
        cv = std / max(1, point)  # Coefficient of variation
        if cv < 0.1:
            confidence = ForecastConfidence.HIGH
        elif cv < 0.25:
            confidence = ForecastConfidence.MEDIUM
        else:
            confidence = ForecastConfidence.LOW

        return DemandForecast(
            station=station,
            forecast_time=timestamp,
            generated_at=datetime.now(timezone.utc),
            granularity=granularity,
            demand_by_type=demand_by_type,
            total_demand=total_quantile,
            scheduled_departures=int(point / 8),  # Rough: 8 ULDs per flight
            confidence=confidence,
            is_anomaly=demand.get("is_holiday", False),
            features_used=["seasonality", "dow", "holiday", "trend"],
        )

    def add_anomalies(
        self,
        forecasts: list[DemandForecast],
        anomaly_rate: float = 0.02,
    ) -> list[DemandForecast]:
        """
        Add random anomalies to a forecast series.

        Args:
            forecasts: Original forecasts
            anomaly_rate: Probability of anomaly

        Returns:
            Forecasts with anomalies added
        """
        result = []

        for forecast in forecasts:
            if self.rng.random() < anomaly_rate:
                # Create anomalous demand (spike or drop)
                factor = self.rng.choice([0.3, 2.0, 2.5])

                new_q50 = int(forecast.total_demand.q50 * factor)
                std = max(1, new_q50 * 0.15)

                new_total = QuantileForecast(
                    q05=max(0, int(new_q50 - 1.645 * std)),
                    q25=max(0, int(new_q50 - 0.675 * std)),
                    q50=new_q50,
                    q75=int(new_q50 + 0.675 * std),
                    q95=int(new_q50 + 1.645 * std),
                )

                # Create modified forecast
                modified = DemandForecast(
                    station=forecast.station,
                    forecast_time=forecast.forecast_time,
                    generated_at=forecast.generated_at,
                    granularity=forecast.granularity,
                    demand_by_type=forecast.demand_by_type,
                    total_demand=new_total,
                    scheduled_departures=forecast.scheduled_departures,
                    confidence=ForecastConfidence.LOW,
                    is_anomaly=True,
                    anomaly_score=abs(factor - 1.0),
                    features_used=forecast.features_used,
                )
                result.append(modified)
            else:
                result.append(forecast)

        return result


class SupplyPatternGenerator:
    """
    Generate supply patterns based on flight arrivals and departures.

    Supply is driven by:
    - Arriving flights bringing ULDs
    - Departing flights removing ULDs
    - Repositioning movements
    - Maintenance removals
    """

    def __init__(self, seed: int | None = None):
        """Initialize generator."""
        self.rng = np.random.default_rng(seed)

    def generate_from_demand(
        self,
        demand_forecasts: list[DemandForecast],
        initial_inventory: int,
        arrival_ratio: float = 0.95,
    ) -> list[tuple[DemandForecast, int]]:
        """
        Generate supply series based on demand.

        Creates supply that roughly balances demand with some imbalance.

        Args:
            demand_forecasts: Demand forecasts
            initial_inventory: Starting inventory
            arrival_ratio: Average arrivals as ratio of departures

        Returns:
            List of (demand_forecast, supply) tuples
        """
        results = []
        current_inventory = initial_inventory

        for demand in demand_forecasts:
            departures = demand.total_demand.q50

            # Arrivals with some randomness around ratio
            arrivals = int(departures * self.rng.uniform(
                arrival_ratio - 0.1,
                arrival_ratio + 0.1
            ))

            # Update inventory
            current_inventory = max(0, current_inventory - departures + arrivals)

            results.append((demand, current_inventory))

        return results
