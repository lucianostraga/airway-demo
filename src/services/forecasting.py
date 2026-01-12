"""
Forecasting Service.

Generates demand and supply forecasts for ULDs across the network.
Implements hierarchical probabilistic ensemble forecasting.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from src.domain import (
    DemandForecast,
    SupplyForecast,
    ImbalanceForecast,
    NetworkForecast,
    QuantileForecast,
    ForecastGranularity,
    ForecastConfidence,
    ULDType,
    Flight,
    FlightSchedule,
    DELTA_STATIONS,
)


class ForecastingService:
    """
    Service for generating ULD demand and supply forecasts.

    Implements a simplified version of the ML framework:
    - Base forecasters (schedule-based, historical patterns)
    - Ensemble combination
    - Quantile predictions with uncertainty

    For production, this would integrate with the full ML pipeline
    in src/forecasting/.

    Usage:
        service = ForecastingService()
        forecast = await service.forecast_demand("ATL", hours_ahead=24)
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

        # ULD type distribution (typical for mixed fleet)
        self.type_distribution = {
            ULDType.AKE: 0.50,
            ULDType.AKH: 0.15,
            ULDType.PMC: 0.20,
            ULDType.AKN: 0.10,
            ULDType.AAP: 0.05,
        }

    async def forecast_demand(
        self,
        station: str,
        hours_ahead: int = 24,
        granularity: ForecastGranularity = ForecastGranularity.HOURLY,
        schedule: FlightSchedule | None = None,
    ) -> list[DemandForecast]:
        """
        Generate demand forecast for a station.

        Args:
            station: Station code
            hours_ahead: Forecast horizon in hours
            granularity: Time granularity
            schedule: Optional flight schedule for schedule-based forecast

        Returns:
            List of DemandForecast objects
        """
        forecasts = []
        now = datetime.now(timezone.utc)

        # Determine time step
        if granularity == ForecastGranularity.HOURLY:
            step = timedelta(hours=1)
            n_periods = hours_ahead
        elif granularity == ForecastGranularity.DAILY:
            step = timedelta(days=1)
            n_periods = (hours_ahead + 23) // 24
        else:
            step = timedelta(weeks=1)
            n_periods = (hours_ahead + 167) // 168

        # Get base demand from station tier
        base_demand = self._get_station_base_demand(station)

        for i in range(n_periods):
            forecast_time = now + step * i

            # Calculate demand components
            demand = self._calculate_demand_components(
                base_demand, forecast_time, schedule, station
            )

            forecast = self._create_demand_forecast(
                station=station,
                forecast_time=forecast_time,
                generated_at=now,
                demand=demand,
                granularity=granularity,
            )
            forecasts.append(forecast)

        return forecasts

    async def forecast_supply(
        self,
        station: str,
        current_inventory: dict[ULDType, int],
        hours_ahead: int = 24,
        schedule: FlightSchedule | None = None,
    ) -> list[SupplyForecast]:
        """
        Generate supply forecast for a station.

        Args:
            station: Station code
            current_inventory: Current inventory by type
            hours_ahead: Forecast horizon
            schedule: Flight schedule for arrival/departure inference

        Returns:
            List of SupplyForecast objects
        """
        forecasts = []
        now = datetime.now(timezone.utc)

        # Simple supply model: current inventory + expected arrivals - departures
        running_inventory = sum(current_inventory.values())

        for hour in range(hours_ahead):
            forecast_time = now + timedelta(hours=hour)

            # Estimate arrivals and departures
            arrivals, departures = self._estimate_movements(
                station, forecast_time, schedule
            )

            running_inventory = max(0, running_inventory + arrivals - departures)

            # Add uncertainty
            std = max(1, running_inventory * 0.1)

            total_supply = QuantileForecast(
                q05=max(0, running_inventory - 1.645 * std),
                q25=max(0, running_inventory - 0.675 * std),
                q50=float(running_inventory),
                q75=running_inventory + 0.675 * std,
                q95=running_inventory + 1.645 * std,
            )

            # By type (proportional)
            supply_by_type = {}
            for uld_type, ratio in self.type_distribution.items():
                type_supply = running_inventory * ratio
                type_std = max(1, type_supply * 0.15)
                supply_by_type[uld_type] = QuantileForecast(
                    q05=max(0, type_supply - 1.645 * type_std),
                    q25=max(0, type_supply - 0.675 * type_std),
                    q50=type_supply,
                    q75=type_supply + 0.675 * type_std,
                    q95=type_supply + 1.645 * type_std,
                )

            forecasts.append(
                SupplyForecast(
                    station=station,
                    forecast_time=forecast_time,
                    generated_at=now,
                    granularity=ForecastGranularity.HOURLY,
                    supply_by_type=supply_by_type,
                    total_supply=total_supply,
                    current_inventory=sum(current_inventory.values()),
                    expected_arrivals=arrivals,
                    expected_departures=departures,
                    confidence=ForecastConfidence.MEDIUM,
                )
            )

        return forecasts

    async def forecast_imbalance(
        self,
        station: str,
        demand_forecasts: list[DemandForecast],
        supply_forecasts: list[SupplyForecast],
    ) -> list[ImbalanceForecast]:
        """
        Calculate supply-demand imbalance forecasts.

        Args:
            station: Station code
            demand_forecasts: Demand forecasts
            supply_forecasts: Supply forecasts

        Returns:
            List of ImbalanceForecast objects
        """
        imbalances = []
        now = datetime.now(timezone.utc)

        # Match demand and supply forecasts by time
        for demand, supply in zip(demand_forecasts, supply_forecasts):
            # Calculate imbalance (positive = surplus, negative = shortage)
            imbalance_point = supply.total_supply.q50 - demand.total_demand.q50

            # Propagate uncertainty
            combined_std = np.sqrt(
                (supply.total_supply.iqr / 1.35) ** 2
                + (demand.total_demand.iqr / 1.35) ** 2
            )

            total_imbalance = QuantileForecast(
                q05=imbalance_point - 1.645 * combined_std,
                q25=imbalance_point - 0.675 * combined_std,
                q50=imbalance_point,
                q75=imbalance_point + 0.675 * combined_std,
                q95=imbalance_point + 1.645 * combined_std,
            )

            # Calculate shortage probability (P(supply < demand))
            # Approximate using normal CDF
            if combined_std > 0:
                z_score = -imbalance_point / combined_std
                from scipy.stats import norm  # type: ignore
                shortage_prob = float(norm.cdf(z_score))
            else:
                shortage_prob = 0.0 if imbalance_point >= 0 else 1.0

            # Severity classification
            if shortage_prob > 0.8:
                severity = "critical"
            elif shortage_prob > 0.5:
                severity = "warning"
            else:
                severity = "normal"

            # By type
            imbalance_by_type = {}
            for uld_type in ULDType:
                if uld_type in demand.demand_by_type and uld_type in supply.supply_by_type:
                    d = demand.demand_by_type[uld_type]
                    s = supply.supply_by_type[uld_type]
                    type_imbalance = s.q50 - d.q50
                    type_std = np.sqrt((s.iqr / 1.35) ** 2 + (d.iqr / 1.35) ** 2)

                    imbalance_by_type[uld_type] = QuantileForecast(
                        q05=type_imbalance - 1.645 * type_std,
                        q25=type_imbalance - 0.675 * type_std,
                        q50=type_imbalance,
                        q75=type_imbalance + 0.675 * type_std,
                        q95=type_imbalance + 1.645 * type_std,
                    )

            imbalances.append(
                ImbalanceForecast(
                    station=station,
                    forecast_time=demand.forecast_time,
                    generated_at=now,
                    imbalance_by_type=imbalance_by_type,
                    total_imbalance=total_imbalance,
                    shortage_probability=shortage_prob,
                    severity=severity,
                )
            )

        return imbalances

    async def forecast_network(
        self,
        stations: list[str] | None = None,
        hours_ahead: int = 24,
        current_inventories: dict[str, dict[ULDType, int]] | None = None,
    ) -> NetworkForecast:
        """
        Generate network-wide forecast.

        Args:
            stations: Stations to include (default: all)
            hours_ahead: Forecast horizon
            current_inventories: Current inventories by station

        Returns:
            NetworkForecast object
        """
        stations = stations or list(DELTA_STATIONS.keys())
        now = datetime.now(timezone.utc)

        station_demand = {}
        station_supply = {}
        station_imbalance = {}
        shortage_stations = []
        surplus_stations = []

        total_demand_sum = 0.0
        total_supply_sum = 0.0

        for station in stations:
            # Get current inventory or default
            inventory = current_inventories.get(station, {}) if current_inventories else {}
            if not inventory:
                # Default inventory based on station tier
                base = self._get_station_base_demand(station) * 2
                inventory = {
                    uld_type: int(base * ratio)
                    for uld_type, ratio in self.type_distribution.items()
                }

            # Generate forecasts
            demand_fc = await self.forecast_demand(station, hours_ahead)
            supply_fc = await self.forecast_supply(station, inventory, hours_ahead)
            imbalance_fc = await self.forecast_imbalance(station, demand_fc, supply_fc)

            # Use first period as representative
            if demand_fc:
                station_demand[station] = demand_fc[0]
                total_demand_sum += demand_fc[0].total_demand.q50

            if supply_fc:
                station_supply[station] = supply_fc[0]
                total_supply_sum += supply_fc[0].total_supply.q50

            if imbalance_fc:
                station_imbalance[station] = imbalance_fc[0]
                if imbalance_fc[0].shortage_probability > 0.5:
                    shortage_stations.append(station)
                elif imbalance_fc[0].total_imbalance.q50 > 10:
                    surplus_stations.append(station)

        # Network totals
        network_std = np.sqrt(len(stations)) * 5  # Rough approximation

        total_network_demand = QuantileForecast(
            q05=max(0, total_demand_sum - 1.645 * network_std),
            q25=max(0, total_demand_sum - 0.675 * network_std),
            q50=total_demand_sum,
            q75=total_demand_sum + 0.675 * network_std,
            q95=total_demand_sum + 1.645 * network_std,
        )

        total_network_supply = QuantileForecast(
            q05=max(0, total_supply_sum - 1.645 * network_std),
            q25=max(0, total_supply_sum - 0.675 * network_std),
            q50=total_supply_sum,
            q75=total_supply_sum + 0.675 * network_std,
            q95=total_supply_sum + 1.645 * network_std,
        )

        return NetworkForecast(
            generated_at=now,
            forecast_horizon_hours=hours_ahead,
            granularity=ForecastGranularity.HOURLY,
            station_demand=station_demand,
            station_supply=station_supply,
            station_imbalance=station_imbalance,
            total_network_demand=total_network_demand,
            total_network_supply=total_network_supply,
            shortage_stations=shortage_stations,
            surplus_stations=surplus_stations,
        )

    def _get_station_base_demand(self, station: str) -> float:
        """Get base demand for a station."""
        station_info = DELTA_STATIONS.get(station)
        if not station_info:
            return 10.0

        tier_base = {
            "hub": 100.0,
            "focus_city": 40.0,
            "spoke": 10.0,
            "international": 25.0,
        }

        return tier_base.get(station_info.tier.value, 10.0)

    def _calculate_demand_components(
        self,
        base_demand: float,
        forecast_time: datetime,
        schedule: FlightSchedule | None,
        station: str,
    ) -> dict[str, Any]:
        """Calculate demand with time-based adjustments."""
        demand = base_demand

        # Hour of day effect
        hour = forecast_time.hour
        if 6 <= hour <= 10:  # Morning bank
            demand *= 1.3
        elif 11 <= hour <= 14:  # Midday
            demand *= 1.0
        elif 15 <= hour <= 20:  # Evening bank
            demand *= 1.2
        else:  # Night
            demand *= 0.5

        # Day of week
        dow = forecast_time.weekday()
        if dow >= 5:  # Weekend
            demand *= 0.85

        # If we have schedule, use it
        if schedule:
            departures = [
                f for f in schedule.flights
                if f.origin == station
                and f.is_widebody
                and abs((f.scheduled_departure - forecast_time).total_seconds()) < 3600
            ]
            demand = max(demand, len(departures) * 8)

        # Add noise
        noise = float(self.rng.lognormal(0, 0.1))
        demand *= noise

        return {
            "point": max(1, int(demand)),
            "hour_factor": hour,
            "dow_factor": dow,
        }

    def _create_demand_forecast(
        self,
        station: str,
        forecast_time: datetime,
        generated_at: datetime,
        demand: dict,
        granularity: ForecastGranularity,
    ) -> DemandForecast:
        """Create a DemandForecast object."""
        point = demand["point"]
        std = max(1, point * 0.15)

        total_demand = QuantileForecast(
            q05=max(0, point - 1.645 * std),
            q25=max(0, point - 0.675 * std),
            q50=float(point),
            q75=point + 0.675 * std,
            q95=point + 1.645 * std,
        )

        demand_by_type = {}
        for uld_type, ratio in self.type_distribution.items():
            type_point = int(point * ratio)
            type_std = max(1, type_point * 0.2)
            demand_by_type[uld_type] = QuantileForecast(
                q05=max(0, type_point - 1.645 * type_std),
                q25=max(0, type_point - 0.675 * type_std),
                q50=float(type_point),
                q75=type_point + 0.675 * type_std,
                q95=type_point + 1.645 * type_std,
            )

        return DemandForecast(
            station=station,
            forecast_time=forecast_time,
            generated_at=generated_at,
            granularity=granularity,
            demand_by_type=demand_by_type,
            total_demand=total_demand,
            scheduled_departures=int(point / 8),
            confidence=ForecastConfidence.MEDIUM,
        )

    def _estimate_movements(
        self,
        station: str,
        forecast_time: datetime,
        schedule: FlightSchedule | None,
    ) -> tuple[int, int]:
        """Estimate arrivals and departures for a time period."""
        if schedule:
            # Count flights in the hour
            arrivals = sum(
                1 for f in schedule.flights
                if f.destination == station
                and f.is_widebody
                and abs((f.scheduled_arrival - forecast_time).total_seconds()) < 3600
            )
            departures = sum(
                1 for f in schedule.flights
                if f.origin == station
                and f.is_widebody
                and abs((f.scheduled_departure - forecast_time).total_seconds()) < 3600
            )
            return arrivals * 8, departures * 8

        # Default estimate based on hour
        hour = forecast_time.hour
        if 6 <= hour <= 10 or 15 <= hour <= 20:
            return 10, 10
        else:
            return 3, 3
