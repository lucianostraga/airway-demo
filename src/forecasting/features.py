"""
Feature Engineering for ULD Demand Forecasting

Implements the temporal, lag, rolling, and external signal features
defined in the mathematical framework.

Mathematical foundation:
- Cyclical encoding: sin(2*pi*t/T), cos(2*pi*t/T)
- Fourier basis: sum of harmonics for flexible seasonality
- Exponential weighted moving average: alpha * x_t + (1-alpha) * EWMA_{t-1}
- Information-theoretic lag selection: Conditional Mutual Information
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline."""

    # Lag features
    lag_days: list[int] = field(
        default_factory=lambda: [1, 2, 3, 7, 14, 21, 28, 364, 365, 371]
    )

    # Rolling window sizes
    rolling_windows: list[int] = field(default_factory=lambda: [7, 14, 28])

    # EWMA spans
    ewma_spans: list[int] = field(default_factory=lambda: [7, 28])

    # Fourier terms for seasonality
    fourier_terms_weekly: int = 2  # Number of harmonics for weekly
    fourier_terms_annual: int = 3  # Number of harmonics for annual

    # Holiday lookahead/lookback days
    holiday_window: int = 7


class FeatureEngineer:
    """
    Feature engineering pipeline for ULD demand forecasting.

    Creates features in the following categories:
    1. Temporal (day of week, month, cyclical encodings)
    2. Lag features (demand from previous periods)
    3. Rolling statistics (mean, std, min, max, EWMA)
    4. Holiday features (is_holiday, days_to_holiday)
    5. Fourier features (flexible seasonality)
    6. External signals (weather, events) - when available
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._holiday_calendar = USFederalHolidayCalendar()
        self._holidays: Optional[pd.DatetimeIndex] = None

    def fit(self, df: pd.DataFrame, date_col: str = "date") -> "FeatureEngineer":
        """
        Fit the feature engineer to the data.

        Learns:
        - Date range for holiday computation
        - Optional: feature statistics for normalization
        """
        dates = pd.to_datetime(df[date_col])
        start_date = dates.min() - pd.Timedelta(days=30)
        end_date = dates.max() + pd.Timedelta(days=365)
        self._holidays = self._holiday_calendar.holidays(
            start=start_date, end=end_date
        )
        return self

    def transform(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        target_col: str = "uld_demand",
    ) -> pd.DataFrame:
        """
        Generate all features for the input data.

        Args:
            df: Input DataFrame with date and target columns
            date_col: Name of the date column
            target_col: Name of the target variable for lag features

        Returns:
            DataFrame with all engineered features
        """
        result = df.copy()
        dt = pd.to_datetime(result[date_col])

        # 1. Temporal features
        result = self._add_temporal_features(result, dt)

        # 2. Cyclical encodings
        result = self._add_cyclical_features(result, dt)

        # 3. Fourier features
        result = self._add_fourier_features(result, dt)

        # 4. Lag features
        if target_col in result.columns:
            result = self._add_lag_features(result, target_col)
            result = self._add_rolling_features(result, target_col)

        # 5. Holiday features
        result = self._add_holiday_features(result, dt)

        return result

    def _add_temporal_features(
        self, df: pd.DataFrame, dt: pd.Series
    ) -> pd.DataFrame:
        """Add basic temporal features."""
        df = df.copy()
        df["day_of_week"] = dt.dt.dayofweek
        df["day_of_month"] = dt.dt.day
        df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
        df["month"] = dt.dt.month
        df["quarter"] = dt.dt.quarter
        df["year"] = dt.dt.year
        df["is_weekend"] = dt.dt.dayofweek >= 5
        df["is_month_start"] = dt.dt.is_month_start
        df["is_month_end"] = dt.dt.is_month_end
        return df

    def _add_cyclical_features(
        self, df: pd.DataFrame, dt: pd.Series
    ) -> pd.DataFrame:
        """
        Add cyclical encodings using sine/cosine transformation.

        Maps periodic features to unit circle to preserve distance:
        - Day 0 (Monday) is close to Day 6 (Sunday)
        - December is close to January

        Formula: sin(2*pi*t/T), cos(2*pi*t/T)
        """
        df = df.copy()

        # Day of week (period = 7)
        df["day_of_week_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

        # Month (period = 12)
        df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)

        # Day of year (period = 365.25)
        df["day_of_year_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
        df["day_of_year_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)

        return df

    def _add_fourier_features(
        self, df: pd.DataFrame, dt: pd.Series
    ) -> pd.DataFrame:
        """
        Add Fourier basis features for flexible seasonality modeling.

        Fourier series: sum_{k=1}^{K} [a_k * sin(2*pi*k*t/T) + b_k * cos(2*pi*k*t/T)]

        The coefficients a_k, b_k are learned by the model.
        K controls smoothness (higher K = more flexible but risk overfitting).
        """
        df = df.copy()

        # Weekly seasonality (T = 7)
        day_of_week = dt.dt.dayofweek.values
        for k in range(1, self.config.fourier_terms_weekly + 1):
            df[f"fourier_weekly_{k}_sin"] = np.sin(2 * np.pi * k * day_of_week / 7)
            df[f"fourier_weekly_{k}_cos"] = np.cos(2 * np.pi * k * day_of_week / 7)

        # Annual seasonality (T = 365.25)
        day_of_year = dt.dt.dayofyear.values
        for k in range(1, self.config.fourier_terms_annual + 1):
            df[f"fourier_annual_{k}_sin"] = np.sin(
                2 * np.pi * k * day_of_year / 365.25
            )
            df[f"fourier_annual_{k}_cos"] = np.cos(
                2 * np.pi * k * day_of_year / 365.25
            )

        return df

    def _add_lag_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Add lag features for the target variable.

        Standard lags:
        - Short-term: 1, 2, 3, 7 (day-over-day, week-over-week)
        - Medium-term: 14, 21, 28 (bi-weekly patterns)
        - Long-term: 364, 365, 371 (year-over-year, same weekday)
        """
        df = df.copy()
        for lag in self.config.lag_days:
            df[f"demand_lag_{lag}"] = df[target_col].shift(lag)
        return df

    def _add_rolling_features(
        self, df: pd.DataFrame, target_col: str
    ) -> pd.DataFrame:
        """
        Add rolling window statistics and exponential weighted moving averages.

        Rolling statistics capture recent trends and volatility.
        EWMA gives more weight to recent observations.

        EWMA formula: alpha * x_t + (1-alpha) * EWMA_{t-1}
        where alpha = 2 / (span + 1)
        """
        df = df.copy()

        for window in self.config.rolling_windows:
            rolling = df[target_col].rolling(window=window, min_periods=1)
            df[f"demand_roll_mean_{window}"] = rolling.mean()
            df[f"demand_roll_std_{window}"] = rolling.std()
            df[f"demand_roll_min_{window}"] = rolling.min()
            df[f"demand_roll_max_{window}"] = rolling.max()

            # Coefficient of variation (normalized volatility)
            df[f"demand_roll_cv_{window}"] = (
                df[f"demand_roll_std_{window}"] / df[f"demand_roll_mean_{window}"]
            ).fillna(0)

        # Exponential weighted moving averages
        for span in self.config.ewma_spans:
            df[f"demand_ewm_{span}"] = (
                df[target_col].ewm(span=span, min_periods=1).mean()
            )

        return df

    def _add_holiday_features(
        self, df: pd.DataFrame, dt: pd.Series
    ) -> pd.DataFrame:
        """
        Add holiday-related features.

        Includes:
        - is_holiday: Binary indicator
        - days_to_holiday: Days until next holiday
        - days_from_holiday: Days since last holiday
        - is_holiday_eve: Day before holiday
        """
        df = df.copy()

        if self._holidays is None:
            raise ValueError("Must call fit() before transform()")

        holidays_set = set(self._holidays.date)

        # Is holiday
        df["is_holiday"] = dt.dt.date.isin(holidays_set).astype(int)

        # Is holiday eve (day before)
        holiday_eves = set((h - pd.Timedelta(days=1)).date() for h in self._holidays)
        df["is_holiday_eve"] = dt.dt.date.isin(holiday_eves).astype(int)

        # Days to next holiday and from last holiday
        days_to = []
        days_from = []
        holidays_sorted = sorted(self._holidays)

        for date in dt:
            # Days to next holiday
            future = [h for h in holidays_sorted if h >= date]
            if future:
                days_to.append((future[0] - date).days)
            else:
                days_to.append(365)  # Default if no future holiday

            # Days from last holiday
            past = [h for h in holidays_sorted if h <= date]
            if past:
                days_from.append((date - past[-1]).days)
            else:
                days_from.append(365)  # Default if no past holiday

        df["days_to_holiday"] = days_to
        df["days_from_holiday"] = days_from

        return df


class ExternalSignalProcessor:
    """
    Process external signals (weather, events) for feature engineering.

    Weather Severity Index (composite):
    W_{s,t} = w_1 * I[IFR] + w_2 * I[SIGMET] + w_3 * (wind_speed / threshold)

    Event Impact Score:
    E_{s,t} = sum_{e} attendance_e * category_weight_e * decay(distance_e)
    """

    # Category weights for different event types (empirically calibrated)
    EVENT_CATEGORY_WEIGHTS = {
        "sports": 1.0,
        "concerts": 0.9,
        "conferences": 0.7,
        "festivals": 0.8,
        "community": 0.3,
        "expos": 0.6,
        "performing-arts": 0.5,
    }

    # Weather severity weights
    WEATHER_WEIGHTS = {
        "ifr_conditions": 3.0,
        "sigmet_active": 2.0,
        "wind_normalized": 1.0,
    }

    def __init__(self, distance_decay_sigma: float = 50.0):
        """
        Args:
            distance_decay_sigma: Decay constant for distance (in km)
        """
        self.distance_decay_sigma = distance_decay_sigma

    def compute_weather_severity(
        self,
        is_ifr: bool,
        is_sigmet: bool,
        wind_speed: float,
        wind_threshold: float = 25.0,
    ) -> float:
        """
        Compute composite weather severity index.

        Args:
            is_ifr: Instrument Flight Rules conditions
            is_sigmet: Significant Meteorological Information active
            wind_speed: Current wind speed (knots)
            wind_threshold: Wind speed threshold for significant impact

        Returns:
            Weather severity score (0 = clear, higher = more severe)
        """
        w = self.WEATHER_WEIGHTS
        severity = (
            w["ifr_conditions"] * int(is_ifr)
            + w["sigmet_active"] * int(is_sigmet)
            + w["wind_normalized"] * min(wind_speed / wind_threshold, 2.0)
        )
        return severity

    def compute_event_impact(
        self,
        events: list[dict],
        station_lat: float,
        station_lon: float,
    ) -> float:
        """
        Compute aggregate event impact score for a station.

        Uses exponential distance decay: exp(-d / sigma)

        Args:
            events: List of events with 'attendance', 'category', 'lat', 'lon'
            station_lat: Station latitude
            station_lon: Station longitude

        Returns:
            Aggregate event impact score
        """
        total_impact = 0.0

        for event in events:
            # Great circle distance (simplified for small distances)
            dlat = event["lat"] - station_lat
            dlon = event["lon"] - station_lon
            distance_km = np.sqrt(dlat**2 + dlon**2) * 111.0  # Approx km per degree

            # Distance decay
            decay = np.exp(-distance_km / self.distance_decay_sigma)

            # Category weight
            category = event.get("category", "community")
            cat_weight = self.EVENT_CATEGORY_WEIGHTS.get(category, 0.3)

            # Attendance impact
            attendance = event.get("attendance", 0)

            total_impact += attendance * cat_weight * decay

        # Normalize to reasonable scale
        return np.log1p(total_impact / 1000)


class NetworkFeatureBuilder:
    """
    Build cross-station and network-level features.

    Hub Pressure: Weighted imbalance at connected hubs
    Network Imbalance: Inflow - Outflow for a station
    """

    def __init__(self, station_connectivity: dict[str, list[str]]):
        """
        Args:
            station_connectivity: Mapping of station -> connected hubs
        """
        self.connectivity = station_connectivity

    def compute_hub_pressure(
        self,
        station: str,
        hub_demand: dict[str, float],
        hub_baseline: dict[str, float],
        hub_std: dict[str, float],
    ) -> float:
        """
        Compute hub pressure metric.

        HubPressure = sum_h ((D_h - baseline_h) / std_h) * connectivity(s, h)

        When hubs are stressed, pressure propagates to connected stations.
        """
        connected_hubs = self.connectivity.get(station, [])
        if not connected_hubs:
            return 0.0

        pressure = 0.0
        for hub in connected_hubs:
            if hub in hub_demand and hub in hub_baseline and hub in hub_std:
                z_score = (hub_demand[hub] - hub_baseline[hub]) / max(hub_std[hub], 1.0)
                pressure += z_score

        return pressure / len(connected_hubs)

    def compute_network_imbalance(
        self,
        station: str,
        inflows: dict[str, float],
        outflows: dict[str, float],
    ) -> float:
        """
        Compute net flow imbalance for a station.

        Imbalance = Inflow - Outflow

        Positive: More ULDs arriving than leaving (potential surplus)
        Negative: More ULDs leaving than arriving (potential shortage)
        """
        inflow = inflows.get(station, 0.0)
        outflow = outflows.get(station, 0.0)
        return inflow - outflow
