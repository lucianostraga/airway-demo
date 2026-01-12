---
name: ml-patterns
description: Machine learning patterns for forecasting and optimization. Activates when working on ML models, training, or predictions.
allowed-tools: Read, Grep, Glob, Edit, Write, Bash
---

# ML Patterns for ULD Forecasting

## Forecasting Pipeline Structure

```python
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class ForecastConfig:
    horizon_days: int = 7
    train_window_days: int = 365
    validation_days: int = 30
    target_column: str = "uld_demand"

class DemandForecaster:
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        preprocessor = ColumnTransformer([
            ("numeric", StandardScaler(), ["passenger_count", "historical_avg"]),
            ("categorical", OneHotEncoder(), ["day_of_week", "season"]),
            ("cyclical", CyclicalEncoder(), ["month", "day_of_month"]),
        ])
        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(objective="reg:squarederror")),
        ])
```

## Feature Engineering

### Temporal Features
```python
def create_temporal_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[date_col])

    # Basic temporal
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = dt.dt.dayofweek >= 5

    # Cyclical encoding
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)

    # Lag features
    for lag in [1, 7, 14, 28]:
        df[f"demand_lag_{lag}"] = df["uld_demand"].shift(lag)

    # Rolling statistics
    for window in [7, 14, 28]:
        df[f"demand_roll_mean_{window}"] = (
            df["uld_demand"].rolling(window).mean()
        )
    return df
```

### Holiday Features
```python
from pandas.tseries.holiday import USFederalHolidayCalendar

def create_holiday_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start="2020-01-01", end="2030-12-31")

    df["is_holiday"] = df[date_col].isin(holidays)
    df["days_to_holiday"] = df[date_col].apply(
        lambda d: min((h - d).days for h in holidays if h >= d)
    )
    return df
```

## Model Evaluation

```python
from sklearn.metrics import mean_absolute_percentage_error

def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray,
) -> dict[str, float]:
    return {
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "bias": (y_pred - y_true).mean(),
        "coverage": np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper)),
        "interval_width": (y_pred_upper - y_pred_lower).mean(),
    }
```

## Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> list[dict]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        results.append({
            "mape": mean_absolute_percentage_error(y_val, y_pred),
            "val_size": len(val_idx),
        })
    return results
```

## Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| MAPE | < 15% | 7-day forecast accuracy |
| Bias | ±2% | Systematic over/under prediction |
| Coverage | > 80% | % actuals within prediction interval |
