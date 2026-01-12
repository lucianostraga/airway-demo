"""
Data clients for external APIs.

This module provides a flexible abstraction layer for external data sources.
Implementations can be swapped between free and paid APIs without changing business logic.

Free APIs (current):
- AviationStack (100 req/month)
- NOAA Aviation Weather (unlimited)
- Open-Meteo (unlimited)

Paid APIs (future):
- FlightAware AeroAPI
- PredictHQ Events
- Tomorrow.io Weather
"""

from .base import DataClient, FlightDataClient, WeatherDataClient, EventsDataClient
from .aviation_stack import AviationStackClient
from .noaa_weather import NOAAWeatherClient
from .open_meteo import OpenMeteoClient

__all__ = [
    "DataClient",
    "FlightDataClient",
    "WeatherDataClient",
    "EventsDataClient",
    "AviationStackClient",
    "NOAAWeatherClient",
    "OpenMeteoClient",
]
