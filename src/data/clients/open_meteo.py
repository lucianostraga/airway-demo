"""
Open-Meteo Weather API Client.

Free, open-source weather API with no API key required.
Provides current weather and forecasts up to 16 days.

API Documentation: https://open-meteo.com/

Rate Limits: 10,000 requests/day (generous for free tier)
"""

from datetime import date, datetime, timezone
from typing import Any

import httpx

from .base import (
    DataClientFactory,
    WeatherDataClient,
    WeatherForecast,
    WeatherObservation,
)


# Mapping of IATA codes to coordinates for major Delta airports
AIRPORT_COORDINATES: dict[str, tuple[float, float]] = {
    # Delta Hubs
    "KATL": (33.6407, -84.4277),  # Atlanta
    "KDTW": (42.2124, -83.3534),  # Detroit
    "KMSP": (44.8820, -93.2218),  # Minneapolis
    "KSLC": (40.7884, -111.9778),  # Salt Lake City
    # Focus Cities
    "KBOS": (42.3656, -71.0096),  # Boston
    "KJFK": (40.6413, -73.7781),  # New York JFK
    "KLAX": (33.9416, -118.4085),  # Los Angeles
    "KSEA": (47.4502, -122.3088),  # Seattle
    "KLGA": (40.7769, -73.8740),  # New York LaGuardia
    "KDCA": (38.8512, -77.0402),  # Washington DCA
    "KEWR": (40.6895, -74.1745),  # Newark
    "KORD": (41.9742, -87.9073),  # Chicago O'Hare
    "KSFO": (37.6213, -122.3790),  # San Francisco
    "KMIA": (25.7959, -80.2870),  # Miami
    "KDFW": (32.8998, -97.0403),  # Dallas
    "KDEN": (39.8561, -104.6737),  # Denver
    "KPHX": (33.4373, -112.0078),  # Phoenix
    # International
    "EGLL": (51.4700, -0.4543),  # London Heathrow
    "LFPG": (49.0097, 2.5479),  # Paris CDG
    "EHAM": (52.3105, 4.7683),  # Amsterdam
}


class OpenMeteoClient(WeatherDataClient):
    """
    Open-Meteo API client for general weather forecasts.

    Complements NOAA aviation weather with:
    - Global coverage (not just US)
    - Extended forecast horizon (16 days)
    - Precipitation probability
    - Temperature forecasts

    Example:
        client = OpenMeteoClient()
        forecasts = await client.get_forecast("KATL", hours_ahead=48)
    """

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def health_check(self) -> bool:
        """Check if Open-Meteo API is available."""
        try:
            client = await self._get_client()
            response = await client.get(
                self.BASE_URL,
                params={
                    "latitude": 33.64,
                    "longitude": -84.43,
                    "current_weather": "true",
                },
            )
            return response.status_code == 200
        except Exception:
            return False

    def _get_coordinates(self, station: str) -> tuple[float, float] | None:
        """Get coordinates for an airport station."""
        return AIRPORT_COORDINATES.get(station.upper())

    async def get_current_weather(self, station: str) -> WeatherObservation | None:
        """
        Get current weather for a station.

        Args:
            station: ICAO airport code (e.g., "KATL")

        Returns:
            WeatherObservation or None
        """
        coords = self._get_coordinates(station)
        if not coords:
            return None

        client = await self._get_client()

        try:
            response = await client.get(
                self.BASE_URL,
                params={
                    "latitude": coords[0],
                    "longitude": coords[1],
                    "current_weather": "true",
                    "windspeed_unit": "kn",
                    "timezone": "UTC",
                },
            )
            response.raise_for_status()
            data = response.json()

            current = data.get("current_weather", {})
            return WeatherObservation(
                station=station.upper(),
                observed_at=datetime.fromisoformat(current["time"]).replace(
                    tzinfo=timezone.utc
                ),
                temperature_c=current.get("temperature"),
                wind_speed_kts=int(current.get("windspeed", 0)),
                wind_direction=int(current.get("winddirection", 0)),
                visibility_miles=None,  # Not available in Open-Meteo
                ceiling_ft=None,  # Not available
                flight_category=self._estimate_flight_category(current),
                raw_metar=None,
            )

        except httpx.HTTPError:
            return None

    async def get_forecast(
        self,
        station: str,
        hours_ahead: int = 24,
    ) -> list[WeatherForecast]:
        """
        Get hourly forecast for a station.

        Args:
            station: ICAO airport code
            hours_ahead: Number of hours to forecast (max 384 = 16 days)

        Returns:
            List of WeatherForecast objects
        """
        coords = self._get_coordinates(station)
        if not coords:
            return []

        client = await self._get_client()

        try:
            response = await client.get(
                self.BASE_URL,
                params={
                    "latitude": coords[0],
                    "longitude": coords[1],
                    "hourly": "temperature_2m,precipitation_probability,windspeed_10m,weathercode",
                    "windspeed_unit": "kn",
                    "timezone": "UTC",
                    "forecast_days": min((hours_ahead // 24) + 1, 16),
                },
            )
            response.raise_for_status()
            data = response.json()

            return self._parse_hourly_forecast(station, data, hours_ahead)

        except httpx.HTTPError:
            return []

    async def get_aviation_weather(
        self,
        station: str,
    ) -> tuple[WeatherObservation | None, list[WeatherForecast]]:
        """Get current and forecast weather."""
        current = await self.get_current_weather(station)
        forecast = await self.get_forecast(station, hours_ahead=24)
        return current, forecast

    async def get_extended_forecast(
        self,
        station: str,
        days: int = 7,
    ) -> list[WeatherForecast]:
        """
        Get daily forecast for extended planning.

        Args:
            station: ICAO airport code
            days: Number of days (max 16)

        Returns:
            List of daily WeatherForecast objects
        """
        coords = self._get_coordinates(station)
        if not coords:
            return []

        client = await self._get_client()

        try:
            response = await client.get(
                self.BASE_URL,
                params={
                    "latitude": coords[0],
                    "longitude": coords[1],
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,windspeed_10m_max,weathercode",
                    "windspeed_unit": "kn",
                    "timezone": "UTC",
                    "forecast_days": min(days, 16),
                },
            )
            response.raise_for_status()
            data = response.json()

            return self._parse_daily_forecast(station, data)

        except httpx.HTTPError:
            return []

    def _parse_hourly_forecast(
        self,
        station: str,
        data: dict[str, Any],
        hours_ahead: int,
    ) -> list[WeatherForecast]:
        """Parse Open-Meteo hourly forecast response."""
        forecasts = []
        hourly = data.get("hourly", {})

        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        precip = hourly.get("precipitation_probability", [])
        winds = hourly.get("windspeed_10m", [])
        codes = hourly.get("weathercode", [])

        for i in range(min(len(times), hours_ahead)):
            try:
                valid_time = datetime.fromisoformat(times[i]).replace(tzinfo=timezone.utc)
                valid_end = valid_time.replace(hour=valid_time.hour + 1)

                forecasts.append(
                    WeatherForecast(
                        station=station.upper(),
                        valid_from=valid_time,
                        valid_to=valid_end,
                        temperature_c=temps[i] if i < len(temps) else None,
                        precipitation_probability=precip[i] / 100 if i < len(precip) else None,
                        wind_speed_kts=int(winds[i]) if i < len(winds) else None,
                        conditions=self._weather_code_to_condition(
                            codes[i] if i < len(codes) else 0
                        ),
                        flight_category=None,
                        raw_taf=None,
                    )
                )
            except Exception:
                continue

        return forecasts

    def _parse_daily_forecast(
        self,
        station: str,
        data: dict[str, Any],
    ) -> list[WeatherForecast]:
        """Parse Open-Meteo daily forecast response."""
        forecasts = []
        daily = data.get("daily", {})

        times = daily.get("time", [])
        temp_max = daily.get("temperature_2m_max", [])
        temp_min = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_probability_max", [])
        winds = daily.get("windspeed_10m_max", [])
        codes = daily.get("weathercode", [])

        for i in range(len(times)):
            try:
                day = date.fromisoformat(times[i])
                valid_from = datetime.combine(day, datetime.min.time()).replace(
                    tzinfo=timezone.utc
                )
                valid_to = datetime.combine(day, datetime.max.time()).replace(
                    tzinfo=timezone.utc
                )

                # Use average of min/max temp
                temp = None
                if i < len(temp_max) and i < len(temp_min):
                    temp = (temp_max[i] + temp_min[i]) / 2

                forecasts.append(
                    WeatherForecast(
                        station=station.upper(),
                        valid_from=valid_from,
                        valid_to=valid_to,
                        temperature_c=temp,
                        precipitation_probability=precip[i] / 100 if i < len(precip) else None,
                        wind_speed_kts=int(winds[i]) if i < len(winds) else None,
                        conditions=self._weather_code_to_condition(
                            codes[i] if i < len(codes) else 0
                        ),
                        flight_category=None,
                        raw_taf=None,
                    )
                )
            except Exception:
                continue

        return forecasts

    def _weather_code_to_condition(self, code: int) -> str:
        """Convert WMO weather code to condition string."""
        # WMO Weather interpretation codes
        # https://open-meteo.com/en/docs
        conditions = {
            0: "clear",
            1: "mainly_clear",
            2: "partly_cloudy",
            3: "overcast",
            45: "fog",
            48: "fog",
            51: "drizzle",
            53: "drizzle",
            55: "drizzle",
            61: "rain",
            63: "rain",
            65: "heavy_rain",
            71: "snow",
            73: "snow",
            75: "heavy_snow",
            77: "snow_grains",
            80: "rain_showers",
            81: "rain_showers",
            82: "heavy_rain_showers",
            85: "snow_showers",
            86: "heavy_snow_showers",
            95: "thunderstorm",
            96: "thunderstorm_hail",
            99: "thunderstorm_hail",
        }
        return conditions.get(code, "unknown")

    def _estimate_flight_category(self, current: dict[str, Any]) -> str | None:
        """Estimate flight category from weather code."""
        code = current.get("weathercode", 0)
        if code in (45, 48):  # Fog
            return "IFR"
        elif code in (95, 96, 99):  # Thunderstorm
            return "IFR"
        elif code in (65, 75, 82, 86):  # Heavy precip
            return "MVFR"
        elif code >= 51:  # Any precip
            return "MVFR"
        else:
            return "VFR"


# Register with factory
DataClientFactory.register_weather_client("open_meteo", OpenMeteoClient)
