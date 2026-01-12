"""
NOAA Aviation Weather API Client.

Free, authoritative source for US aviation weather data.
Provides METAR observations and TAF forecasts.

API Documentation: https://aviationweather.gov/data/api/

Rate Limits: Reasonable use (no strict limits documented)
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx

from .base import (
    DataClientFactory,
    WeatherDataClient,
    WeatherForecast,
    WeatherObservation,
)


class NOAAWeatherClient(WeatherDataClient):
    """
    NOAA Aviation Weather API client.

    Provides METAR (current observations) and TAF (terminal area forecasts)
    for aviation weather at airports.

    Example:
        client = NOAAWeatherClient()
        metar = await client.get_current_weather("KATL")
        print(f"ATL conditions: {metar.flight_category}")
    """

    BASE_URL = "https://aviationweather.gov/api/data"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"Accept": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def health_check(self) -> bool:
        """Check if NOAA API is available."""
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.BASE_URL}/metar",
                params={"ids": "KATL", "format": "json"},
            )
            return response.status_code == 200
        except Exception:
            return False

    async def get_current_weather(self, station: str) -> WeatherObservation | None:
        """
        Get current METAR observation for a station.

        Args:
            station: ICAO airport code (e.g., "KATL" for Atlanta)

        Returns:
            WeatherObservation or None if not available
        """
        client = await self._get_client()

        try:
            response = await client.get(
                f"{self.BASE_URL}/metar",
                params={
                    "ids": station.upper(),
                    "format": "json",
                    "taf": "false",
                },
            )
            response.raise_for_status()
            data = response.json()

            if not data:
                return None

            metar = data[0] if isinstance(data, list) else data
            return self._parse_metar(metar)

        except httpx.HTTPError:
            return None

    async def get_forecast(
        self,
        station: str,
        hours_ahead: int = 24,
    ) -> list[WeatherForecast]:
        """
        Get TAF forecast for a station.

        Args:
            station: ICAO airport code
            hours_ahead: Hours of forecast to retrieve (max ~30)

        Returns:
            List of WeatherForecast objects
        """
        client = await self._get_client()

        try:
            response = await client.get(
                f"{self.BASE_URL}/taf",
                params={
                    "ids": station.upper(),
                    "format": "json",
                },
            )
            response.raise_for_status()
            data = response.json()

            if not data:
                return []

            taf = data[0] if isinstance(data, list) else data
            return self._parse_taf(taf, hours_ahead)

        except httpx.HTTPError:
            return []

    async def get_aviation_weather(
        self,
        station: str,
    ) -> tuple[WeatherObservation | None, list[WeatherForecast]]:
        """
        Get both METAR and TAF for a station in a single call.

        Args:
            station: ICAO airport code

        Returns:
            Tuple of (current observation, list of forecasts)
        """
        # Fetch both in parallel
        metar_task = self.get_current_weather(station)
        taf_task = self.get_forecast(station)

        metar, taf = await asyncio.gather(metar_task, taf_task)
        return metar, taf

    async def get_multiple_stations(
        self,
        stations: list[str],
    ) -> dict[str, WeatherObservation | None]:
        """
        Get METAR for multiple stations efficiently.

        Args:
            stations: List of ICAO codes

        Returns:
            Dict mapping station code to observation
        """
        client = await self._get_client()

        try:
            # NOAA API accepts comma-separated station IDs
            ids = ",".join(s.upper() for s in stations)
            response = await client.get(
                f"{self.BASE_URL}/metar",
                params={
                    "ids": ids,
                    "format": "json",
                },
            )
            response.raise_for_status()
            data = response.json()

            results: dict[str, WeatherObservation | None] = {s: None for s in stations}
            for metar in data:
                obs = self._parse_metar(metar)
                if obs:
                    results[obs.station] = obs

            return results

        except httpx.HTTPError:
            return {s: None for s in stations}

    def _parse_metar(self, data: dict[str, Any]) -> WeatherObservation | None:
        """Parse NOAA METAR JSON into WeatherObservation."""
        try:
            # Parse observation time
            obs_time_str = data.get("obsTime") or data.get("reportTime")
            if obs_time_str:
                obs_time = datetime.fromisoformat(obs_time_str.replace("Z", "+00:00"))
            else:
                obs_time = datetime.now(timezone.utc)

            return WeatherObservation(
                station=data.get("icaoId", data.get("stationId", "UNKN")),
                observed_at=obs_time,
                temperature_c=data.get("temp"),
                wind_speed_kts=data.get("wspd"),
                wind_direction=data.get("wdir"),
                visibility_miles=data.get("visib"),
                ceiling_ft=self._extract_ceiling(data),
                flight_category=data.get("fltcat"),
                raw_metar=data.get("rawOb"),
            )
        except Exception:
            return None

    def _parse_taf(
        self,
        data: dict[str, Any],
        hours_ahead: int,
    ) -> list[WeatherForecast]:
        """Parse NOAA TAF JSON into list of WeatherForecast."""
        forecasts = []

        try:
            station = data.get("icaoId", data.get("stationId", "UNKN"))
            raw_taf = data.get("rawTAF")

            # Parse forecast periods
            for period in data.get("fcsts", []):
                try:
                    valid_from = datetime.fromisoformat(
                        period.get("timeFrom", "").replace("Z", "+00:00")
                    )
                    valid_to = datetime.fromisoformat(
                        period.get("timeTo", "").replace("Z", "+00:00")
                    )

                    forecasts.append(
                        WeatherForecast(
                            station=station,
                            valid_from=valid_from,
                            valid_to=valid_to,
                            wind_speed_kts=period.get("wspd"),
                            conditions=self._extract_conditions(period),
                            flight_category=period.get("fltcat"),
                            raw_taf=raw_taf,
                        )
                    )
                except Exception:
                    continue

        except Exception:
            pass

        return forecasts[:hours_ahead]

    def _extract_ceiling(self, data: dict[str, Any]) -> int | None:
        """Extract ceiling from cloud layers."""
        clouds = data.get("clouds", [])
        for cloud in clouds:
            cover = cloud.get("cover", "")
            if cover in ("BKN", "OVC"):  # Broken or Overcast = ceiling
                return cloud.get("base")
        return None

    def _extract_conditions(self, data: dict[str, Any]) -> str | None:
        """Extract weather conditions string."""
        wx = data.get("wxString") or data.get("wx")
        if wx:
            return wx
        # Derive from visibility and ceiling
        visib = data.get("visib", 10)
        if visib < 1:
            return "fog"
        elif visib < 3:
            return "mist"
        return "clear"


# Register with factory
DataClientFactory.register_weather_client("noaa", NOAAWeatherClient)
