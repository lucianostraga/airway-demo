"""
Bureau of Transportation Statistics (BTS) Data Downloader.

Downloads public aviation data from transtats.bts.gov:
- T-100 Domestic/International traffic data
- On-Time Performance data
- Air Carrier statistics

Data is free to use and authoritative (official US government source).
"""

import io
import zipfile
from datetime import date
from pathlib import Path
from typing import Literal

import httpx
import pandas as pd


class BTSDownloader:
    """
    Download and cache BTS aviation datasets.

    Example:
        downloader = BTSDownloader(cache_dir="./data/raw")

        # Download T-100 domestic segment data
        df = await downloader.download_t100_domestic(year=2023, month=12)

        # Download on-time performance
        df = await downloader.download_ontime_performance(year=2023, month=12)
    """

    # BTS TranStats API endpoints
    BASE_URL = "https://transtats.bts.gov"

    # Known table IDs for common datasets
    TABLE_IDS = {
        "t100_domestic_segment": "T_T100D_SEGMENT_US_CARRIER_ONLY",
        "t100_domestic_market": "T_T100D_MARKET_US_CARRIER_ONLY",
        "t100_international": "T_T100I_SEGMENT_ALL_CARRIER",
        "ontime_reporting": "T_ONTIME_REPORTING",
        "ontime_marketing": "T_ONTIME_MARKETING",
    }

    def __init__(
        self,
        cache_dir: str | Path = "./data/raw/bts",
        timeout: float = 120.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _get_cache_path(self, dataset: str, year: int, month: int | None = None) -> Path:
        """Get cache file path for a dataset."""
        if month:
            filename = f"{dataset}_{year}_{month:02d}.parquet"
        else:
            filename = f"{dataset}_{year}.parquet"
        return self.cache_dir / filename

    async def download_t100_domestic(
        self,
        year: int,
        month: int,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Download T-100 domestic segment data.

        Contains monthly traffic data by carrier, origin, destination:
        - Passengers
        - Freight (lbs)
        - Mail (lbs)
        - Departures performed
        - Aircraft type

        Args:
            year: Year (e.g., 2023)
            month: Month (1-12)
            force_refresh: Re-download even if cached

        Returns:
            DataFrame with T-100 segment data
        """
        cache_path = self._get_cache_path("t100_domestic", year, month)

        if cache_path.exists() and not force_refresh:
            return pd.read_parquet(cache_path)

        # Download from BTS
        df = await self._download_table(
            table_id=self.TABLE_IDS["t100_domestic_segment"],
            year=year,
            month=month,
        )

        if df is not None and not df.empty:
            df.to_parquet(cache_path)

        return df

    async def download_ontime_performance(
        self,
        year: int,
        month: int,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Download On-Time Performance data.

        Contains flight-level data including:
        - Scheduled and actual departure/arrival times
        - Delay causes (carrier, weather, NAS, security, late aircraft)
        - Cancellation and diversion info
        - Distance and air time

        Args:
            year: Year (e.g., 2023)
            month: Month (1-12)
            force_refresh: Re-download even if cached

        Returns:
            DataFrame with on-time performance data
        """
        cache_path = self._get_cache_path("ontime", year, month)

        if cache_path.exists() and not force_refresh:
            return pd.read_parquet(cache_path)

        df = await self._download_table(
            table_id=self.TABLE_IDS["ontime_reporting"],
            year=year,
            month=month,
        )

        if df is not None and not df.empty:
            # Select key columns to reduce size
            key_cols = [
                "YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK",
                "FL_DATE", "OP_CARRIER", "OP_CARRIER_FL_NUM",
                "ORIGIN", "DEST", "CRS_DEP_TIME", "DEP_TIME",
                "DEP_DELAY", "CRS_ARR_TIME", "ARR_TIME", "ARR_DELAY",
                "CANCELLED", "CANCELLATION_CODE", "DIVERTED",
                "CRS_ELAPSED_TIME", "ACTUAL_ELAPSED_TIME", "DISTANCE",
                "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
                "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
            ]
            available_cols = [c for c in key_cols if c in df.columns]
            df = df[available_cols]
            df.to_parquet(cache_path)

        return df

    async def download_carrier_statistics(
        self,
        year: int,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Download annual carrier summary statistics.

        Includes load factors, revenue passenger miles, etc.
        """
        cache_path = self._get_cache_path("carrier_stats", year)

        if cache_path.exists() and not force_refresh:
            return pd.read_parquet(cache_path)

        # This would download from BTS carrier statistics tables
        # For now, return empty DataFrame as placeholder
        return pd.DataFrame()

    async def _download_table(
        self,
        table_id: str,
        year: int,
        month: int,
    ) -> pd.DataFrame:
        """
        Download a table from BTS TranStats.

        Note: BTS uses a complex form-based download system.
        This is a simplified version that may need adjustment
        based on the actual BTS API behavior.
        """
        client = await self._get_client()

        # BTS download URL pattern
        # The actual URL format may vary - this is a common pattern
        download_url = (
            f"{self.BASE_URL}/DownLoad_Table.asp"
            f"?Table_ID={table_id}"
            f"&Has_Group=3"
            f"&UserTableName={table_id}"
        )

        try:
            # First, try direct CSV download
            csv_url = f"{self.BASE_URL}/PREZIP/{table_id}_{year}_{month}.zip"

            response = await client.get(csv_url)

            if response.status_code == 200:
                # Extract CSV from ZIP
                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
                    if csv_files:
                        with zf.open(csv_files[0]) as csv_file:
                            return pd.read_csv(csv_file, low_memory=False)

            # If direct download fails, return empty DataFrame
            # In production, would implement form-based download
            print(f"Warning: Could not download {table_id} for {year}-{month}")
            return pd.DataFrame()

        except Exception as e:
            print(f"Error downloading BTS data: {e}")
            return pd.DataFrame()

    async def get_delta_traffic(
        self,
        year: int,
        month: int,
    ) -> pd.DataFrame:
        """
        Get T-100 traffic data filtered for Delta Air Lines.

        Returns:
            DataFrame with Delta-only traffic data
        """
        df = await self.download_t100_domestic(year, month)

        if df.empty:
            return df

        # Filter for Delta (carrier codes: DL, Delta)
        carrier_col = "UNIQUE_CARRIER" if "UNIQUE_CARRIER" in df.columns else "CARRIER"
        if carrier_col in df.columns:
            df = df[df[carrier_col].isin(["DL", "DELTA"])]

        return df

    async def get_airport_traffic_summary(
        self,
        airport: str,
        year: int,
        month: int,
    ) -> dict:
        """
        Get traffic summary for an airport.

        Args:
            airport: IATA airport code (e.g., "ATL")
            year: Year
            month: Month

        Returns:
            Dict with traffic statistics
        """
        df = await self.download_t100_domestic(year, month)

        if df.empty:
            return {}

        # Get origin column name
        origin_col = "ORIGIN" if "ORIGIN" in df.columns else "ORIGIN_AIRPORT_ID"
        dest_col = "DEST" if "DEST" in df.columns else "DEST_AIRPORT_ID"

        # Filter for airport
        departures = df[df[origin_col] == airport.upper()]
        arrivals = df[df[dest_col] == airport.upper()]

        passengers_col = "PASSENGERS" if "PASSENGERS" in df.columns else "PAX"
        freight_col = "FREIGHT" if "FREIGHT" in df.columns else "FREIGHT_LBS"

        return {
            "airport": airport.upper(),
            "year": year,
            "month": month,
            "departing_passengers": departures[passengers_col].sum() if passengers_col in departures.columns else 0,
            "arriving_passengers": arrivals[passengers_col].sum() if passengers_col in arrivals.columns else 0,
            "departing_flights": len(departures),
            "arriving_flights": len(arrivals),
            "departing_freight_lbs": departures[freight_col].sum() if freight_col in departures.columns else 0,
            "arriving_freight_lbs": arrivals[freight_col].sum() if freight_col in arrivals.columns else 0,
        }


# Convenience function for quick data access
async def get_bts_data(
    dataset: Literal["t100", "ontime"],
    year: int,
    month: int,
    cache_dir: str = "./data/raw/bts",
) -> pd.DataFrame:
    """
    Quick access to BTS datasets.

    Example:
        df = await get_bts_data("ontime", 2023, 12)
    """
    downloader = BTSDownloader(cache_dir=cache_dir)
    try:
        if dataset == "t100":
            return await downloader.download_t100_domestic(year, month)
        elif dataset == "ontime":
            return await downloader.download_ontime_performance(year, month)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    finally:
        await downloader.close()
