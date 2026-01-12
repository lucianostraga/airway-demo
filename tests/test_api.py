"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health endpoints."""

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_health(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data


class TestStationsEndpoints:
    """Tests for stations endpoints."""

    def test_list_stations(self, client):
        """Test listing all stations."""
        response = client.get("/api/v1/stations/")

        assert response.status_code == 200
        data = response.json()
        assert "stations" in data
        assert "total" in data
        assert data["total"] > 0

    def test_list_stations_filter_by_tier(self, client):
        """Test filtering stations by tier."""
        response = client.get("/api/v1/stations/?tier=hub")

        assert response.status_code == 200
        data = response.json()
        assert all(s["tier"] == "hub" for s in data["stations"])

    def test_list_hubs(self, client):
        """Test listing hub stations."""
        response = client.get("/api/v1/stations/hubs")

        assert response.status_code == 200
        data = response.json()
        assert len(data["stations"]) == 4  # ATL, DTW, MSP, SLC

    def test_get_station(self, client):
        """Test getting a specific station."""
        response = client.get("/api/v1/stations/ATL")

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == "ATL"
        assert data["tier"] == "hub"

    def test_get_station_not_found(self, client):
        """Test getting unknown station."""
        response = client.get("/api/v1/stations/XXX")

        assert response.status_code == 404

    def test_get_station_connections(self, client):
        """Test getting station connections."""
        response = client.get("/api/v1/stations/ATL/connections")

        assert response.status_code == 200
        data = response.json()
        assert data["station"] == "ATL"
        assert "connections" in data
        assert data["total_connections"] > 0


class TestTrackingEndpoints:
    """Tests for tracking endpoints."""

    def test_get_position_not_found(self, client):
        """Test getting position for unknown ULD."""
        response = client.get("/api/v1/tracking/position/UNKNOWN123")

        assert response.status_code == 404

    def test_record_geolocation(self, client):
        """Test recording geolocation."""
        response = client.post(
            "/api/v1/tracking/position/geolocation",
            json={
                "uld_id": "AKEDL00001",
                "uld_type": "AKE",
                "station": "ATL",
                "confidence": 0.95,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["uld_id"] == "AKEDL00001"
        assert data["station"] == "ATL"

    def test_record_geolocation_invalid_type(self, client):
        """Test recording with invalid ULD type."""
        response = client.post(
            "/api/v1/tracking/position/geolocation",
            json={
                "uld_id": "AKEDL00001",
                "uld_type": "INVALID",
                "station": "ATL",
            },
        )

        assert response.status_code == 400

    def test_get_inventory(self, client):
        """Test getting station inventory."""
        response = client.get("/api/v1/tracking/inventory/ATL")

        assert response.status_code == 200
        data = response.json()
        assert data["station"] == "ATL"
        assert "total_count" in data

    def test_get_network_summary(self, client):
        """Test getting network summary."""
        response = client.get("/api/v1/tracking/network/summary")

        assert response.status_code == 200
        data = response.json()
        assert "stations" in data


class TestForecastingEndpoints:
    """Tests for forecasting endpoints."""

    def test_get_demand_forecast(self, client):
        """Test getting demand forecast."""
        response = client.get("/api/v1/forecasting/demand/ATL?hours_ahead=6")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 6
        assert all(f["station"] == "ATL" for f in data)

    def test_get_demand_forecast_daily(self, client):
        """Test getting daily demand forecast."""
        response = client.get(
            "/api/v1/forecasting/demand/ATL?hours_ahead=48&granularity=daily"
        )

        assert response.status_code == 200
        data = response.json()
        assert all(f["granularity"] == "daily" for f in data)

    def test_get_supply_forecast(self, client):
        """Test getting supply forecast."""
        response = client.get("/api/v1/forecasting/supply/ATL?hours_ahead=6")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 6

    def test_get_imbalance_forecast(self, client):
        """Test getting imbalance forecast."""
        response = client.get("/api/v1/forecasting/imbalance/ATL?hours_ahead=6")

        assert response.status_code == 200
        data = response.json()
        assert all("shortage_probability" in f for f in data)

    def test_get_network_forecast(self, client):
        """Test getting network forecast."""
        response = client.get("/api/v1/forecasting/network?hours_ahead=12")

        assert response.status_code == 200
        data = response.json()
        assert data["forecast_horizon_hours"] == 12
        assert "total_network_demand" in data
        assert "network_balanced" in data

    def test_get_priority_stations(self, client):
        """Test getting priority stations."""
        response = client.get("/api/v1/forecasting/network/priority?top_n=3")

        assert response.status_code == 200
        data = response.json()
        assert "priority_stations" in data


class TestRecommendationsEndpoints:
    """Tests for recommendations endpoints."""

    def test_get_repositioning_recommendations(self, client):
        """Test getting repositioning recommendations."""
        response = client.get("/api/v1/recommendations/repositioning?hours_ahead=12")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_station_plan(self, client):
        """Test getting station action plan."""
        response = client.get("/api/v1/recommendations/station/ATL/plan?hours_ahead=12")

        assert response.status_code == 200
        data = response.json()
        assert data["station"] == "ATL"
        assert "repositioning_out" in data
        assert "repositioning_in" in data

    def test_run_optimization(self, client):
        """Test running network optimization."""
        response = client.post(
            "/api/v1/recommendations/optimize?hours_ahead=12&max_moves_per_station=3"
        )

        assert response.status_code == 200
        data = response.json()
        assert "solver_status" in data
        assert "total_moves" in data
        assert "moves" in data

    def test_get_recommendations_summary(self, client):
        """Test getting recommendations summary."""
        response = client.get("/api/v1/recommendations/summary?hours_ahead=12")

        assert response.status_code == 200
        data = response.json()
        assert "total_recommendations" in data
        assert "estimated_total_cost" in data
