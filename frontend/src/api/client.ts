import axios from "axios";
import type {
  Station,
  NetworkInventory,
  DemandForecast,
  SupplyForecast,
  RepositioningRecommendation,
  OptimizationResult,
  HealthStatus,
} from "@/types/api";

// Create axios instance with base configuration
const api = axios.create({
  baseURL: "/",
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 30000,
});

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error("API Error:", error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Health endpoint
export async function getHealth(): Promise<HealthStatus> {
  const response = await api.get<HealthStatus>("/health");
  return response.data;
}

// Station endpoints
export async function getStations(): Promise<Station[]> {
  const response = await api.get<{ stations: Station[]; total: number }>("/api/v1/stations/");
  return response.data.stations;
}

export async function getHubStations(): Promise<Station[]> {
  const response = await api.get<{ stations: Station[]; total: number }>("/api/v1/stations/hubs");
  return response.data.stations;
}

// Tracking endpoints
export async function getNetworkInventory(): Promise<NetworkInventory> {
  const response = await api.get<NetworkInventory>("/api/v1/tracking/network/summary");
  return response.data;
}

// Forecasting endpoints
export async function getDemandForecast(
  station: string,
  hoursAhead: number = 24
): Promise<DemandForecast[]> {
  const response = await api.get<DemandForecast[]>(
    `/api/v1/forecasting/demand/${station}`,
    {
      params: { hours_ahead: hoursAhead },
    }
  );
  return response.data;
}

export async function getSupplyForecast(
  station: string,
  hoursAhead: number = 24
): Promise<SupplyForecast[]> {
  const response = await api.get<SupplyForecast[]>(
    `/api/v1/forecasting/supply/${station}`,
    {
      params: { hours_ahead: hoursAhead },
    }
  );
  return response.data;
}

// Recommendation endpoints
export async function getRepositioningRecommendations(): Promise<
  RepositioningRecommendation[]
> {
  const response = await api.get<RepositioningRecommendation[]>(
    "/api/v1/recommendations/repositioning"
  );
  return response.data;
}

export async function runOptimization(): Promise<OptimizationResult> {
  const response = await api.post<OptimizationResult>(
    "/api/v1/recommendations/optimize"
  );
  return response.data;
}

export default api;
