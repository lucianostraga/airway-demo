// Station types
export interface Station {
  code: string;
  name: string;
  city: string;
  country: string;
  hub_tier: number;
  timezone: string;
  coordinates: {
    latitude: number;
    longitude: number;
  };
  capacity: Record<string, number>;
}

// ULD types
export type ULDType = "AKE" | "AKH" | "PMC" | "LD3" | "LD7";
export type ULDStatus = "serviceable" | "in_use" | "empty" | "damaged" | "out_of_service";

export interface ULDPosition {
  uld_id: string;
  uld_type: ULDType;
  station: string;
  status: ULDStatus;
  last_update: string;
}

// Inventory types
export interface StationInventory {
  station: string;
  timestamp: string;
  inventory: Record<ULDType, Record<ULDStatus, number>>;
  total_count: number;
}

export interface NetworkInventory {
  timestamp: string;
  stations: Record<string, {
    total: number;
    available: number;
    availability_ratio: number;
  }>;
}

// Forecast types
export interface QuantileForecast {
  q05: number;
  q25: number;
  q50: number;
  q75: number;
  q95: number;
}

export interface DemandForecast {
  station: string;
  forecast_time: string;
  granularity: string;
  total_demand: QuantileForecast;
  demand_by_type: Record<ULDType, QuantileForecast>;
  confidence: string;
  is_anomaly: boolean;
}

export interface SupplyForecast {
  station: string;
  forecast_time: string;
  total_supply: QuantileForecast;
  expected_arrivals: number;
  expected_departures: number;
  confidence: string;
}

// Recommendation types
export type PriorityLevel = "critical" | "high" | "medium" | "low";

export interface CostBenefit {
  transportation_cost: number;
  handling_cost: number;
  total_cost: number;
  avoided_shortage_cost: number;
  revenue_protected: number;
  total_benefit: number;
  net_benefit: number;
  roi: number | null;
}

export interface RepositioningRecommendation {
  recommendation_id: string;
  priority: PriorityLevel;
  uld_type: ULDType;
  quantity: number;
  origin_station: string;
  destination_station: string;
  transport_method: string;
  recommended_departure: string;
  required_by: string;
  reason: string;
  shortage_probability: number;
  cost_benefit: CostBenefit;
}

export interface OptimizationResult {
  generated_at: string;
  solver_status: string;
  total_moves: number;
  total_ulds_moved: number;
  total_cost: number;
  solve_time_seconds: number;
  moves: RepositioningRecommendation[];
}

// Health check
export interface HealthStatus {
  status: string;
  timestamp: string;
  version: string;
  components: Record<string, { status: string; message?: string }>;
}
