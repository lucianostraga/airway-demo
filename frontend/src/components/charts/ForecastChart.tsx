import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { TrendingUp, Info } from "lucide-react";
import type { DemandForecast, SupplyForecast } from "@/types/api";

interface ForecastChartProps {
  demandForecast?: DemandForecast[];
  supplyForecast?: SupplyForecast[];
  loading?: boolean;
  station: string;
}

export function ForecastChart({
  demandForecast,
  supplyForecast,
  loading = false,
  station,
}: ForecastChartProps) {
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Demand Forecast
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-[300px] w-full" />
        </CardContent>
      </Card>
    );
  }

  // Combine demand and supply data
  const chartData =
    demandForecast?.map((f, i) => {
      const supplyPoint = supplyForecast?.[i];
      const timestamp = new Date(f.forecast_time);
      return {
        time: timestamp.toLocaleTimeString("en-US", {
          hour: "2-digit",
          minute: "2-digit",
        }),
        fullTime: timestamp.toLocaleString(),
        demand: f.total_demand.q50,
        demandLower: f.total_demand.q05,
        demandUpper: f.total_demand.q95,
        arrivals: supplyPoint?.expected_arrivals || 0,
        departures: supplyPoint?.expected_departures || 0,
        supply: supplyPoint?.total_supply.q50 || 0,
      };
    }) || [];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Demand & Supply Forecast - {station}
            <Tooltip>
              <TooltipTrigger asChild>
                <button type="button" className="inline-flex items-center focus:outline-none">
                  <Info className="h-3.5 w-3.5 text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))] cursor-help transition-colors" />
                </button>
              </TooltipTrigger>
              <TooltipContent>
                <p>ML-powered forecast showing predicted ULD demand and supply over next 24 hours with confidence intervals</p>
              </TooltipContent>
            </Tooltip>
          </span>
          {demandForecast && demandForecast.length > 0 && (
            <span className="text-xs font-normal text-[hsl(var(--muted-foreground))]">
              {demandForecast.length} forecast points
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {chartData.length === 0 ? (
          <div className="h-[300px] flex items-center justify-center text-[hsl(var(--muted-foreground))]">
            No forecast data available
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart
              data={chartData}
              margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
            >
              <defs>
                <linearGradient id="demandGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#c8102e" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#c8102e" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="supplyGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#0085ad" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#0085ad" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#c8102e" stopOpacity={0.1} />
                  <stop offset="95%" stopColor="#c8102e" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(var(--border))"
                opacity={0.5}
              />
              <XAxis
                dataKey="time"
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
                tickLine={false}
                axisLine={false}
                tickFormatter={(value) => `${value}`}
              />
              <RechartsTooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--popover))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                  fontSize: "12px",
                }}
                labelFormatter={(label, payload) =>
                  payload?.[0]?.payload?.fullTime || label
                }
              />
              <Legend />
              {/* Confidence interval */}
              <Area
                type="monotone"
                dataKey="demandUpper"
                stackId="confidence"
                stroke="transparent"
                fill="url(#confidenceGradient)"
                name="Upper Bound"
              />
              {/* Demand */}
              <Area
                type="monotone"
                dataKey="demand"
                stroke="#c8102e"
                strokeWidth={2}
                fill="url(#demandGradient)"
                name="Demand"
                dot={false}
                activeDot={{ r: 4, fill: "#c8102e" }}
              />
              {/* Supply */}
              <Area
                type="monotone"
                dataKey="supply"
                stroke="#0085ad"
                strokeWidth={2}
                fill="url(#supplyGradient)"
                name="Supply (Median)"
                dot={false}
                activeDot={{ r: 4, fill: "#0085ad" }}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
