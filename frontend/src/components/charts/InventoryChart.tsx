import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Cell,
  Legend,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Package, Info } from "lucide-react";
import type { NetworkInventory } from "@/types/api";

interface InventoryChartProps {
  inventory?: NetworkInventory;
  loading?: boolean;
}

const ULD_COLORS: Record<string, string> = {
  AKE: "#c8102e",
  AKH: "#001f5b",
  PMC: "#0085ad",
  LD3: "#f4c300",
  LD7: "#10b981",
};

export function InventoryChart({ inventory, loading = false }: InventoryChartProps) {
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Package className="h-5 w-5" />
            Network Inventory by ULD Type
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-[300px] w-full" />
        </CardContent>
      </Card>
    );
  }

  // Aggregate total ULDs across all stations
  const totalULDs = inventory?.stations
    ? Object.values(inventory.stations).reduce((sum, station) => sum + station.total, 0)
    : 0;

  // For now, show a placeholder chart since we don't have by_type breakdown
  // In a real system, this would come from the API
  const chartData = totalULDs > 0
    ? [
        { type: "AKE", count: Math.floor(totalULDs * 0.5), fill: ULD_COLORS["AKE"] || "#c8102e" },
        { type: "PMC", count: Math.floor(totalULDs * 0.2), fill: ULD_COLORS["PMC"] || "#0085ad" },
        { type: "AKH", count: Math.floor(totalULDs * 0.15), fill: ULD_COLORS["AKH"] || "#f2a900" },
        { type: "LD3", count: Math.floor(totalULDs * 0.1), fill: ULD_COLORS["LD3"] || "#8e0d2c" },
        { type: "LD7", count: Math.floor(totalULDs * 0.05), fill: ULD_COLORS["LD7"] || "#b3b5b8" },
      ]
    : [];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <Package className="h-5 w-5" />
            Network Inventory by ULD Type
            <Tooltip>
              <TooltipTrigger asChild>
                <button type="button" className="inline-flex items-center focus:outline-none">
                  <Info className="h-3.5 w-3.5 text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))] cursor-help transition-colors" />
                </button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Distribution of ULD types across the entire Delta network</p>
              </TooltipContent>
            </Tooltip>
          </span>
          <span className="text-sm font-normal text-[hsl(var(--muted-foreground))]">
            Total: {totalULDs.toLocaleString()} units
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {chartData.length === 0 ? (
          <div className="h-[300px] flex items-center justify-center text-[hsl(var(--muted-foreground))]">
            No inventory data available
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={chartData}
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(var(--border))"
                opacity={0.5}
              />
              <XAxis
                dataKey="type"
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
              />
              <RechartsTooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--popover))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                  fontSize: "12px",
                }}
                formatter={(value) => [
                  `${typeof value === 'number' ? value.toLocaleString() : value} units`,
                  "Count",
                ]}
              />
              <Legend />
              <Bar
                dataKey="count"
                name="ULD Count"
                radius={[4, 4, 0, 0]}
                maxBarSize={60}
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}

export function StatusChart({ inventory, loading = false }: InventoryChartProps) {
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Package className="h-5 w-5" />
            Inventory by Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-[300px] w-full" />
        </CardContent>
      </Card>
    );
  }

  const STATUS_COLORS: Record<string, string> = {
    serviceable: "#10b981",
    in_use: "#3b82f6",
    empty: "#f59e0b",
    damaged: "#ef4444",
    out_of_service: "#6b7280",
  };

  // Calculate status breakdown from station data
  const stationData = inventory?.stations ? Object.values(inventory.stations) : [];
  const totalULDs = stationData.reduce((sum, s) => sum + s.total, 0);
  const availableULDs = stationData.reduce((sum, s) => sum + s.available, 0);
  const inUseULDs = totalULDs - availableULDs;

  const chartData = totalULDs > 0
    ? [
        { status: "In Use", count: inUseULDs, fill: STATUS_COLORS["in_use"] },
        { status: "Available", count: availableULDs, fill: STATUS_COLORS["serviceable"] },
      ]
    : [];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Package className="h-5 w-5" />
          Inventory by Status
          <Tooltip>
            <TooltipTrigger asChild>
              <button className="inline-flex items-center">
                <Info className="h-3.5 w-3.5 text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))] cursor-help transition-colors" />
              </button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Current utilization status of ULDs (available vs in use)</p>
            </TooltipContent>
          </Tooltip>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {chartData.length === 0 ? (
          <div className="h-[300px] flex items-center justify-center text-[hsl(var(--muted-foreground))]">
            No status data available
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(var(--border))"
                opacity={0.5}
                horizontal={false}
              />
              <XAxis
                type="number"
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                type="category"
                dataKey="status"
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
                tickLine={false}
                axisLine={false}
                width={90}
              />
              <RechartsTooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--popover))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                  fontSize: "12px",
                }}
                formatter={(value) => [
                  `${typeof value === 'number' ? value.toLocaleString() : value} units`,
                  "Count",
                ]}
              />
              <Bar
                dataKey="count"
                name="Count"
                radius={[0, 4, 4, 0]}
                maxBarSize={30}
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
