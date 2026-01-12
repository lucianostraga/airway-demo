import { useState, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Package,
  Plane,
  AlertTriangle,
  TrendingUp,
  RefreshCw,
  Zap,
  Info,
} from "lucide-react";
import { Header } from "@/components/dashboard/Header";
import { KPICard } from "@/components/dashboard/KPICard";
import { StationSelector } from "@/components/dashboard/StationSelector";
import { RecommendationsTable } from "@/components/dashboard/RecommendationsTable";
import { ForecastChart } from "@/components/charts/ForecastChart";
import { InventoryChart, StatusChart } from "@/components/charts/InventoryChart";
import { NetworkMap } from "@/components/charts/NetworkMap";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  getHealth,
  getStations,
  getNetworkInventory,
  getDemandForecast,
  getSupplyForecast,
  getRepositioningRecommendations,
  runOptimization,
} from "@/api/client";

function App() {
  const [selectedStation, setSelectedStation] = useState<string>("ATL");
  const queryClient = useQueryClient();

  // Queries
  const { data: health, isLoading: isHealthLoading } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30000,
    retry: 1,
  });

  const { data: stations = [], isLoading: isStationsLoading } = useQuery({
    queryKey: ["stations"],
    queryFn: getStations,
    staleTime: 5 * 60 * 1000,
  });

  const { data: inventory, isLoading: isInventoryLoading } = useQuery({
    queryKey: ["inventory"],
    queryFn: getNetworkInventory,
    refetchInterval: 60000,
  });

  const { data: demandForecast, isLoading: isDemandLoading } = useQuery({
    queryKey: ["demand", selectedStation],
    queryFn: () => getDemandForecast(selectedStation, 24),
    enabled: !!selectedStation,
  });

  const { data: supplyForecast, isLoading: isSupplyLoading } = useQuery({
    queryKey: ["supply", selectedStation],
    queryFn: () => getSupplyForecast(selectedStation, 24),
    enabled: !!selectedStation,
  });

  const { data: recommendations = [], isLoading: isRecsLoading } = useQuery({
    queryKey: ["recommendations"],
    queryFn: getRepositioningRecommendations,
    refetchInterval: 120000,
  });

  // Mutations
  const optimizeMutation = useMutation({
    mutationFn: runOptimization,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["recommendations"] });
    },
  });

  // Handlers
  const handleStationChange = useCallback((station: string) => {
    setSelectedStation(station);
  }, []);

  const handleRefresh = useCallback(() => {
    queryClient.invalidateQueries();
  }, [queryClient]);

  const handleOptimize = useCallback(() => {
    optimizeMutation.mutate();
  }, [optimizeMutation]);

  // Calculated KPIs
  const stationData = Object.values(inventory?.stations || {});
  const totalULDs = stationData.reduce((sum, s) => sum + s.total, 0);
  const availableULDs = stationData.reduce((sum, s) => sum + s.available, 0);
  const utilizationRate = totalULDs > 0 ? (totalULDs - availableULDs) / totalULDs : 0;
  const criticalRecs = recommendations.filter((r) => r.priority === "critical").length;
  const stationCount = stationData.length;

  return (
    <TooltipProvider delayDuration={200} skipDelayDuration={100}>
      <div className="min-h-screen bg-[hsl(var(--background))] gradient-mesh">
        <Header health={health} isHealthLoading={isHealthLoading} />

        <main className="container mx-auto px-4 py-6 space-y-6">
          {/* Top Actions Bar */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div>
              <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
                Operations Dashboard
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button type="button" className="inline-flex items-center focus:outline-none">
                      <Info className="h-4 w-4 text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))] cursor-help transition-colors" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Real-time monitoring of ULD inventory, demand forecasts, and optimization recommendations</p>
                  </TooltipContent>
                </Tooltip>
              </h1>
              <p className="text-[hsl(var(--muted-foreground))]">
                Real-time ULD tracking, forecasting, and optimization
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleRefresh}
                    className="gap-2"
                  >
                    <RefreshCw className="h-4 w-4" />
                    Refresh
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Refresh all data from the server</p>
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    size="sm"
                    onClick={handleOptimize}
                    disabled={optimizeMutation.isPending}
                    className="gap-2"
                  >
                    <Zap className="h-4 w-4" />
                    {optimizeMutation.isPending ? "Optimizing..." : "Run Optimization"}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Run network-wide optimization to find optimal ULD repositioning moves</p>
                </TooltipContent>
              </Tooltip>
            </div>
          </div>

        {/* KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <KPICard
            title="Total ULD Fleet"
            value={totalULDs}
            icon={Package}
            loading={isInventoryLoading}
            change={2.5}
            changeLabel="vs last week"
          />
          <KPICard
            title="Active Stations"
            value={stationCount}
            icon={Plane}
            loading={isInventoryLoading}
            variant="success"
          />
          <KPICard
            title="Utilization Rate"
            value={`${(utilizationRate * 100).toFixed(1)}%`}
            icon={TrendingUp}
            loading={isInventoryLoading}
            variant="default"
            change={-1.2}
            changeLabel="vs yesterday"
          />
          <KPICard
            title="Critical Alerts"
            value={criticalRecs}
            icon={AlertTriangle}
            loading={isRecsLoading}
            variant={criticalRecs > 0 ? "danger" : "success"}
          />
        </div>

        {/* Station Selector */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base font-medium flex items-center gap-2">
              Station Analysis
              <Tooltip>
                <TooltipTrigger asChild>
                  <button type="button" className="inline-flex items-center focus:outline-none">
                    <Info className="h-3.5 w-3.5 text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))] cursor-help transition-colors" />
                  </button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Select a station to view detailed demand and supply forecasts</p>
                </TooltipContent>
              </Tooltip>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <StationSelector
              stations={stations}
              selectedStation={selectedStation}
              onStationChange={handleStationChange}
              loading={isStationsLoading}
            />
          </CardContent>
        </Card>

        {/* Main Content Tabs */}
        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="forecasting">Forecasting</TabsTrigger>
            <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            {/* Network Map */}
            <NetworkMap
              stations={stations}
              inventory={inventory}
              selectedStation={selectedStation}
              onStationSelect={handleStationChange}
              loading={isStationsLoading || isInventoryLoading}
            />

            {/* Charts Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <InventoryChart
                inventory={inventory}
                loading={isInventoryLoading}
              />
              <StatusChart inventory={inventory} loading={isInventoryLoading} />
            </div>
          </TabsContent>

          <TabsContent value="forecasting" className="space-y-4">
            <ForecastChart
              demandForecast={demandForecast}
              supplyForecast={supplyForecast}
              loading={isDemandLoading || isSupplyLoading}
              station={selectedStation}
            />

            {/* Forecast Details */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Demand Summary</CardTitle>
                </CardHeader>
                <CardContent>
                  {demandForecast && demandForecast.length > 0 ? (
                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-[hsl(var(--muted-foreground))]">
                          Station
                        </span>
                        <span className="font-semibold">
                          {demandForecast[0].station}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-[hsl(var(--muted-foreground))]">
                          Forecast Points
                        </span>
                        <span className="font-semibold">
                          {demandForecast.length}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-[hsl(var(--muted-foreground))]">
                          Avg Demand (Median)
                        </span>
                        <span className="font-semibold">
                          {Math.round(
                            demandForecast.reduce((sum, f) => sum + f.total_demand.q50, 0) /
                            demandForecast.length
                          )} ULDs
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-[hsl(var(--muted-foreground))]">
                          Confidence
                        </span>
                        <span className="font-semibold capitalize">
                          {demandForecast[0].confidence}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <p className="text-[hsl(var(--muted-foreground))]">
                      No forecast data available
                    </p>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Supply Summary</CardTitle>
                </CardHeader>
                <CardContent>
                  {supplyForecast && supplyForecast.length > 0 ? (
                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-[hsl(var(--muted-foreground))]">
                          Station
                        </span>
                        <span className="font-semibold">
                          {supplyForecast[0].station}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-[hsl(var(--muted-foreground))]">
                          Total Arrivals
                        </span>
                        <span className="font-semibold">
                          {supplyForecast.reduce((sum, f) => sum + f.expected_arrivals, 0)}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-[hsl(var(--muted-foreground))]">
                          Total Departures
                        </span>
                        <span className="font-semibold">
                          {supplyForecast.reduce((sum, f) => sum + f.expected_departures, 0)}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-[hsl(var(--muted-foreground))]">
                          Net Change
                        </span>
                        <span className="font-semibold">
                          {supplyForecast.reduce(
                            (sum, f) => sum + (f.expected_arrivals - f.expected_departures),
                            0
                          )}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <p className="text-[hsl(var(--muted-foreground))]">
                      No supply data available
                    </p>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="recommendations" className="space-y-4">
            <RecommendationsTable
              recommendations={recommendations}
              loading={isRecsLoading}
            />

            {/* Optimization Results */}
            {optimizeMutation.isSuccess && optimizeMutation.data && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Zap className="h-5 w-5 text-delta-gold" />
                    Latest Optimization Results
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <p className="text-sm text-[hsl(var(--muted-foreground))]">
                        Total Moves
                      </p>
                      <p className="text-2xl font-bold">
                        {optimizeMutation.data.total_moves}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-[hsl(var(--muted-foreground))]">
                        ULDs Moved
                      </p>
                      <p className="text-2xl font-bold">
                        {optimizeMutation.data.total_ulds_moved}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-[hsl(var(--muted-foreground))]">
                        Total Cost
                      </p>
                      <p className="text-2xl font-bold">
                        ${optimizeMutation.data.total_cost.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-[hsl(var(--muted-foreground))]">
                        Status
                      </p>
                      <p className="text-2xl font-bold capitalize">
                        {optimizeMutation.data.solver_status}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </main>

        {/* Footer */}
        <footer className="border-t border-[hsl(var(--border))] mt-8 py-6">
          <div className="container mx-auto px-4 text-center text-sm text-[hsl(var(--muted-foreground))]">
            <p>
              Delta ULD Forecasting & Allocation System |
              Powered by Machine Learning
            </p>
          </div>
        </footer>
      </div>
    </TooltipProvider>
  );
}

export default App;
