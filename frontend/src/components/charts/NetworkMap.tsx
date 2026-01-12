import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Map, Circle, Info } from "lucide-react";
import type { Station, NetworkInventory } from "@/types/api";
import { cn } from "@/lib/utils";
import usaMap from "@svg-maps/usa";

interface NetworkMapProps {
  stations: Station[];
  inventory?: NetworkInventory;
  selectedStation: string;
  onStationSelect: (station: string) => void;
  loading?: boolean;
}

// USA map coordinates (aligned with @svg-maps/usa viewBox: 192 9 1028 746)
const US_BOUNDS = {
  minLat: 24.5,
  maxLat: 49.5,
  minLng: -125,
  maxLng: -66,
};

// Map viewBox dimensions from @svg-maps/usa (includes Alaska/Hawaii offset)
const MAP_VIEWBOX = {
  x: 192,
  y: 9,
  width: 1028,
  height: 746,
};

function normalizeCoords(lat: number, lng: number) {
  // Normalize to 0-1 range
  const normalizedX = (lng - US_BOUNDS.minLng) / (US_BOUNDS.maxLng - US_BOUNDS.minLng);
  const normalizedY = 1 - (lat - US_BOUNDS.minLat) / (US_BOUNDS.maxLat - US_BOUNDS.minLat);

  // Apply Albers Equal Area Conic projection adjustments (approximate)
  // The @svg-maps/usa uses a projection that compresses east-west more in the north
  const latFactor = Math.cos((lat * Math.PI) / 180) * 0.85 + 0.15;

  // Scale to viewBox dimensions with projection correction
  // Continental US is roughly in the center-right portion of the viewBox
  const continentalWidth = 920; // Approximate width for continental US
  const continentalOffsetX = 250; // Offset to center continental US
  const continentalOffsetY = 100; // Offset from top
  const continentalHeight = 550; // Approximate height for continental US

  const x = continentalOffsetX + (normalizedX * continentalWidth * latFactor);
  const y = continentalOffsetY + (normalizedY * continentalHeight);

  return { x, y };
}

export function NetworkMap({
  stations,
  inventory,
  selectedStation,
  onStationSelect,
  loading = false,
}: NetworkMapProps) {
  const stationData = useMemo(() => {
    return stations.map((station) => {
      const { x, y } = normalizeCoords(
        station.coordinates.latitude,
        station.coordinates.longitude
      );
      const stationInventory = inventory?.stations[station.code];
      const count = stationInventory?.total || 0;

      return {
        ...station,
        x,
        y,
        count,
        isSelected: station.code === selectedStation,
      };
    });
  }, [stations, inventory, selectedStation]);

  if (loading) {
    return (
      <Card className="col-span-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Map className="h-5 w-5" />
            Network Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-[400px] w-full" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="col-span-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <Map className="h-5 w-5" />
            Network Overview
            <Tooltip>
              <TooltipTrigger asChild>
                <button type="button" className="inline-flex items-center focus:outline-none">
                  <Info className="h-3.5 w-3.5 text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))] cursor-help transition-colors" />
                </button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Interactive map showing all Delta stations with current ULD inventory levels. Click stations to view forecasts.</p>
              </TooltipContent>
            </Tooltip>
          </span>
          <div className="flex items-center gap-4 text-sm font-normal">
            <div className="flex items-center gap-2">
              <Circle className="h-3 w-3 fill-delta-red text-delta-red" />
              <span className="text-[hsl(var(--muted-foreground))]">Hub</span>
            </div>
            <div className="flex items-center gap-2">
              <Circle className="h-3 w-3 fill-delta-sky text-delta-sky" />
              <span className="text-[hsl(var(--muted-foreground))]">Focus City</span>
            </div>
            <div className="flex items-center gap-2">
              <Circle className="h-3 w-3 fill-delta-silver text-delta-silver" />
              <span className="text-[hsl(var(--muted-foreground))]">Station</span>
            </div>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="relative w-full h-[400px] bg-gradient-to-b from-[hsl(var(--muted))/0.2] to-[hsl(var(--muted))/0.4] rounded-lg overflow-hidden border border-[hsl(var(--border))]">
          {/* US Geographic Map */}
          <svg
            viewBox={usaMap.viewBox}
            className="absolute inset-0 w-full h-full"
            preserveAspectRatio="xMidYMid meet"
          >
            <defs>
              {/* Glow effect for selected station */}
              <filter id="glow">
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>

            {/* USA Map Background */}
            <g>
              {usaMap.locations.map((location) => (
                <path
                  key={location.id}
                  d={location.path}
                  fill="hsl(var(--muted))"
                  fillOpacity="0.4"
                  stroke="hsl(var(--foreground))"
                  strokeWidth="2"
                  strokeLinejoin="round"
                  strokeOpacity="0.5"
                />
              ))}
            </g>

            {/* Connection lines between hubs */}
            {stationData
              .filter((s) => s.hub_tier === 1)
              .map((hub, i, hubs) =>
                hubs.slice(i + 1).map((otherHub) => (
                  <line
                    key={`${hub.code}-${otherHub.code}`}
                    x1={hub.x}
                    y1={hub.y}
                    x2={otherHub.x}
                    y2={otherHub.y}
                    stroke="hsl(var(--border))"
                    strokeWidth="3"
                    strokeDasharray="10,10"
                    opacity="0.5"
                  />
                ))
              )}

            {/* Station markers */}
            {stationData.map((station) => {
              const size =
                station.hub_tier === 1
                  ? 35
                  : station.hub_tier === 2
                  ? 25
                  : 18;
              const color =
                station.hub_tier === 1
                  ? "#c8102e"
                  : station.hub_tier === 2
                  ? "#0085ad"
                  : "#b3b5b8";

              return (
                <g
                  key={station.code}
                  transform={`translate(${station.x}, ${station.y})`}
                  className="cursor-pointer"
                  onClick={(e) => {
                    e.stopPropagation();
                    e.preventDefault();
                    onStationSelect(station.code);
                  }}
                  style={{ pointerEvents: "all" }}
                  filter={station.isSelected ? "url(#glow)" : undefined}
                >
                  {/* Selection ring */}
                  {station.isSelected && (
                    <>
                      <circle
                        r={size + 15}
                        fill="none"
                        stroke={color}
                        strokeWidth="4"
                        className="animate-pulse"
                        opacity="0.6"
                      />
                      <circle
                        r={size + 8}
                        fill="none"
                        stroke={color}
                        strokeWidth="3"
                        opacity="0.8"
                      />
                    </>
                  )}
                  {/* Station dot with border */}
                  <circle
                    r={size + 0.5}
                    fill="white"
                    opacity={station.isSelected ? "0.3" : "0"}
                    className="transition-opacity"
                  />
                  <circle
                    r={size}
                    fill={color}
                    stroke="white"
                    strokeWidth={station.isSelected ? "8" : "4"}
                    className="transition-all hover:opacity-80"
                    opacity={station.isSelected ? "1" : "0.9"}
                  />
                  {/* Inner highlight */}
                  <circle
                    r={size * 0.4}
                    fill="white"
                    opacity="0.3"
                    transform={`translate(-${size * 0.2}, -${size * 0.2})`}
                  />
                  {/* Label */}
                  {(station.hub_tier <= 2 || station.isSelected) && (
                    <>
                      {/* Label background for better readability */}
                      <rect
                        x="-40"
                        y={size + 10}
                        width="80"
                        height="30"
                        fill="hsl(var(--background))"
                        opacity="0.9"
                        rx="5"
                      />
                      <text
                        y={size + 30}
                        textAnchor="middle"
                        fontSize="20"
                        fill="hsl(var(--foreground))"
                        fontWeight={station.isSelected ? "bold" : "600"}
                        pointerEvents="none"
                        className="select-none"
                      >
                        {station.code}
                      </text>
                    </>
                  )}
                </g>
              );
            })}
          </svg>

          {/* Selected station info overlay */}
          {selectedStation && (
            <div className="absolute bottom-4 left-4 p-4 glass rounded-lg max-w-xs">
              <div className="flex items-center gap-2 mb-2">
                <Badge variant="default">{selectedStation}</Badge>
                <span className="font-medium">
                  {stations.find((s) => s.code === selectedStation)?.city}
                </span>
              </div>
              <div className="text-sm text-[hsl(var(--muted-foreground))]">
                <p>
                  ULD Count:{" "}
                  <span className="font-semibold text-[hsl(var(--foreground))]">
                    {(
                      inventory?.stations[selectedStation]?.total || 0
                    ).toLocaleString()}
                  </span>
                </p>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
