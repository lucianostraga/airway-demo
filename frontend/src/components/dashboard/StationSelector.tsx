import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { MapPin } from "lucide-react";
import type { Station } from "@/types/api";

interface StationSelectorProps {
  stations: Station[];
  selectedStation: string;
  onStationChange: (station: string) => void;
  loading?: boolean;
}

export function StationSelector({
  stations,
  selectedStation,
  onStationChange,
  loading = false,
}: StationSelectorProps) {
  const hubs = stations.filter((s) => s.hub_tier === 1);
  const focusCities = stations.filter((s) => s.hub_tier === 2);
  const otherStations = stations.filter((s) => s.hub_tier > 2);

  return (
    <div className="flex items-center gap-2">
      <MapPin className="h-4 w-4 text-[hsl(var(--muted-foreground))]" />
      <Select
        value={selectedStation}
        onValueChange={onStationChange}
        disabled={loading}
      >
        <SelectTrigger className="w-[280px]">
          <SelectValue placeholder="Select a station" />
        </SelectTrigger>
        <SelectContent>
          {hubs.length > 0 && (
            <>
              <div className="px-2 py-1.5 text-xs font-semibold text-[hsl(var(--muted-foreground))] bg-[hsl(var(--muted))]">
                Hub Stations
              </div>
              {hubs.map((station) => (
                <SelectItem key={station.code} value={station.code}>
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-semibold text-delta-red">
                      {station.code}
                    </span>
                    <span className="text-[hsl(var(--muted-foreground))]">-</span>
                    <span>{station.city}</span>
                  </div>
                </SelectItem>
              ))}
            </>
          )}
          {focusCities.length > 0 && (
            <>
              <div className="px-2 py-1.5 text-xs font-semibold text-[hsl(var(--muted-foreground))] bg-[hsl(var(--muted))] mt-1">
                Focus Cities
              </div>
              {focusCities.map((station) => (
                <SelectItem key={station.code} value={station.code}>
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-semibold text-delta-navy dark:text-delta-sky">
                      {station.code}
                    </span>
                    <span className="text-[hsl(var(--muted-foreground))]">-</span>
                    <span>{station.city}</span>
                  </div>
                </SelectItem>
              ))}
            </>
          )}
          {otherStations.length > 0 && (
            <>
              <div className="px-2 py-1.5 text-xs font-semibold text-[hsl(var(--muted-foreground))] bg-[hsl(var(--muted))] mt-1">
                Other Stations
              </div>
              {otherStations.map((station) => (
                <SelectItem key={station.code} value={station.code}>
                  <div className="flex items-center gap-2">
                    <span className="font-mono">{station.code}</span>
                    <span className="text-[hsl(var(--muted-foreground))]">-</span>
                    <span>{station.city}</span>
                  </div>
                </SelectItem>
              ))}
            </>
          )}
        </SelectContent>
      </Select>
    </div>
  );
}
