import { ArrowRight, Clock, DollarSign, TrendingUp, Info } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn, formatNumber } from "@/lib/utils";
import type { RepositioningRecommendation, PriorityLevel } from "@/types/api";

interface RecommendationsTableProps {
  recommendations: RepositioningRecommendation[];
  loading?: boolean;
}

function getPriorityVariant(
  priority: PriorityLevel
): "danger" | "warning" | "info" | "success" {
  const variants: Record<PriorityLevel, "danger" | "warning" | "info" | "success"> = {
    critical: "danger",
    high: "warning",
    medium: "info",
    low: "success",
  };
  return variants[priority];
}

export function RecommendationsTable({
  recommendations,
  loading = false,
}: RecommendationsTableProps) {
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Repositioning Recommendations
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="flex items-center gap-4">
                <Skeleton className="h-12 w-12 rounded-lg" />
                <div className="flex-1 space-y-2">
                  <Skeleton className="h-4 w-48" />
                  <Skeleton className="h-3 w-32" />
                </div>
                <Skeleton className="h-6 w-16" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (recommendations.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Repositioning Recommendations
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-[hsl(var(--muted-foreground))]">
            <TrendingUp className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p>No active recommendations at this time.</p>
            <p className="text-sm">The network is balanced.</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Repositioning Recommendations
            <Tooltip>
              <TooltipTrigger asChild>
                <button type="button" className="inline-flex items-center focus:outline-none">
                  <Info className="h-3.5 w-3.5 text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))] cursor-help transition-colors" />
                </button>
              </TooltipTrigger>
              <TooltipContent>
                <p>AI-generated recommendations for optimal ULD repositioning to prevent shortages and minimize costs</p>
              </TooltipContent>
            </Tooltip>
          </span>
          <Badge variant="secondary">{recommendations.length} Active</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {recommendations.slice(0, 5).map((rec, index) => (
            <div
              key={rec.recommendation_id}
              className={cn(
                "flex items-center gap-4 p-4 rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] transition-all hover:border-[hsl(var(--ring))]",
                "animate-slide-up"
              )}
              style={{ animationDelay: `${index * 50}ms` }}
            >
              {/* Route */}
              <div className="flex items-center gap-2 min-w-[140px]">
                <span className="font-mono font-bold text-delta-red">
                  {rec.origin_station}
                </span>
                <ArrowRight className="h-4 w-4 text-[hsl(var(--muted-foreground))]" />
                <span className="font-mono font-bold text-delta-navy dark:text-delta-sky">
                  {rec.destination_station}
                </span>
              </div>

              {/* ULD Info */}
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <Badge variant="outline">{rec.uld_type}</Badge>
                  <span className="text-sm font-medium">
                    x{rec.quantity} units
                  </span>
                </div>
                <p className="text-xs text-[hsl(var(--muted-foreground))] mt-1 line-clamp-1">
                  {rec.reason}
                </p>
              </div>

              {/* Flight & Time */}
              <div className="text-right min-w-[100px]">
                <div className="flex items-center gap-1 justify-end">
                  <Clock className="h-3 w-3 text-[hsl(var(--muted-foreground))]" />
                  <span className="text-sm font-medium capitalize">
                    {rec.transport_method}
                  </span>
                </div>
                <p className="text-xs text-[hsl(var(--muted-foreground))]">
                  {new Date(rec.recommended_departure).toLocaleTimeString("en-US", {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </p>
              </div>

              {/* Cost & Benefit */}
              <div className="text-right min-w-[80px]">
                <div className="flex items-center gap-1 justify-end text-[hsl(var(--muted-foreground))]">
                  <DollarSign className="h-3 w-3" />
                  <span className="text-sm">
                    {formatNumber(rec.cost_benefit.total_cost)}
                  </span>
                </div>
                <p className="text-xs text-success">
                  +{rec.cost_benefit.net_benefit.toFixed(1)}
                </p>
              </div>

              {/* Priority */}
              <Badge variant={getPriorityVariant(rec.priority)}>
                {rec.priority}
              </Badge>
            </div>
          ))}
        </div>

        {recommendations.length > 5 && (
          <div className="mt-4 text-center">
            <span className="text-sm text-[hsl(var(--muted-foreground))]">
              +{recommendations.length - 5} more recommendations
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
