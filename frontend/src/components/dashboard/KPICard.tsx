import { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn, formatNumber } from "@/lib/utils";
import { TrendingUp, TrendingDown, Minus, type LucideIcon } from "lucide-react";

interface KPICardProps {
  title: string;
  value: number | string;
  change?: number;
  changeLabel?: string;
  icon: LucideIcon;
  loading?: boolean;
  variant?: "default" | "success" | "warning" | "danger";
  suffix?: string;
  prefix?: string;
}

export function KPICard({
  title,
  value,
  change,
  changeLabel,
  icon: Icon,
  loading = false,
  variant = "default",
  suffix = "",
  prefix = "",
}: KPICardProps) {
  const [displayValue, setDisplayValue] = useState<number | string>(0);
  const [isAnimating, setIsAnimating] = useState(false);

  // Animate number counting up
  useEffect(() => {
    if (loading || typeof value !== "number") {
      setDisplayValue(value);
      return;
    }

    setIsAnimating(true);
    const duration = 1000;
    const steps = 30;
    const stepDuration = duration / steps;
    const increment = value / steps;
    let current = 0;
    let step = 0;

    const timer = setInterval(() => {
      step++;
      current = Math.min(Math.round(increment * step), value);
      setDisplayValue(current);

      if (step >= steps) {
        clearInterval(timer);
        setDisplayValue(value);
        setIsAnimating(false);
      }
    }, stepDuration);

    return () => clearInterval(timer);
  }, [value, loading]);

  const variantStyles = {
    default: "bg-delta-navy/10 text-delta-navy dark:bg-delta-sky/10 dark:text-delta-sky",
    success: "bg-success/10 text-success",
    warning: "bg-warning/10 text-warning",
    danger: "bg-danger/10 text-danger",
  };

  const TrendIcon = change
    ? change > 0
      ? TrendingUp
      : change < 0
      ? TrendingDown
      : Minus
    : Minus;

  const trendColor = change
    ? change > 0
      ? "text-success"
      : change < 0
      ? "text-danger"
      : "text-[hsl(var(--muted-foreground))]"
    : "text-[hsl(var(--muted-foreground))]";

  if (loading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-start justify-between">
            <div className="space-y-2">
              <Skeleton className="h-4 w-24" />
              <Skeleton className="h-8 w-32" />
              <Skeleton className="h-3 w-20" />
            </div>
            <Skeleton className="h-12 w-12 rounded-lg" />
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="animate-slide-up overflow-hidden">
      <CardContent className="p-6">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <p className="text-sm font-medium text-[hsl(var(--muted-foreground))]">
              {title}
            </p>
            <div className="flex items-baseline gap-1">
              <span className="text-sm text-[hsl(var(--muted-foreground))]">
                {prefix}
              </span>
              <span
                className={cn(
                  "text-3xl font-bold tracking-tight",
                  isAnimating && "transition-all"
                )}
              >
                {typeof displayValue === "number"
                  ? formatNumber(displayValue)
                  : displayValue}
              </span>
              <span className="text-sm text-[hsl(var(--muted-foreground))]">
                {suffix}
              </span>
            </div>
            {change !== undefined && (
              <div className={cn("flex items-center gap-1 text-sm", trendColor)}>
                <TrendIcon className="h-3 w-3" />
                <span>
                  {change > 0 ? "+" : ""}
                  {change.toFixed(1)}%
                </span>
                {changeLabel && (
                  <span className="text-[hsl(var(--muted-foreground))]">
                    {changeLabel}
                  </span>
                )}
              </div>
            )}
          </div>
          <div
            className={cn(
              "flex items-center justify-center w-12 h-12 rounded-lg",
              variantStyles[variant]
            )}
          >
            <Icon className="w-6 h-6" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
