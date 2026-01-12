import { Moon, Sun, Plane, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "@/hooks/useTheme";
import type { HealthStatus } from "@/types/api";

interface HeaderProps {
  health?: HealthStatus;
  isHealthLoading: boolean;
}

export function Header({ health, isHealthLoading }: HeaderProps) {
  const { theme, setTheme, resolvedTheme } = useTheme();

  const toggleTheme = () => {
    if (theme === "system") {
      setTheme(resolvedTheme === "dark" ? "light" : "dark");
    } else {
      setTheme(theme === "dark" ? "light" : "dark");
    }
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b border-[hsl(var(--border))] bg-[hsl(var(--background))/0.95] backdrop-blur supports-[backdrop-filter]:bg-[hsl(var(--background))/0.6]">
      <div className="container flex h-16 items-center justify-between px-4 mx-auto">
        {/* Logo and Brand */}
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-delta-red">
            <Plane className="w-6 h-6 text-white transform -rotate-45" />
          </div>
          <div className="flex flex-col">
            <span className="text-lg font-bold tracking-tight">
              <span className="text-delta-red">Delta</span>{" "}
              <span className="text-[hsl(var(--foreground))]">ULD Ops</span>
            </span>
            <span className="text-xs text-[hsl(var(--muted-foreground))]">
              Forecasting & Allocation System
            </span>
          </div>
        </div>

        {/* Center - System Status */}
        <div className="hidden md:flex items-center gap-2 px-4 py-2 rounded-full bg-[hsl(var(--muted))]">
          <Activity className="w-4 h-4" />
          {isHealthLoading ? (
            <span className="text-sm text-[hsl(var(--muted-foreground))]">
              Checking system...
            </span>
          ) : health?.status === "healthy" ? (
            <>
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-success"></span>
              </span>
              <span className="text-sm font-medium text-success">
                System Operational
              </span>
            </>
          ) : (
            <>
              <span className="relative flex h-2 w-2">
                <span className="relative inline-flex rounded-full h-2 w-2 bg-warning"></span>
              </span>
              <span className="text-sm font-medium text-warning">
                System Degraded
              </span>
            </>
          )}
        </div>

        {/* Right - Actions */}
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            className="rounded-full"
          >
            {resolvedTheme === "dark" ? (
              <Sun className="h-5 w-5" />
            ) : (
              <Moon className="h-5 w-5" />
            )}
            <span className="sr-only">Toggle theme</span>
          </Button>

          <div className="hidden sm:block h-6 w-px bg-[hsl(var(--border))]" />

          <div className="hidden sm:flex items-center gap-2 text-sm">
            <span className="text-[hsl(var(--muted-foreground))]">v1.0.0</span>
          </div>
        </div>
      </div>
    </header>
  );
}
