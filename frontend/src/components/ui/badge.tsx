import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-[hsl(var(--ring))] focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-delta-red text-white",
        secondary:
          "border-transparent bg-[hsl(var(--muted))] text-[hsl(var(--muted-foreground))]",
        destructive:
          "border-transparent bg-danger text-white",
        outline: "text-[hsl(var(--foreground))]",
        success:
          "border-transparent bg-success/10 text-success border-success/20",
        warning:
          "border-transparent bg-warning/10 text-warning border-warning/20",
        danger:
          "border-transparent bg-danger/10 text-danger border-danger/20",
        info:
          "border-transparent bg-info/10 text-info border-info/20",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
