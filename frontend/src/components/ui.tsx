import { AlertCircle, Loader2 } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";

export function cx(...c: (string | false | null | undefined)[]) {
  return c.filter(Boolean).join(" ");
}

export function Button({
  variant = "primary", loading, className, children, ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost"; loading?: boolean }) {
  return (
    <button
      {...rest}
      disabled={rest.disabled || loading}
      className={cx(
        "inline-flex items-center justify-center gap-1.5 rounded-md px-4 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed",
        variant === "primary" && "bg-brand-600 text-white hover:bg-brand-700 disabled:bg-brand-300",
        variant === "secondary" && "border border-ink-200 bg-white text-ink-700 hover:bg-ink-50 disabled:text-ink-300",
        variant === "ghost" && "text-ink-500 hover:bg-ink-100 hover:text-ink-900",
        className,
      )}
    >
      {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
      {children}
    </button>
  );
}

export function Card({ title, subtitle, children, className, actions }: { title?: ReactNode; subtitle?: ReactNode; children: ReactNode; className?: string; actions?: ReactNode }) {
  return (
    <section className={cx("rounded-lg border border-ink-200 bg-white shadow-card print:break-inside-avoid print:shadow-none", className)}>
      {title && (
        <header className="flex flex-wrap items-start justify-between gap-2 border-b border-ink-100 px-5 py-3.5">
          <div>
            <h2 className="text-base font-semibold text-ink-900">{title}</h2>
            {subtitle && <p className="mt-0.5 text-sm text-ink-500">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

export function Spinner({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-10 text-sm text-ink-500" role="status">
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> {label}
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex items-start gap-3 rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-800" role="alert">
      <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <div className="flex-1">{message}</div>
      {onRetry && <Button variant="secondary" onClick={onRetry}>Try again</Button>}
    </div>
  );
}

/** One colour per result status, used by badges and every chart so the meaning stays consistent. */
export const STATUS_STYLE: Record<string, { label: string; color: string; badge: string }> = {
  covered: { label: "Well covered", color: "#0d8572", badge: "bg-teal-50 text-teal-700 ring-teal-100" },
  partially_covered: { label: "Partly covered", color: "#3f9ac3", badge: "bg-brand-50 text-brand-700 ring-brand-100" },
  underemphasized: { label: "Needs practical work", color: "#d99a1e", badge: "bg-amber-50 text-amber-800 ring-amber-200" },
  potentially_missing: { label: "Missing", color: "#c2453d", badge: "bg-red-50 text-red-700 ring-red-100" },
  uncertain: { label: "Unclear", color: "#9aa4b5", badge: "bg-ink-100 text-ink-700 ring-ink-200" },
  not_applicable: { label: "Not applicable", color: "#d6dce6", badge: "bg-ink-50 text-ink-500 ring-ink-200" },
};
export const STATUS_ORDER = ["covered", "partially_covered", "underemphasized", "potentially_missing", "uncertain"];

export function StatusBadge({ status }: { status: string }) {
  const s = STATUS_STYLE[status] ?? STATUS_STYLE.uncertain;
  return <span className={cx("inline-flex whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset", s.badge)}>{s.label}</span>;
}
