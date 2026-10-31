import { useEffect, useRef, useState } from "react";

type PlotlyModule = {
  react: (el: HTMLElement, data: unknown[], layout?: unknown, config?: unknown) => Promise<unknown>;
  purge: (el: HTMLElement) => void;
};

const FONT = { family: 'Inter, "Segoe UI", system-ui, sans-serif', size: 12, color: "#3a4760" };

// Plotly is large, so it is loaded once, lazily, and shared by every chart.
let plotlyPromise: Promise<PlotlyModule> | null = null;
let plotly: PlotlyModule | null = null;
function loadPlotly(): Promise<PlotlyModule> {
  plotlyPromise ??= import("plotly.js-dist-min").then((m) => (plotly = m.default as PlotlyModule));
  return plotlyPromise;
}

/** Renders a Plotly chart. Re-renders update the chart in place; it is only cleared when the component unmounts. */
export default function Plot({ data, layout, height = 380, label }: { data: unknown[]; layout?: Record<string, unknown>; height?: number; label: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [failed, setFailed] = useState(false);

  // Draw / update. No clean-up here: purging in the clean-up of this effect raced with the next draw
  // (React development mode runs effects twice), which left charts blank.
  useEffect(() => {
    let alive = true;
    loadPlotly()
      .then((P) => {
        const el = ref.current;
        if (!alive || !el) return;
        return P.react(
          el,
          data,
          {
            height,
            font: FONT,
            margin: { l: 10, r: 10, t: 10, b: 40 },
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            hoverlabel: { font: FONT, bgcolor: "#ffffff", bordercolor: "#d6dce6" },
            legend: { orientation: "h", y: -0.18, x: 0 },
            ...layout,
          },
          {
            displaylogo: false,
            responsive: true,
            modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines", "hoverCompareCartesian", "hoverClosestCartesian"],
            toImageButtonOptions: { format: "png", filename: label.replace(/\W+/g, "-").toLowerCase(), scale: 2 },
          },
        );
      })
      .catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, [data, layout, height, label]);

  // Free Plotly's resources only when the chart really leaves the page (synchronously, so it cannot race).
  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el && plotly) plotly.purge(el);
    };
  }, []);

  if (failed) return <p className="text-sm text-ink-500">The chart could not be drawn.</p>;
  return <div ref={ref} role="img" aria-label={label} style={{ minHeight: height }} className="w-full" />;
}
