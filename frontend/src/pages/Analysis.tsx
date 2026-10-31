import { ArrowLeft, BrainCircuit, Briefcase, CheckCircle2, Circle, FlaskConical, GitBranch, Loader2, Plus, Printer, Search, TrendingUp } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, errorMessage } from "../api/client";
import type { Analysis, Recommendation, SkillRow } from "../api/types";
import Plot from "../components/Plot";
import { Button, Card, ErrorState, STATUS_ORDER, STATUS_STYLE, Spinner, StatusBadge, cx } from "../components/ui";
import { useAsync } from "../components/useAsync";

const PROGRESS = [
  { key: "reading", label: "Reading the PDF and finding courses, topics and skills" },
  { key: "comparing", label: "Comparing with industry job postings using the trained ML models" },
  { key: "done", label: "Preparing recommendations" },
];

function Progress({ a }: { a: Analysis }) {
  const idx = PROGRESS.findIndex((p) => p.key === a.step);
  return (
    <Card title="Analysing your syllabus…" subtitle="This usually takes a few seconds.">
      <ol className="space-y-3">
        {PROGRESS.map((p) => {
          const i = PROGRESS.findIndex((x) => x.key === p.key);
          const state = a.status === "queued" ? "todo" : i < idx ? "done" : i === idx ? "now" : "todo";
          return (
            <li key={p.key} className="flex items-center gap-3 text-sm">
              {state === "done" ? <CheckCircle2 className="h-5 w-5 text-teal-600" aria-hidden />
                : state === "now" ? <Loader2 className="h-5 w-5 animate-spin text-brand-600" aria-hidden />
                : <Circle className="h-5 w-5 text-ink-200" aria-hidden />}
              <span className={state === "todo" ? "text-ink-500" : "text-ink-900"}>{p.label}</span>
            </li>
          );
        })}
      </ol>
      {a.status === "queued" && <p className="mt-4 text-sm text-ink-500">{a.message}</p>}
    </Card>
  );
}

function Stat({ label, value, hint, tone }: { label: string; value: string | number; hint?: string; tone?: string }) {
  return (
    <div className="rounded-lg border border-ink-200 bg-white p-4 shadow-card print:shadow-none">
      <div className="text-xs font-semibold uppercase tracking-wide text-ink-500">{label}</div>
      <div className={cx("mt-1 font-serif text-3xl font-semibold tabular-nums", tone ?? "text-ink-900")}>{value}</div>
      {hint && <div className="mt-0.5 text-xs text-ink-500">{hint}</div>}
    </div>
  );
}

const ACTION_ICON = { add: Plus, practice: FlaskConical, strengthen: TrendingUp, check: Search } as const;
const ACTION_STYLE = {
  add: "bg-red-50 text-red-700 ring-red-100",
  practice: "bg-amber-50 text-amber-800 ring-amber-200",
  strengthen: "bg-brand-50 text-brand-700 ring-brand-100",
  check: "bg-ink-100 text-ink-700 ring-ink-200",
} as const;

function RecommendationItem({ r, n }: { r: Recommendation; n: number }) {
  const Icon = ACTION_ICON[r.kind];
  return (
    <li className="flex gap-4 py-4 print:break-inside-avoid">
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-ink-100 text-sm font-semibold text-ink-700">{n}</div>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className={cx("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-semibold ring-1 ring-inset", ACTION_STYLE[r.kind])}>
            <Icon className="h-3 w-3" aria-hidden />{r.action}
          </span>
          <span className="text-base font-semibold text-ink-900">{r.skill}</span>
          <span className="text-xs text-ink-500">· {r.priority} priority · in {r.demand_pct}% of jobs</span>
        </div>
        <p className="mt-1 text-sm text-ink-700">{r.text}</p>
        {(r.ml ?? []).length > 0 && (
          <ul className="mt-1.5 flex flex-wrap gap-1.5">
            {r.ml!.map((m) => (
              <li key={m} className="inline-flex items-center gap-1 rounded bg-violet-50 px-2 py-0.5 text-xs text-violet-700 ring-1 ring-inset ring-violet-100">
                {m.startsWith("Apriori") ? <GitBranch className="h-3 w-3" aria-hidden /> : <BrainCircuit className="h-3 w-3" aria-hidden />}{m}
              </li>
            ))}
          </ul>
        )}
        {r.example_jobs.length > 0 && (
          <p className="mt-1 flex items-start gap-1.5 text-xs text-ink-500">
            <Briefcase className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
            <span>Asked for in jobs such as: {r.example_jobs.join(", ")}</span>
          </p>
        )}
      </div>
    </li>
  );
}

function short(s: string, n = 38) {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

function useCharts(a: Analysis) {
  return useMemo(() => {
    const skills = a.skills ?? [];
    const top = skills.slice(0, 20).reverse();
    const bar = STATUS_ORDER.map((st) => {
      const rows = top.filter((s) => s.status_key === st);
      return {
        type: "bar", orientation: "h", name: STATUS_STYLE[st].label,
        x: rows.map((s) => s.demand_pct), y: rows.map((s) => s.skill),
        customdata: rows.map((s) => [s.coverage, s.jobs]),
        marker: { color: STATUS_STYLE[st].color },
        hovertemplate: "<b>%{y}</b><br>In %{x:.1f}% of jobs (%{customdata[1]:,} postings)<br>Syllabus coverage: %{customdata[0]:.0f}/100<extra></extra>",
      };
    }).filter((t) => (t.x as number[]).length);
    const barLayout = {
      barmode: "stack",
      margin: { l: 170, r: 20, t: 10, b: 60 },
      xaxis: { title: { text: "% of relevant job postings asking for the skill" }, gridcolor: "#eef1f6", zeroline: false, ticksuffix: "%" },
      yaxis: { categoryorder: "array", categoryarray: top.map((s) => s.skill), automargin: true },
      legend: { orientation: "h", y: -0.2, x: 0 },
    };

    const counts = STATUS_ORDER.map((st) => skills.filter((s) => s.status_key === st).length);
    const donut = [{
      type: "pie", hole: 0.55, sort: false, direction: "clockwise",
      labels: STATUS_ORDER.map((st) => STATUS_STYLE[st].label), values: counts,
      marker: { colors: STATUS_ORDER.map((st) => STATUS_STYLE[st].color), line: { color: "#fff", width: 2 } },
      textinfo: "value", hovertemplate: "%{label}: %{value} skills (%{percent})<extra></extra>",
    }];
    const donutLayout = {
      margin: { l: 10, r: 10, t: 10, b: 10 }, showlegend: true, legend: { orientation: "v", x: 1, y: 0.5 },
      annotations: [{ text: `<b>${skills.length}</b><br>skills`, showarrow: false, font: { size: 14, color: "#14213a" } }],
    };

    const maxX = Math.max(5, ...skills.map((s) => s.demand_pct));
    const labelled = new Set(skills.slice(0, 8).map((s) => s.skill));
    const scatter = STATUS_ORDER.map((st) => {
      const rows = skills.filter((s) => s.status_key === st);
      return {
        type: "scatter", mode: "markers+text", name: STATUS_STYLE[st].label,
        x: rows.map((s) => s.demand_pct), y: rows.map((s) => s.coverage),
        text: rows.map((s) => (labelled.has(s.skill) ? s.skill : "")), textposition: "top center",
        textfont: { size: 10, color: "#3a4760" },
        customdata: rows.map((s) => s.skill),
        marker: { color: STATUS_STYLE[st].color, size: 10, line: { color: "#fff", width: 1 } },
        hovertemplate: "<b>%{customdata}</b><br>Demand: %{x:.1f}% of jobs<br>Coverage: %{y:.0f}/100<extra></extra>",
      };
    }).filter((t) => (t.x as number[]).length);
    const scatterLayout = {
      margin: { l: 55, r: 15, t: 10, b: 60 },
      xaxis: { title: { text: "Industry demand (% of jobs)" }, range: [0, maxX * 1.08], gridcolor: "#eef1f6", zeroline: false, ticksuffix: "%" },
      yaxis: { title: { text: "Syllabus coverage (0–100)" }, range: [-6, 106], gridcolor: "#eef1f6", zeroline: false },
      shapes: [{ type: "rect", xref: "x", yref: "y", x0: maxX / 3, x1: maxX * 1.08, y0: -6, y1: 40, fillcolor: "rgba(194,69,61,0.06)", line: { width: 0 }, layer: "below" }],
      showlegend: false,
    };

    const courses = (a.courses ?? []).filter((c) => c.skills.length);
    const totals = new Map<string, number>();
    courses.forEach((c) => c.skills.forEach((s) => totals.set(s.skill, (totals.get(s.skill) ?? 0) + s.mentions)));
    const cols = [...totals.entries()].sort((x, y) => y[1] - x[1]).slice(0, 18).map(([k]) => k);
    const heat = courses.length > 1 ? [{
      type: "heatmap",
      x: cols, y: courses.map((c) => short(c.name)),
      z: courses.map((c) => cols.map((k) => c.skills.find((s) => s.skill === k)?.mentions ?? 0)),
      customdata: courses.map((c) => cols.map(() => c.name)),
      colorscale: [[0, "#f7f9fc"], [0.001, "#d5ecf5"], [0.5, "#3f9ac3"], [1, "#155372"]],
      xgap: 2, ygap: 2, showscale: false,
      hovertemplate: "<b>%{customdata}</b><br>%{x}: %{z} mention(s)<extra></extra>",
    }] : null;
    const heatLayout = {
      margin: { l: 250, r: 10, t: 10, b: 120 },
      xaxis: { tickangle: -40, automargin: true, side: "bottom" },
      yaxis: { autorange: "reversed", automargin: true },
    };
    const roles = (a.roles ?? []).slice().reverse();
    const roleBar = [{
      type: "bar", orientation: "h", x: roles.map((r) => 100 * r.probability), y: roles.map((r) => r.role),
      marker: { color: roles.map((_, i) => (i === roles.length - 1 ? "#17668c" : "#acd8ea")) },
      text: roles.map((r) => `${(100 * r.probability).toFixed(0)}%`), textposition: "outside", cliponaxis: false,
      hovertemplate: "%{y}: %{x:.1f}%<extra></extra>",
    }];
    const roleLayout = { margin: { l: 200, r: 40, t: 10, b: 40 }, xaxis: { title: { text: "Model 1 probability (%)" }, gridcolor: "#eef1f6", ticksuffix: "%", range: [0, Math.max(10, ...roles.map((r) => 100 * r.probability)) * 1.2] }, yaxis: { automargin: true }, showlegend: false };
    const bundles = a.bundles ?? [];
    const radar = bundles.length >= 3 ? [{
      type: "scatterpolar", fill: "toself", r: [...bundles.map((b) => b.readiness), bundles[0].readiness],
      theta: [...bundles.map((b) => short(b.name, 22)), short(bundles[0].name, 22)],
      line: { color: "#17668c" }, fillcolor: "rgba(31,126,169,0.25)", name: "Syllabus readiness",
      hovertemplate: "%{theta}<br>Readiness %{r:.0f}/100<extra></extra>",
    }] : null;
    const radarLayout = { margin: { l: 120, r: 120, t: 30, b: 30 }, polar: { radialaxis: { range: [0, 100], tickfont: { size: 10 }, gridcolor: "#d6dce6" }, angularaxis: { tickfont: { size: 10 } } }, showlegend: false };
    return { bar, barLayout, donut, donutLayout, scatter, scatterLayout, heat, heatLayout, heatHeight: Math.max(260, courses.length * 26 + 140), roleBar, roleLayout, radar, radarLayout };
  }, [a]);
}

function SkillsTable({ skills }: { skills: SkillRow[] }) {
  const [filter, setFilter] = useState<string>("all");
  const rows = filter === "all" ? skills : skills.filter((s) => s.status_key === filter);
  return (
    <>
      <div className="mb-3 flex flex-wrap gap-1.5 print:hidden">
        {["all", ...STATUS_ORDER].map((k) => (
          <button key={k} onClick={() => setFilter(k)}
            className={cx("rounded-full px-3 py-1 text-xs font-medium ring-1 ring-inset", filter === k ? "bg-ink-900 text-white ring-ink-900" : "bg-white text-ink-700 ring-ink-200 hover:bg-ink-50")}>
            {k === "all" ? `All (${skills.length})` : `${STATUS_STYLE[k].label} (${skills.filter((s) => s.status_key === k).length})`}
          </button>
        ))}
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="border-b border-ink-200">
            <tr>
              <th className="th">Skill</th>
              <th className="th">Result</th>
              <th className="th text-right">Jobs asking</th>
              <th className="th text-right" title="Share of those postings where the skill is required (Model 2)">Required</th>
              <th className="th text-right">Syllabus coverage</th>
              <th className="th">Found in syllabus as</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-ink-100">
            {rows.map((s) => (
              <tr key={s.skill}>
                <td className="td font-medium text-ink-900">{s.skill}<div className="text-xs font-normal text-ink-500">{s.group}</div></td>
                <td className="td"><StatusBadge status={s.status_key} /></td>
                <td className="td text-right tabular-nums">{s.demand_pct}%</td>
                <td className="td text-right tabular-nums">{Math.round(s.required_pct)}%</td>
                <td className="td text-right tabular-nums">{Math.round(s.coverage)}/100</td>
                <td className="td text-xs text-ink-500">
                  {s.in_syllabus_as.length ? s.in_syllabus_as.join(", ") : "not found"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function Result({ a }: { a: Analysis }) {
  const s = a.summary!;
  const charts = useCharts(a);
  const recs = a.recommendations ?? [];
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <Link to="/" className="mb-2 inline-flex items-center gap-1 text-sm text-ink-500 hover:text-ink-900 print:hidden"><ArrowLeft className="h-4 w-4" />All analyses</Link>
          <h1 className="font-serif text-2xl font-semibold text-ink-900 sm:text-3xl">{a.course_name}</h1>
          <p className="mt-1 text-sm text-ink-500">
            {[a.institution, a.filename, `${s.pages} pages`].filter(Boolean).join(" · ")}
            <br />Compared with {a.market?.postings_compared.toLocaleString("en-IN")} real job postings ({a.job_field_label}{a.market?.period ? `, ${a.market.period}` : ""}).
          </p>
        </div>
        <div className="flex gap-2 print:hidden">
          <Button variant="secondary" onClick={() => window.print()}><Printer className="h-4 w-4" />Print / save as PDF</Button>
          <Link to="/" className="inline-flex items-center rounded-md bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700">New analysis</Link>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Industry match" value={`${s.match_pct}%`} hint={`of the ${s.industry_skills_compared} most requested skills`} tone="text-brand-700" />
        <Stat label="Well covered" value={s.well_covered} hint="skills taught at the level industry asks" tone="text-teal-700" />
        <Stat label="Missing" value={s.missing} hint="requested skills not found in the PDF" tone="text-red-700" />
        <Stat label="Courses found" value={s.courses} hint={`${s.skills_taught} distinct skills taught`} />
      </div>

      {a.warnings?.map((w) => <ErrorState key={w} message={w} />)}

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title={<span className="inline-flex items-center gap-1.5"><BrainCircuit className="h-4 w-4 text-brand-600" />Job roles this syllabus prepares for</span>}
          subtitle={`ML model 1 (${a.models?.role_classifier.algorithm ?? "job-role classifier"}) reads the syllabus text and predicts the matching job roles`}>
          {(a.roles ?? []).length ? <Plot label="Predicted job roles" data={charts.roleBar} layout={charts.roleLayout} height={300} /> : <p className="text-sm text-ink-500">Not enough text to predict.</p>}
        </Card>
        <Card title={<span className="inline-flex items-center gap-1.5"><BrainCircuit className="h-4 w-4 text-brand-600" />Readiness for in-demand skill bundles</span>}
          subtitle={`ML model 3 (K-Means) found ${a.bundles?.length ?? 0} groups of skills employers ask for together; how much of each the syllabus covers`}>
          {charts.radar ? <Plot label="Skill bundle readiness" data={charts.radar} layout={charts.radarLayout} height={300} /> : <p className="text-sm text-ink-500">No bundles available.</p>}
        </Card>
      </div>

      <Card title="Recommendations for the syllabus" subtitle="Generated by the ML pipeline, most important first: demand weighted by the requirement classifier (SVM), job roles from the role classifier (SVM), links between skills from Apriori association rules.">
        {recs.length ? (
          <ol className="-my-4 divide-y divide-ink-100">{recs.map((r, i) => <RecommendationItem key={r.skill} r={r} n={i + 1} />)}</ol>
        ) : (
          <p className="text-sm text-ink-700">No gaps found: the syllabus covers the skills employers ask for most.</p>
        )}
      </Card>

      <Card title="What employers ask for, and how your syllabus covers it" subtitle="The 20 skills most often requested in the relevant job postings. Colour shows the syllabus coverage.">
        <Plot label="Top requested skills" data={charts.bar} layout={charts.barLayout} height={560} />
      </Card>

      <div className="grid gap-6 lg:grid-cols-5">
        <Card title="Overall coverage" subtitle="All compared skills by result" className="lg:col-span-2">
          <Plot label="Coverage breakdown" data={charts.donut} layout={charts.donutLayout} height={320} />
        </Card>
        <Card title="Demand vs coverage" subtitle="Shaded area = high demand, low coverage: attend to these first" className="lg:col-span-3">
          <Plot label="Demand versus coverage" data={charts.scatter} layout={charts.scatterLayout} height={320} />
        </Card>
      </div>

      {charts.heat && (
        <Card title="Which course teaches which skill" subtitle="Courses detected in the PDF and the skills found in each (darker = mentioned more often)">
          <Plot label="Skills by course" data={charts.heat} layout={charts.heatLayout} height={charts.heatHeight} />
        </Card>
      )}

      <Card title="What the syllabus covers" subtitle="Skills found in each course, and the job role ML model 1 links each course to. Underlined skills are backed by a lab, practical or project.">
        <div className="grid gap-4 md:grid-cols-2">
          {(a.courses ?? []).map((c) => (
            <div key={c.name} className="rounded-md border border-ink-100 p-3 print:break-inside-avoid">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div className="text-sm font-semibold text-ink-900">{c.name}</div>
                {c.role ? <span className="rounded-full bg-brand-50 px-2 py-0.5 text-xs text-brand-700 ring-1 ring-inset ring-brand-100" title="Predicted by ML model 1">{c.role.role} · {Math.round(c.role.probability * 100)}%</span>
                  : c.skills.length > 0 && <span className="rounded-full bg-ink-50 px-2 py-0.5 text-xs text-ink-500 ring-1 ring-inset ring-ink-200" title="ML model 1 does not link this course to one job role">Foundation course</span>}
              </div>
              {c.skills.length ? (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {c.skills.map((k) => (
                    <span key={k.skill} className={cx("rounded bg-ink-50 px-2 py-0.5 text-xs text-ink-700 ring-1 ring-inset ring-ink-200", k.practical && "underline decoration-teal-500 decoration-2 underline-offset-2")}>{k.skill}</span>
                  ))}
                </div>
              ) : (
                <p className="mt-1 text-xs text-ink-500">No skills from the skill list were found in this course.</p>
              )}
            </div>
          ))}
        </div>
      </Card>

      <Card title="All skills compared">
        <SkillsTable skills={a.skills ?? []} />
      </Card>

      {a.models && (
        <Card title="Machine-learning models used in this analysis" actions={<Link to="/models" className="text-sm font-medium text-brand-700 hover:underline print:hidden">See training and accuracy →</Link>}>
          <ul className="grid gap-3 text-sm text-ink-700 md:grid-cols-3">
            <li><b className="text-ink-900">1. Job-role classifier</b><br />{a.models.role_classifier.algorithm}, test accuracy {(a.models.role_classifier.accuracy * 100).toFixed(1)}%. Predicts which job roles the syllabus and each course prepare for, and labels the job postings by role.</li>
            <li><b className="text-ink-900">2. Requirement-level classifier</b><br />{a.models.requirement_classifier.algorithm}, test accuracy {(a.models.requirement_classifier.accuracy * 100).toFixed(1)}%. Decides whether each job-ad sentence states a required or a preferred skill; this weights the demand score.</li>
            <li><b className="text-ink-900">3. Skill recommender</b><br />Apriori ({a.models.recommender.rules} association rules) and K-Means ({a.models.recommender.clusters} skill bundles). Explains recommendations and measures bundle readiness.</li>
          </ul>
        </Card>
      )}

      {a.note && <p className="text-xs text-ink-500">Note: {a.note}</p>}
    </div>
  );
}

export default function AnalysisPage() {
  const { id } = useParams();
  const q = useAsync(() => api.get<Analysis>(`/api/analyses/${id}`), [id]);
  const [retrying, setRetrying] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const status = q.data?.status;

  useEffect(() => {
    if (status !== "queued" && status !== "processing" && !(status === "done" && !q.data?.summary)) return;
    const t = setInterval(() => q.reload(true), 1000);
    return () => clearInterval(t);
  }, [status, q.reload]);

  if (q.loading && !q.data) return <Spinner />;
  if (q.error) return <ErrorState message={q.error} onRetry={() => q.reload()} />;
  const a = q.data!;
  if (a.status === "done" && a.summary) return <Result a={a} />;
  if (a.status === "done") return <Spinner label="Loading the result…" />;
  return (
    <div className="mx-auto max-w-2xl space-y-4">
      <Link to="/" className="inline-flex items-center gap-1 text-sm text-ink-500 hover:text-ink-900"><ArrowLeft className="h-4 w-4" />All analyses</Link>
      <h1 className="font-serif text-2xl font-semibold text-ink-900">{a.course_name}</h1>
      {a.status === "failed" ? (
        <Card title="The analysis could not be completed">
          <p className="text-sm text-ink-700">{a.message}</p>
          {err && <div className="mt-3"><ErrorState message={err} /></div>}
          <div className="mt-4 flex gap-2">
            <Button loading={retrying} onClick={async () => {
              setRetrying(true);
              setErr(null);
              try { await api.post(`/api/analyses/${a.id}/retry`); q.reload(true); } catch (e) { setErr(errorMessage(e)); } finally { setRetrying(false); }
            }}>Try again</Button>
            <Link to="/" className="inline-flex items-center rounded-md border border-ink-200 px-4 py-2 text-sm text-ink-700 hover:bg-ink-50">Upload a different PDF</Link>
          </div>
        </Card>
      ) : (
        <Progress a={a} />
      )}
    </div>
  );
}
