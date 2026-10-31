import { BrainCircuit, Database, GitBranch, Layers } from "lucide-react";
import { useMemo } from "react";
import { api } from "../api/client";
import type { ClassifierReport, Metrics, Pipeline } from "../api/types";
import Plot from "../components/Plot";
import { Card, ErrorState, Spinner, cx } from "../components/ui";
import { useAsync } from "../components/useAsync";

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-ink-200 bg-white p-4 shadow-card">
      <div className="text-xs font-semibold uppercase tracking-wide text-ink-500">{label}</div>
      <div className="mt-1 font-serif text-2xl font-semibold tabular-nums text-ink-900">{value}</div>
    </div>
  );
}

function ClassifierSection({ n, title, report, purpose }: { n: number; title: string; report: ClassifierReport; purpose: string }) {
  const names = Object.keys(report.models).filter((k) => !k.endsWith("(calibrated)"));
  const best = report.models[report.best_model];
  const compare = useMemo(() => [
    { type: "bar", name: "Accuracy", x: names, y: names.map((k) => 100 * report.models[k].accuracy), marker: { color: "#17668c" },
      text: names.map((k) => pct(report.models[k].accuracy)), textposition: "outside", cliponaxis: false },
    { type: "bar", name: "Macro F1", x: names, y: names.map((k) => 100 * report.models[k].macro_f1), marker: { color: "#13a38b" },
      text: names.map((k) => pct(report.models[k].macro_f1)), textposition: "outside", cliponaxis: false },
  ], [report, names]);
  const short = (s: string) => (s.length > 22 ? `${s.slice(0, 21)}…` : s);
  const cm = best.confusion_matrix;
  const norm = cm.map((row) => { const t = row.reduce((a, b) => a + b, 0) || 1; return row.map((v) => (100 * v) / t); });
  const heat = useMemo(() => [{
    type: "heatmap", z: norm, x: report.labels.map(short), y: report.labels.map(short),
    text: cm.map((row) => row.map(String)), texttemplate: "%{text}", textfont: { size: 10 },
    colorscale: [[0, "#f7f9fc"], [0.5, "#77bcd9"], [1, "#155372"]], showscale: false,
    hovertemplate: "True: %{y}<br>Predicted: %{x}<br>%{text} postings (%{z:.0f}% of the row)<extra></extra>",
  }], [report]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <section className="space-y-4">
      <h2 className="flex items-center gap-2 font-serif text-xl font-semibold text-ink-900">
        <span className="flex h-7 w-7 items-center justify-center rounded-full bg-brand-600 text-sm text-white">{n}</span>{title}
      </h2>
      <Card>
        <dl className="grid gap-x-6 gap-y-2 text-sm md:grid-cols-2">
          <div><dt className="font-semibold text-ink-900">Task</dt><dd className="text-ink-700">{report.task}</dd></div>
          <div><dt className="font-semibold text-ink-900">Used in the app for</dt><dd className="text-ink-700">{purpose}</dd></div>
          <div><dt className="font-semibold text-ink-900">Features</dt><dd className="text-ink-700">{report.features}</dd></div>
          <div><dt className="font-semibold text-ink-900">Data</dt><dd className="text-ink-700">{report.train_size.toLocaleString("en-IN")} training / {report.test_size.toLocaleString("en-IN")} test examples (80/20 stratified split){report.label_source ? `; labels: ${report.label_source}` : ""}</dd></div>
        </dl>
      </Card>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Algorithm comparison (held-out test set)" subtitle={`Best by macro-F1: ${report.best_model}. The baseline always predicts the most common class.`}>
          <Plot label={`${title} algorithm comparison`} data={compare} height={360}
            layout={{ barmode: "group", margin: { l: 45, r: 10, t: 40, b: 100 }, yaxis: { range: [0, 110], ticksuffix: "%", gridcolor: "#eef1f6" }, xaxis: { tickangle: -25, automargin: true }, legend: { orientation: "h", x: 0, y: 1.14 } }} />
        </Card>
        <Card title={`Confusion matrix: ${report.best_model}`} subtitle="Rows = true class, columns = predicted (counts; colour = row %)">
          <Plot label={`${title} confusion matrix`} data={heat} height={Math.max(300, report.labels.length * 30 + 120)}
            layout={{ margin: { l: 160, r: 10, t: 10, b: 120 }, xaxis: { tickangle: -40, automargin: true }, yaxis: { autorange: "reversed", automargin: true } }} />
        </Card>
      </div>
      <Card title="Per-class results (best model)">
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead className="border-b border-ink-200"><tr><th className="th">Class</th><th className="th text-right">Precision</th><th className="th text-right">Recall</th><th className="th text-right">F1</th><th className="th text-right">Test examples</th></tr></thead>
            <tbody className="divide-y divide-ink-100">
              {report.labels.map((l) => {
                const c = best.per_class[l];
                return (
                  <tr key={l}><td className="td font-medium text-ink-900">{l}</td><td className="td text-right tabular-nums">{pct(c.precision)}</td><td className="td text-right tabular-nums">{pct(c.recall)}</td>
                    <td className={cx("td text-right tabular-nums", c.f1 >= 0.7 ? "text-teal-700" : c.f1 < 0.5 ? "text-red-700" : "")}>{pct(c.f1)}</td><td className="td text-right tabular-nums">{c.support.toLocaleString("en-IN")}</td></tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="mt-3 overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead className="border-b border-ink-200"><tr><th className="th">Algorithm</th><th className="th text-right">Accuracy</th><th className="th text-right">Macro F1</th><th className="th text-right">Weighted F1</th><th className="th text-right">Training time</th></tr></thead>
            <tbody className="divide-y divide-ink-100">
              {names.map((k) => (
                <tr key={k} className={k === report.best_model ? "bg-teal-50/50" : k.startsWith("Baseline") ? "italic text-ink-500" : ""}><td className="td font-medium text-ink-900">{k}{k === report.best_model ? " ★" : ""}</td>
                  <td className="td text-right tabular-nums">{pct(report.models[k].accuracy)}</td><td className="td text-right tabular-nums">{pct(report.models[k].macro_f1)}</td>
                  <td className="td text-right tabular-nums">{pct(report.models[k].weighted_f1)}</td><td className="td text-right tabular-nums">{report.models[k].train_seconds}s</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </section>
  );
}

const fmt = (v: number | string) => (typeof v === "number" ? v.toLocaleString("en-IN") : v);

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <li className="relative flex gap-4 pb-6 last:pb-0">
      <div className="flex flex-col items-center">
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-brand-600 text-sm font-semibold text-white">{n}</span>
        <span className="mt-1 w-px flex-1 bg-ink-200" aria-hidden />
      </div>
      <div className="min-w-0 flex-1 pt-1">
        <h3 className="font-semibold text-ink-900">{title}</h3>
        <div className="mt-2 text-sm text-ink-700">{children}</div>
      </div>
    </li>
  );
}

function PipelineCard({ p }: { p: Pipeline }) {
  return (
    <Card title="How the models were built" subtitle="The complete machine-learning pipeline, with the real numbers from the last training run">
      <ol>
        <Step n={1} title="Datasets">
          <p>{p.datasets.description}</p>
          <table className="mt-2 text-sm">
            <tbody>
              {Object.entries(p.datasets.files).map(([f, n]) => (
                <tr key={f}><td className="pr-6 font-mono text-xs text-ink-700">{f}</td><td className="text-right tabular-nums">{fmt(n)} rows</td></tr>
              ))}
              <tr className="border-t border-ink-200"><td className="pr-6 font-medium text-ink-900">Total</td><td className="text-right font-medium tabular-nums text-ink-900">{fmt(p.datasets.raw_rows)} rows</td></tr>
            </tbody>
          </table>
        </Step>
        <Step n={2} title="Data cleaning and preprocessing">
          <table className="w-full text-sm">
            <tbody className="divide-y divide-ink-100">
              {p.preprocessing.map((r) => (
                <tr key={r.step}><td className="py-1.5 pr-4">{r.step}</td><td className="py-1.5 text-right font-medium tabular-nums text-ink-900">{fmt(r.count)}</td></tr>
              ))}
            </tbody>
          </table>
        </Step>
        <Step n={3} title="Feature engineering">
          <ul className="space-y-1.5">
            {p.features.map((f) => <li key={f.model}><b className="text-ink-900">{f.model}:</b> {f.features}</li>)}
          </ul>
        </Step>
        <Step n={4} title="Training and model selection">
          <ul className="space-y-2">
            {p.training.map((t) => (
              <li key={t.model}>
                <b className="text-ink-900">{t.model}:</b> {t.split}.<br />
                Algorithms trained: {t.algorithms.join(", ")}.<br />
                Selected: <b className="text-teal-700">{t.selected}</b> ({t.selection}).
              </li>
            ))}
          </ul>
        </Step>
        <Step n={5} title="Evaluation">
          Accuracy, precision, recall, F1 (macro and weighted) and confusion matrices on held-out test data the models never
          saw during training; silhouette score for the clusters. Shown in the sections below.
        </Step>
      </ol>
    </Card>
  );
}

export default function ModelsPage() {
  const q = useAsync(() => api.get<Metrics>("/api/models"), []);
  const m = q.data;
  const sil = useMemo(() => {
    if (!m) return null;
    const ks = Object.keys(m.skill_recommender.kmeans.silhouette);
    return [{ type: "scatter", mode: "lines+markers", x: ks.map(Number), y: ks.map((k) => m.skill_recommender.kmeans.silhouette[k]),
      line: { color: "#17668c" }, marker: { size: 8, color: ks.map((k) => (Number(k) === m.skill_recommender.kmeans.k ? "#c2453d" : "#17668c")) },
      hovertemplate: "k = %{x}<br>silhouette %{y:.3f}<extra></extra>" }];
  }, [m]);
  if (q.loading) return <Spinner />;
  if (q.error || !m) return <ErrorState message={q.error ?? "No model report"} onRetry={() => q.reload()} />;
  const ap = m.skill_recommender.apriori;
  const km = m.skill_recommender.kmeans;
  return (
    <div className="space-y-10">
      <div>
        <h1 className="font-serif text-3xl font-semibold tracking-tight text-ink-900">Machine-learning models</h1>
        <p className="mt-2 max-w-3xl text-ink-500">
          SkillPulse trains three models on real job postings (<code className="rounded bg-ink-100 px-1 text-xs">python -m ml.train_all</code>) and uses them in every syllabus analysis.
          All numbers below are measured on held-out test data that the models never saw during training.
        </p>
      </div>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Job postings used" value={m.data.postings_loaded.toLocaleString("en-IN")} />
        <Stat label="Software & IT postings" value={m.data.it_postings.toLocaleString("en-IN")} />
        <Stat label="Skills tracked" value={String(m.data.skills_in_taxonomy)} />
        <Stat label="Period" value={m.data.period ?? "–"} />
      </div>
      <ul className="grid gap-3 text-sm md:grid-cols-3">
        {[
          { icon: BrainCircuit, t: "1 · Job-role classifier", d: "Supervised text classification (11 job roles)" },
          { icon: Database, t: "2 · Requirement-level classifier", d: "Supervised sentence classification (required / preferred / neutral)" },
          { icon: Layers, t: "3 · Skill recommender", d: "Unsupervised: Apriori association rules + K-Means clustering" },
        ].map((x) => (
          <li key={x.t} className="flex gap-3 rounded-lg border border-ink-200 bg-white p-4"><x.icon className="h-5 w-5 shrink-0 text-brand-600" /><div><b className="text-ink-900">{x.t}</b><p className="text-ink-500">{x.d}</p></div></li>
        ))}
      </ul>

      {m.pipeline && <PipelineCard p={m.pipeline} />}

      <ClassifierSection n={1} title="Job-role classifier" report={m.role_classifier}
        purpose="Predicts which job roles a syllabus and each of its courses prepare students for; assigns a role to job postings with vague titles, which powers the 'target job role' filter." />
      <ClassifierSection n={2} title="Requirement-level classifier" report={m.requirement_classifier}
        purpose="Labels job-description sentences without a heading as required, preferred or neutral; required skills weigh more in the demand score." />

      <section className="space-y-4">
        <h2 className="flex items-center gap-2 font-serif text-xl font-semibold text-ink-900">
          <span className="flex h-7 w-7 items-center justify-center rounded-full bg-brand-600 text-sm text-white">3</span>Skill recommender
        </h2>
        <div className="grid gap-6 lg:grid-cols-2">
          <Card title={<span className="inline-flex items-center gap-1.5"><GitBranch className="h-4 w-4 text-violet-600" />Apriori association rules</span>}
            subtitle={`${ap.transactions.toLocaleString("en-IN")} IT postings as skill baskets · min support ${ap.min_support * 100}% · min confidence ${ap.min_confidence * 100}% · min lift ${ap.min_lift}`}>
            <p className="mb-3 text-sm text-ink-700">Frequent itemsets: {ap.frequent_itemsets["1"]} single skills, {ap.frequent_itemsets["2"]} pairs, {ap.frequent_itemsets["3"]} triples → <b>{ap.rule_count.toLocaleString("en-IN")} rules</b>. Used to explain recommendations ("students already learn X; jobs asking for X also ask for Y").</p>
            <div className="max-h-96 overflow-auto">
              <table className="min-w-full text-sm">
                <thead className="sticky top-0 border-b border-ink-200 bg-white"><tr><th className="th">If a job asks for</th><th className="th">it also asks for</th><th className="th text-right">Conf.</th><th className="th text-right">Lift</th></tr></thead>
                <tbody className="divide-y divide-ink-100">
                  {ap.top_rules.map((r, i) => (
                    <tr key={i}><td className="td">{r.if.join(" + ")}</td><td className="td font-medium text-ink-900">{r.then}</td><td className="td text-right tabular-nums">{pct(r.confidence)}</td><td className="td text-right tabular-nums">{r.lift.toFixed(2)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <Card title={<span className="inline-flex items-center gap-1.5"><Layers className="h-4 w-4 text-brand-600" />K-Means skill bundles</span>}
            subtitle={`${km.postings_clustered.toLocaleString("en-IN")} postings clustered on TF-IDF skill vectors · k chosen by silhouette score`}>
            {sil && <Plot label="Silhouette score by k" data={sil} height={220}
              layout={{ margin: { l: 50, r: 10, t: 10, b: 45 }, xaxis: { title: { text: "number of clusters k" }, dtick: 1, gridcolor: "#eef1f6" }, yaxis: { title: { text: "silhouette" }, gridcolor: "#eef1f6" }, showlegend: false }} />}
            <p className="mt-2 text-sm text-ink-700">Chosen <b>k = {km.k}</b> (highest silhouette, shown in red).</p>
            <ul className="mt-3 space-y-2 text-sm">
              {km.clusters.map((c) => (
                <li key={c.id} className="rounded-md border border-ink-100 p-2.5">
                  <div className="flex flex-wrap justify-between gap-2"><b className="text-ink-900">{c.name}</b><span className="text-xs text-ink-500">{pct(c.share)} of postings · mostly {c.main_role}</span></div>
                  <div className="mt-1 text-xs text-ink-500">{c.top_skills.map((s) => `${s.skill} ${Math.round(s.frequency * 100)}%`).join(" · ")}</div>
                </li>
              ))}
            </ul>
          </Card>
        </div>
      </section>
      <p className="text-xs text-ink-500">Models trained {new Date(m.built_at).toLocaleString("en-IN")} in {m.training_minutes} minutes.</p>
    </div>
  );
}
