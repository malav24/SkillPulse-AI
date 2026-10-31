import { CheckCircle2, ChevronRight, FileText, Loader2, Trash2, Upload } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, errorMessage } from "../api/client";
import type { Analysis, Market } from "../api/types";
import { Button, Card, ErrorState, Spinner, cx } from "../components/ui";
import { useAsync } from "../components/useAsync";

const STEPS = [
  { n: 1, title: "Upload your syllabus", text: "The PDF of your course or programme curriculum." },
  { n: 2, title: "SkillPulse reads it", text: "It finds the courses, units, outcomes and labs, and the skills they teach." },
  { n: 3, title: "ML models compare", text: "Trained models match it against 1.3 lakh real job postings and predict the job roles it prepares for." },
  { n: 4, title: "You get recommendations", text: "Which skills to add, strengthen or support with practical work, and why." },
];

function MarketLine({ market }: { market: Market | null }) {
  if (!market) return null;
  if (market.state === "ready")
    return (
      <p className="flex items-center gap-1.5 text-sm text-ink-500">
        <CheckCircle2 className="h-4 w-4 text-teal-600" aria-hidden />
        Industry reference: {market.postings.toLocaleString("en-IN")} real job postings{market.period ? ` (${market.period})` : ""}.
      </p>
    );
  return <p className="text-sm text-red-700">{market.message ?? "The trained models were not found. Run: python -m ml.train_all"}</p>;
}

export default function Home() {
  const navigate = useNavigate();
  const market = useAsync(() => api.get<Market>("/api/market"), []);
  const history = useAsync(() => api.get<Analysis[]>("/api/analyses"), []);
  const [file, setFile] = useState<File | null>(null);
  const [course, setCourse] = useState("");
  const [institution, setInstitution] = useState("");
  const [field, setField] = useState("it");
  const [entry, setEntry] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const running = history.data?.some((a) => a.status === "queued" || a.status === "processing");
    if (!running) return;
    const t = setInterval(() => history.reload(true), 2000);
    return () => clearInterval(t);
  }, [history.data, history.reload]);

  const pick = (f: File | undefined | null) => {
    setError(null);
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".pdf")) {
      setError("Please choose a PDF file.");
      return;
    }
    setFile(f);
    if (!course) setCourse(f.name.replace(/\.pdf$/i, "").replace(/[_-]+/g, " "));
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!file) return setError("Please choose the syllabus PDF.");
    if (!course.trim()) return setError("Please enter the course or programme name.");
    setBusy(true);
    setError(null);
    const form = new FormData();
    form.append("file", file);
    form.append("course_name", course.trim());
    if (institution.trim()) form.append("institution", institution.trim());
    form.append("job_field", field);
    form.append("entry_level", String(entry));
    try {
      const a = await api.post<Analysis>("/api/analyses", form);
      navigate(`/analysis/${a.id}`);
    } catch (err) {
      setError(errorMessage(err));
      setBusy(false);
    }
  };

  const remove = async (id: number) => {
    try {
      await api.del(`/api/analyses/${id}`);
      history.reload(true);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-serif text-3xl font-semibold tracking-tight text-ink-900">Is your syllabus teaching what industry needs?</h1>
        <p className="mt-2 max-w-3xl text-ink-500">
          Upload a course or programme syllabus. SkillPulse finds the skills it teaches, compares them with real job postings and
          tells you which skills to add or strengthen.
        </p>
      </div>

      <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {STEPS.map((s) => (
          <li key={s.n} className="rounded-lg border border-ink-200 bg-white p-4">
            <div className="mb-2 flex h-7 w-7 items-center justify-center rounded-full bg-brand-50 text-sm font-semibold text-brand-700">{s.n}</div>
            <div className="text-sm font-semibold text-ink-900">{s.title}</div>
            <p className="mt-1 text-sm text-ink-500">{s.text}</p>
          </li>
        ))}
      </ol>

      <Card title="Analyse a syllabus">
        <form onSubmit={submit} className="space-y-5">
          <div
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files?.[0]); }}
            onClick={() => input.current?.click()}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
            role="button"
            tabIndex={0}
            className={cx(
              "flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-8 text-center transition-colors",
              drag ? "border-brand-400 bg-brand-50" : "border-ink-200 hover:border-brand-300 hover:bg-ink-50",
            )}
          >
            {file ? (
              <>
                <FileText className="mb-2 h-7 w-7 text-brand-600" aria-hidden />
                <div className="text-sm font-medium text-ink-900">{file.name}</div>
                <div className="text-xs text-ink-500">{(file.size / 1024 / 1024).toFixed(2)} MB · click to change</div>
              </>
            ) : (
              <>
                <Upload className="mb-2 h-7 w-7 text-ink-300" aria-hidden />
                <div className="text-sm font-medium text-ink-900">Drop the syllabus PDF here, or click to choose</div>
                <div className="text-xs text-ink-500">PDF up to {market.data?.max_upload_mb ?? 20} MB. Scanned PDFs are read with built-in OCR (a few seconds per page).</div>
              </>
            )}
            <input ref={input} type="file" accept="application/pdf,.pdf" className="hidden" aria-label="Syllabus PDF"
              onChange={(e) => pick(e.target.files?.[0])} />
          </div>

          <div className="grid gap-4 sm:grid-cols-3">
            <div className="sm:col-span-1">
              <label className="label" htmlFor="course">Course / programme name</label>
              <input id="course" className="input" value={course} onChange={(e) => setCourse(e.target.value)} placeholder="e.g. B.Tech CSE — II Year" />
            </div>
            <div>
              <label className="label" htmlFor="inst">Institution (optional)</label>
              <input id="inst" className="input" value={institution} onChange={(e) => setInstitution(e.target.value)} placeholder="e.g. JNTU Hyderabad" />
            </div>
            <div>
              <label className="label" htmlFor="field">Compare with (target job role)</label>
              <select id="field" className="input" value={field} onChange={(e) => setField(e.target.value)}>
                {(market.data?.job_fields ?? [{ value: "it", label: "All Software & IT jobs", postings: 0 }]).map((f) => (
                  <option key={f.value} value={f.value}>{f.label}{f.postings ? ` (${f.postings.toLocaleString("en-IN")})` : ""}</option>
                ))}
              </select>
            </div>
          </div>
          <label className="flex items-center gap-2 text-sm text-ink-700">
            <input type="checkbox" checked={entry} onChange={(e) => setEntry(e.target.checked)} className="h-4 w-4 rounded border-ink-300 text-brand-600" />
            Only entry-level jobs (0–2 years of experience): what fresh graduates are asked for
          </label>

          {error && <ErrorState message={error} />}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <MarketLine market={market.data} />
            <Button type="submit" loading={busy}>Analyse syllabus</Button>
          </div>
        </form>
      </Card>

      <Card title="Previous analyses">
        {history.loading ? (
          <Spinner />
        ) : history.error ? (
          <ErrorState message={history.error} onRetry={() => history.reload()} />
        ) : !history.data?.length ? (
          <p className="text-sm text-ink-500">No analyses yet. Upload a syllabus above to get started.</p>
        ) : (
          <ul className="divide-y divide-ink-100">
            {history.data.map((a) => (
              <li key={a.id} className="flex items-center gap-3 py-3">
                <Link to={`/analysis/${a.id}`} className="group flex min-w-0 flex-1 items-center gap-3">
                  <FileText className="h-5 w-5 shrink-0 text-ink-300" aria-hidden />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-medium text-ink-900 group-hover:text-brand-700">{a.course_name}</div>
                    <div className="truncate text-xs text-ink-500">
                      {[a.institution, a.job_field_label, a.filename, new Date(a.created_at).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" })].filter(Boolean).join(" · ")}
                    </div>
                  </div>
                  <div className="hidden text-right text-xs sm:block">
                    {a.status === "done" && a.summary ? (
                      <span className="text-ink-700"><b className="text-ink-900">{a.summary.match_pct}%</b> match · {a.summary.missing} missing skills</span>
                    ) : a.status === "failed" ? (
                      <span className="text-red-700">Failed</span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-brand-700"><Loader2 className="h-3 w-3 animate-spin" aria-hidden />In progress</span>
                    )}
                  </div>
                  <ChevronRight className="h-4 w-4 text-ink-300" aria-hidden />
                </Link>
                {(a.status === "done" || a.status === "failed") && (
                  <button onClick={() => remove(a.id)} className="rounded p-1.5 text-ink-300 hover:bg-ink-100 hover:text-red-700" aria-label={`Remove ${a.course_name}`} title="Remove from history">
                    <Trash2 className="h-4 w-4" />
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}
