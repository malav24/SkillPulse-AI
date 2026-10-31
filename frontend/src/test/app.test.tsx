import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Analysis } from "../api/types";
import AnalysisPage from "../pages/Analysis";
import Home from "../pages/Home";
import ModelsPage from "../pages/Models";

vi.mock("../components/Plot", () => ({ default: ({ label }: { label: string }) => <div data-testid="plot">{label}</div> }));

const MARKET = { state: "ready", message: null, postings: 137929, period: "2024–2025", max_upload_mb: 20,
  job_fields: [{ value: "it", label: "All Software & IT jobs", postings: 40000 }, { value: "Data Scientist / ML Engineer", label: "Data Scientist / ML Engineer", postings: 5000 },
    { value: "all", label: "All jobs (any industry)", postings: 130289 }] };

const DONE: Analysis = {
  id: 7, course_name: "B.Tech CSE — II Year", institution: "Test University", job_field: "it", job_field_label: "All Software & IT jobs", entry_level: false,
  status: "done", step: "done", message: "Done", filename: "syllabus.pdf", created_at: "2026-10-25T10:00:00Z", finished_at: "2026-10-25T10:01:00Z",
  summary: { pages: 29, courses: 2, skills_taught: 12, industry_skills_compared: 3, well_covered: 1, partly_covered: 1, missing: 1, unclear: 0, match_pct: 50, top_role: { role: "Software Developer", probability: 0.4 } },
  market: { postings_compared: 9638, postings: 137929, period: "2024–2025" },
  skills: [
    { skill: "Python", group: "Programming language", group_key: "programming_language", status_key: "potentially_missing", status: "Missing", demand_pct: 25.5, jobs: 2400, coverage: 0, gap: 100, in_syllabus_as: [], required_pct: 40, preferred_pct: 5 },
    { skill: "SQL", group: "Database", group_key: "database", status_key: "partially_covered", status: "Partly covered", demand_pct: 21.1, jobs: 2000, coverage: 58, gap: 25, in_syllabus_as: ["topic"], required_pct: 30, preferred_pct: 5 },
    { skill: "Java", group: "Programming language", group_key: "programming_language", status_key: "covered", status: "Well covered", demand_pct: 19, jobs: 1800, coverage: 82, gap: 0, in_syllabus_as: ["topic", "practical"], required_pct: 50, preferred_pct: 5 },
  ],
  recommendations: [
    { skill: "Python", demand_pct: 25.5, priority: "High", action: "Add to syllabus", kind: "add", text: "Python is asked for in 25.5% of the relevant job postings but was not found in this syllabus.", example_jobs: ["Python Developer"], required_pct: 40,
      rule: { if: ["SQL"], then: "Python", support: 0.05, confidence: 0.42, lift: 1.8 }, top_role: { role: "Data Engineer", share: 0.4 },
      ml: ["Demand weighted by Linear SVM (required vs preferred: 40% required)", "Apriori rule: SQL → Python (confidence 42%, lift 1.8)"] },
  ],
  courses: [
    { name: "DATABASE MANAGEMENT SYSTEMS", skills: [{ skill: "SQL", mentions: 4, practical: true }], role: { role: "Data Engineer", probability: 0.5 } },
    { name: "OBJECT ORIENTED PROGRAMMING THROUGH JAVA", skills: [{ skill: "Java", mentions: 9, practical: true }], role: null },
  ],
  roles: [{ role: "Software Developer", probability: 0.4 }, { role: "Data Engineer", probability: 0.3 }],
  bundles: [
    { name: "Java · Spring · SQL", share: 0.2, main_role: "Software Developer", readiness: 60, covered: ["Java"], missing: ["Spring"] },
    { name: "Python · ML · SQL", share: 0.2, main_role: "Data Scientist / ML Engineer", readiness: 30, covered: [], missing: ["Python"] },
    { name: "React · CSS · HTML", share: 0.2, main_role: "Web / Full Stack Developer", readiness: 40, covered: [], missing: [] },
  ],
  models: { role_classifier: { algorithm: "Logistic Regression", accuracy: 0.81 }, requirement_classifier: { algorithm: "Linear SVM", accuracy: 0.6 },
    recommender: { rules: 300, clusters: 9 } },
  warnings: [],
  note: "\"Missing\" means the skill was not found in the uploaded PDF.",
};

function mockFetch(routes: Record<string, unknown>, onPost?: (url: string, body: FormData) => unknown) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    if (init?.method === "POST" && onPost) {
      return new Response(JSON.stringify(onPost(url, init.body as FormData)), { status: 201 });
    }
    const key = Object.keys(routes).find((k) => url.endsWith(k));
    return new Response(JSON.stringify(key ? routes[key] : { error: { message: "not found" } }), { status: key ? 200 : 404 });
  });
}

afterEach(() => vi.restoreAllMocks());

describe("Home", () => {
  it("shows the industry reference and the history", async () => {
    mockFetch({ "/api/market": MARKET, "/api/analyses": [DONE] });
    render(<MemoryRouter><Home /></MemoryRouter>);
    expect(await screen.findByText(/1,37,929 real job postings/)).toBeInTheDocument();
    expect(await screen.findByText("B.Tech CSE — II Year")).toBeInTheDocument();
    expect(screen.getByText(/missing skills/)).toBeInTheDocument();
  });

  it("validates and uploads a PDF", async () => {
    const posted: FormData[] = [];
    mockFetch({ "/api/market": MARKET, "/api/analyses": [] }, (_url, body) => {
      posted.push(body);
      return { ...DONE, status: "queued", id: 9 };
    });
    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={["/"]}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/analysis/:id" element={<div>analysis page</div>} />
        </Routes>
      </MemoryRouter>,
    );
    await screen.findByText(/No analyses yet/);
    await user.click(screen.getByRole("button", { name: "Analyse syllabus" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Please choose the syllabus PDF.");
    const file = new File(["%PDF-1.4"], "Data_Structures.pdf", { type: "application/pdf" });
    await user.upload(screen.getByLabelText("Syllabus PDF"), file);
    expect(screen.getByLabelText("Course / programme name")).toHaveValue("Data Structures");
    await user.selectOptions(screen.getByLabelText("Compare with (target job role)"), "all");
    await user.click(screen.getByLabelText(/Only entry-level jobs/));
    await user.click(screen.getByRole("button", { name: "Analyse syllabus" }));
    expect(await screen.findByText("analysis page")).toBeInTheDocument();
    expect(posted[0].get("course_name")).toBe("Data Structures");
    expect(posted[0].get("job_field")).toBe("all");
    expect(posted[0].get("entry_level")).toBe("true");
  });
});

describe("Analysis page", () => {
  const renderAt = () =>
    render(
      <MemoryRouter initialEntries={["/analysis/7"]}>
        <Routes><Route path="/analysis/:id" element={<AnalysisPage />} /></Routes>
      </MemoryRouter>,
    );

  it("shows summary, recommendations, charts and course breakdown", async () => {
    mockFetch({ "/api/analyses/7": DONE });
    renderAt();
    expect(await screen.findByText("Recommendations for the syllabus")).toBeInTheDocument();
    expect(screen.getByText("Industry match").parentElement).toHaveTextContent("50%");
    expect(screen.getByText(/Python is asked for in 25.5%/)).toBeInTheDocument();
    expect(screen.getByText(/Python Developer/)).toBeInTheDocument();
    expect(screen.getAllByTestId("plot").map((p) => p.textContent)).toEqual(
      ["Predicted job roles", "Skill bundle readiness", "Top requested skills", "Coverage breakdown", "Demand versus coverage", "Skills by course"]);
    expect(screen.getByText("OBJECT ORIENTED PROGRAMMING THROUGH JAVA")).toBeInTheDocument();
    expect(screen.getByText("Data Engineer · 50%")).toBeInTheDocument();
    expect(screen.getByText("Apriori rule: SQL → Python (confidence 42%, lift 1.8)")).toBeInTheDocument();
    expect(screen.getByText(/Demand weighted by Linear SVM/)).toBeInTheDocument();
    expect(screen.getByText(/Logistic Regression, test accuracy 81.0%/)).toBeInTheDocument();
    expect(screen.getByText(/Compared with 9,638 real job postings \(All Software & IT jobs, 2024–2025\)/)).toBeInTheDocument();
  });

  it("filters the skills table by result", async () => {
    mockFetch({ "/api/analyses/7": DONE });
    const user = userEvent.setup();
    renderAt();
    await screen.findByText("All skills compared");
    await user.click(screen.getByRole("button", { name: "Missing (1)" }));
    const table = screen.getByRole("table");
    expect(table).toHaveTextContent("Python");
    expect(table).not.toHaveTextContent("Java");
  });

  it("shows progress while running and a retry when failed", async () => {
    mockFetch({ "/api/analyses/7": { ...DONE, status: "processing", step: "comparing", summary: undefined } });
    const { unmount } = renderAt();
    expect(await screen.findByText("Analysing your syllabus…")).toBeInTheDocument();
    expect(screen.getByText("Comparing with industry job postings using the trained ML models")).toBeInTheDocument();
    unmount();
    vi.restoreAllMocks();
    mockFetch({ "/api/analyses/7": { ...DONE, status: "failed", message: "This PDF is a scanned image", summary: undefined } });
    renderAt();
    expect(await screen.findByText("This PDF is a scanned image")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("button", { name: "Try again" })).toBeEnabled());
  });
});

const EVAL = (labels: string[], acc: number) => ({
  accuracy: acc, macro_f1: acc - 0.05, weighted_f1: acc, train_seconds: 3,
  per_class: Object.fromEntries(labels.map((l) => [l, { precision: acc, recall: acc, f1: acc, support: 10 }])),
  confusion_matrix: labels.map((_, i) => labels.map((__, j) => (i === j ? 8 : 1))),
});
const REQ = ["required", "preferred", "neutral"];
const METRICS = {
  pipeline: {
    datasets: { files: { "job_postings_2025.csv": 97929 }, raw_rows: 97929, description: "Real job postings" },
    preprocessing: [{ step: "Removed exact duplicate postings", count: 7640 }],
    features: [{ model: "Model 1: job-role classifier", features: "TF-IDF word 1–2-grams" }],
    training: [{ model: "Model 1", split: "800 train / 200 test", algorithms: ["Linear SVM"], selected: "Linear SVM", selection: "highest macro-F1" }],
  },
  built_at: "2026-10-25T12:00:00Z", training_minutes: 9.5,
  data: { postings_loaded: 130289, it_postings: 40000, skills_in_taxonomy: 188, posting_skill_links: 300000, period: "2024–2025" },
  role_classifier: { task: "Predict the job role", labels: ["Software Developer", "Other (non-IT)"], train_size: 800, test_size: 200,
    class_counts: {}, features: "TF-IDF", label_source: "titles", best_model: "Logistic Regression", final_role_counts: {},
    models: { "Logistic Regression": EVAL(["Software Developer", "Other (non-IT)"], 0.81), "Complement Naive Bayes": EVAL(["Software Developer", "Other (non-IT)"], 0.7) } },
  requirement_classifier: { task: "Classify sentences", labels: REQ, train_size: 800, test_size: 200, class_counts: {}, features: "TF-IDF",
    best_model: "Linear SVM", models: { "Linear SVM": EVAL(REQ, 0.6) } },
  skill_recommender: {
    apriori: { min_support: 0.01, min_confidence: 0.25, min_lift: 1.3, transactions: 40000, frequent_itemsets: { 1: 80, 2: 300, 3: 120 }, rule_count: 512,
      top_rules: [{ if: ["Java"], then: "Spring Framework", support: 0.05, confidence: 0.41, lift: 3.2 }] },
    kmeans: { k: 9, silhouette: { 6: 0.1, 9: 0.2 }, inertia: { 6: 10, 9: 8 }, postings_clustered: 30000,
      clusters: [{ id: 0, size: 100, share: 0.2, name: "Java · Spring · SQL", top_skills: [{ skill: "Java", frequency: 0.9 }], main_role: "Software Developer", main_role_share: 0.7 }] },
  },
};

describe("ML models page", () => {
  it("shows training data, algorithm comparison, rules and clusters", async () => {
    mockFetch({ "/api/models": METRICS });
    render(<MemoryRouter><ModelsPage /></MemoryRouter>);
    expect(await screen.findByText("Machine-learning models")).toBeInTheDocument();
    expect(screen.getByText("1,30,289")).toBeInTheDocument();
    expect(screen.getByText("How the models were built")).toBeInTheDocument();
    expect(screen.getByText("Removed exact duplicate postings")).toBeInTheDocument();
    expect(screen.getByText("7,640")).toBeInTheDocument();
    expect(screen.getAllByText(/Logistic Regression ★/).length).toBe(1);
    expect(screen.getByText("Spring Framework")).toBeInTheDocument();
    expect(screen.getByText(/k = 9/)).toBeInTheDocument();
    expect(screen.getAllByTestId("plot").length).toBe(5);
  });
});
