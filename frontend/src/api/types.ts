export type AnalysisStatus = "queued" | "processing" | "done" | "failed";
export type SkillStatusKey = "covered" | "partially_covered" | "underemphasized" | "potentially_missing" | "uncertain" | "not_applicable";

export interface Market {
  state: "idle" | "loading" | "ready" | "failed" | "missing";
  message: string | null;
  postings: number;
  period: string | null;
  job_fields: { value: string; label: string; postings: number }[];
  max_upload_mb: number;
}

export interface Summary {
  pages: number;
  courses: number;
  skills_taught: number;
  industry_skills_compared: number;
  well_covered: number;
  partly_covered: number;
  missing: number;
  unclear: number;
  match_pct: number;
  top_role: RoleProb | null;
}

export interface SkillRow {
  skill: string;
  group: string;
  group_key: string;
  status_key: SkillStatusKey;
  status: string;
  demand_pct: number;
  jobs: number;
  coverage: number;
  gap: number;
  in_syllabus_as: string[];
  required_pct: number;
  preferred_pct: number;
}

export interface Recommendation {
  skill: string;
  demand_pct: number;
  priority: "High" | "Medium" | "Low";
  action: string;
  kind: "add" | "practice" | "strengthen" | "check";
  text: string;
  example_jobs: string[];
  required_pct: number;
  rule: { if: string[]; then: string; support: number; confidence: number; lift: number } | null;
  top_role?: { role: string; share: number } | null;
  ml?: string[];
}

export interface RoleProb {
  role: string;
  probability: number;
}

export interface CourseSkills {
  name: string;
  skills: { skill: string; mentions: number; practical: boolean }[];
  role: RoleProb | null;
}

export interface Bundle {
  name: string;
  share: number;
  main_role: string;
  readiness: number;
  covered: string[];
  missing: string[];
}

export interface Analysis {
  id: number;
  course_name: string;
  institution: string | null;
  job_field: string;
  job_field_label: string;
  entry_level: boolean;
  status: AnalysisStatus;
  step: string | null;
  message: string | null;
  filename: string | null;
  created_at: string;
  finished_at: string | null;
  summary?: Summary;
  market?: { postings_compared: number; postings: number; period: string | null };
  skills?: SkillRow[];
  recommendations?: Recommendation[];
  courses?: CourseSkills[];
  roles?: RoleProb[];
  bundles?: Bundle[];
  models?: {
    role_classifier: { algorithm: string; accuracy: number };
    requirement_classifier: { algorithm: string; accuracy: number };
    recommender: { rules: number; clusters: number };
  };
  warnings?: string[];
  note?: string;
}

export interface ModelEval {
  accuracy: number;
  macro_f1: number;
  weighted_f1: number;
  per_class: Record<string, { precision: number; recall: number; f1: number; support: number }>;
  confusion_matrix: number[][];
  train_seconds: number;
}

export interface ClassifierReport {
  task: string;
  labels: string[];
  train_size: number;
  test_size: number;
  class_counts: Record<string, number>;
  features: string;
  label_source?: string;
  models: Record<string, ModelEval>;
  best_model: string;
}

export interface Rule {
  if: string[];
  then: string;
  support: number;
  confidence: number;
  lift: number;
}

export interface Pipeline {
  datasets: { files: Record<string, number>; raw_rows: number; description: string };
  preprocessing: { step: string; count: number | string }[];
  features: { model: string; features: string }[];
  training: { model: string; split: string; algorithms: string[]; selected: string; selection: string }[];
}

export interface Metrics {
  pipeline?: Pipeline;
  built_at: string;
  training_minutes: number;
  data: { postings_loaded: number; it_postings: number; skills_in_taxonomy: number; posting_skill_links: number; period: string | null };
  role_classifier: ClassifierReport & { final_role_counts: Record<string, number> };
  requirement_classifier: ClassifierReport;
  skill_recommender: {
    apriori: { min_support: number; min_confidence: number; min_lift: number; transactions: number; frequent_itemsets: Record<string, number>; rule_count: number; top_rules: Rule[] };
    kmeans: { k: number; silhouette: Record<string, number>; inertia: Record<string, number>; postings_clustered: number;
      clusters: { id: number; size: number; share: number; name: string; top_skills: { skill: string; frequency: number }[]; main_role: string; main_role_share: number }[] };
  };
}
