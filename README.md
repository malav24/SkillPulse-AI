# SkillPulse AI

**Is your syllabus teaching what industry needs?**

A faculty member uploads the PDF of a course or programme syllabus. SkillPulse:

1. **reads the PDF**: finds the courses, units, outcomes and labs, and the skills each one teaches;
2. **compares it with 1.3 lakh real job postings using three trained machine-learning models**;
3. **gives recommendations**: which skills to *add*, *strengthen* or *support with practical work*,
   explained with association rules and example job titles, plus clear Plotly charts.

```
                       ┌──────────── offline: python -m ml.train_all (≈10 min) ────────────┐
Real job postings ───► │ MODEL 2 requirement classifier → skills + levels per posting      │
(data/market/*.csv.gz) │ MODEL 1 job-role classifier    → role per posting                 │──► backend/artifacts/
                       │ MODEL 3 Apriori rules + K-Means skill bundles                      │
                       └────────────────────────────────────────────────────────────────────┘
Syllabus PDF ──► text ──► courses, units, outcomes, labs ──► skills (evidence) ──┐
                                                                                 ├──► coverage vs demand (seconds)
artifacts (loaded once at start-up) ─────────────────────────────────────────────┘     ──► roles, bundles,
                                                                                            recommendations, charts
```

## Machine-learning models

| # | Model | Type | Algorithms compared (best kept) | Trained on | Used for |
|---|---|---|---|---|---|
| 1 | **Job-role classifier** | Supervised, 11 classes | Complement Naive Bayes, Logistic Regression, **Linear SVM**, Random Forest (+ majority baseline) | Skills + description of every posting (TF-IDF 1–2-grams); labels from job-title rules, title removed from the input | Predicts which job roles a syllabus and each course prepare for; labels postings with vague titles; powers the *target job role* filter |
| 2 | **Requirement-level classifier** | Supervised, 3 classes | Multinomial NB, Logistic Regression, **Linear SVM**, Random Forest (+ baseline) | ~1.5 lakh job-ad sentences labelled by their heading (Required / Good to have / Responsibilities), heading words removed | Labels sentences without a heading as required / preferred / neutral; required skills weigh more in demand |
| 3 | **Skill recommender** | Unsupervised | **Apriori** association rules (support ≥ 1%, up to 3 skills) + **K-Means** (k by silhouette) | Skill sets of the IT postings | "Students already learn Java; 41% of Java jobs also ask for Spring (lift 3.2)"; readiness for each skill bundle |

Test-set accuracy, macro-F1, per-class results, confusion matrices, rules and clusters are on the
**ML models** page of the app (`GET /api/models`, from `backend/artifacts/metrics.json`).

### Team modules (3 members)

| Module | Code | Responsibility |
|---|---|---|
| **Member 1: Syllabus intelligence + Model 1** | `app/services/pdf_processing.py`, `structure_parser.py`, `pipeline.py`, `skill_extractor.py`, `ml/role_classifier.py` | PDF reading/OCR, course & unit detection, skill extraction with evidence, job-role classifier |
| **Member 2: Job-market data + Model 2** | `ml/common.py`, `ml/requirement_classifier.py`, `ml/evaluation.py`, `app/services/job_skills.py`, `job_import.py`, `app/services/market_model.py`, `scripts/build_latest_datasets.py` | Dataset collection & cleaning, skill extraction from job ads, requirement-level classifier, evaluation, market index |
| **Member 3: Recommendation engine + Model 3 + UI** | `ml/skill_recommender.py`, `app/services/comparison.py`, `app/services/faculty.py`, `frontend/` | Apriori + K-Means, coverage/demand/gap scoring, recommendations, Plotly dashboard, ML-models page |

There is no login. Open the app, upload a PDF, read the result.

---

## 1. Run it (for teammates: 3 steps)

**Install once:** [Python 3.12](https://www.python.org/downloads/) (3.11 and 3.13 also work; tick *Add python.exe
to PATH*), [Node.js 20 LTS](https://nodejs.org/) and [Git](https://git-scm.com/).

```powershell
git clone https://github.com/malav24/SkillPulse-AI.git
cd SkillPulse-AI
```

| Step | Windows (double-click) | macOS / Linux |
|---|---|---|
| 1. Install everything (once, ≈7 min) | `setup.bat` | `bash setup.sh` |
| 2. Start the app (opens the browser) | `start.bat` | `bash start.sh` |
| 3. Try it | upload `sample_data/JNTUH_R22_BTech_CSE_AIML_II_Year_Syllabus.pdf` | same |

Optional: `train_models.bat` (retrain the 3 ML models, ≈10 min) and `run_tests.bat` (all tests).

**Where is everything?** See [`PROJECT_STRUCTURE.txt`](PROJECT_STRUCTURE.txt): datasets, trained models, code
and which team member owns which module.

The trained models are already included (`backend/artifacts/`), so the app works immediately and an analysis
takes a few seconds. App: <http://localhost:5173> · API docs: <http://localhost:8000/docs>.

<details><summary>Manual commands (instead of the scripts)</summary>

```powershell
# Terminal 1: backend
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000

# Terminal 2: frontend
cd frontend
npm install
npm run dev

# Retrain the models (backend, venv active)
python -m ml.train_all
```
</details>

## 2. Using it

| Step | What the faculty member does | What SkillPulse shows |
|---|---|---|
| 1 | Chooses the syllabus PDF, types the course/programme name, optionally the institution, picks the **target job role** (all IT jobs, a specific role such as *Data Scientist / ML Engineer*, or all jobs) and optionally **entry-level jobs only** | Progress: reading the PDF → comparing with the trained models |
| 2 | Opens the result | **Industry match %**, well-covered / missing counts, courses found |
| 3 | Reads the **recommendations** | Up to 12 actions, most important first, e.g. *"Add to syllabus: Python, asked for in 25% of the relevant job postings but not found in this syllabus"*, with example job titles |
| 4 | Reviews the **ML insights** | Job roles the syllabus prepares for (Model 1) · readiness radar for the K-Means skill bundles (Model 3) · each course's predicted role |
| 5 | Reviews the **charts** (Plotly; hover for details; camera icon saves a PNG) | Top 20 requested skills coloured by coverage · coverage donut · demand vs coverage (focus area shaded) · course × skill heatmap |
| 6 | Checks **what the syllabus covers** | Skills per course; skills backed by a lab/project are underlined |
| 7 | **Print / save as PDF** | A clean printable report |

Previous analyses are listed on the home page and can be removed. Uploading the same PDF again reuses
the already-read document.

### What the results mean

| Result | Meaning | Recommendation |
|---|---|---|
| **Well covered** | Taught at roughly the level industry asks for | none |
| **Partly covered** | Taught, but coverage is low compared with demand | *Strengthen* |
| **Needs practical work** | Only mentioned as a topic (no lab, project or outcome) while demand is high | *Add practical work* |
| **Missing** | Employers ask for it; it was not found in the PDF | *Add to syllabus* |
| **Unclear** | Not found, but a related topic is taught (e.g. C# vs C Programming) | *Check* |

"Missing" means *not found in the uploaded PDF*. The skill may still be taught in practice or in
another year's syllabus. Upload all years for a complete picture.

### How the scores are calculated

- **Demand D(s)**: the share of the compared job postings that ask for skill *s* (required = 1.0,
  unspecified = 0.75, preferred = 0.5; levels from headings or Model 2), scaled 0–100 against the
  most requested skill.
- **Coverage C(s)**: 0–100 from where the skill appears in the syllabus: topic 35, learning outcome 25,
  lab/practical 20, project 10, depth 10. Mentions only in book lists count less.
- **Gap G(s) = max(0, D − C)** decides the result category and the order of recommendations.
- **Industry match %** = (well covered + ½ × partly covered) ÷ skills compared.

## 3. The industry job data

`backend/data/market/` holds the real job postings as compressed `.csv.gz` files (137,929 rows, 2024–2025; 130,289 after removing exact
duplicates). They are used only for training (`python -m ml.train_all`), which writes a compact market
index and the models to `backend/artifacts/`. Every analysis compares against that index, filtered by the
chosen job role (as assigned by Model 1) and, optionally, to entry-level postings (0–2 years).

| File | Postings | Source (credit) |
|---|---|---|
| `job_postings_2025.csv.gz` | 97,929 (collected Oct 2025, with full job descriptions) | Kaggle `shivamshrivastava21/indian-job-market-dataset-2025-2026`, via GitHub `Annusharma30/Job-Market-Intelligence-and-Skill-Analysis`. Licence not stated by the publisher; check before redistributing. |
| `job_postings_2024.csv.gz` | 40,000 (Jan 2024 sample, skill lists) | Kaggle `asaniczka/1-3m-linkedin-jobs-and-skills-2024` (ODC-By 1.0, attribution required), via GitHub release `Agnesor/linkedin-skills@data-v1` |

- **Rebuilding the files:** both files are rebuilt from the downloads by
  `python -m scripts.build_latest_datasets` (needs `pip install pyarrow`).
- **Adding your own job data:** put a CSV (or `.csv.gz`) in `data/market/` with at least a `job_title` column, plus any
  of `job_description`, `key_skills`, `experience`, `posting_date`. Then run `python -m ml.train_all` and
  restart the backend.

## 4. Project layout

```
backend/
  ml/                       training (offline)
    common.py               load + de-duplicate postings, taxonomy, role labels from titles
    requirement_classifier.py  MODEL 2
    role_classifier.py      MODEL 1
    skill_recommender.py    MODEL 3 (Apriori + K-Means)
    evaluation.py           accuracy, precision, recall, F1, confusion matrix
    train_all.py            runs everything, writes artifacts/
  artifacts/                trained models + market index + metrics.json (shipped)
  app/
    main.py, routers/faculty.py        FastAPI app and API
    services/
      faculty.py            upload → read → compare → store result (background worker)
      comparison.py         coverage, demand, gap, roles, bundles, recommendations
      market_model.py       loads artifacts once; demand per job role in milliseconds
      pdf_processing.py, structure_parser.py, pipeline.py, skill_extractor.py   syllabus reading
      job_skills.py, job_import.py                                              job-ad parsing helpers
  data/                     skill taxonomy (188 skills), parsing rules, market CSVs
  tests/                    backend tests
frontend/src/
  pages/Home.tsx            upload + previous analyses
  pages/Analysis.tsx        results, ML insights, recommendations, Plotly charts
  pages/Models.tsx          how the models were trained and how accurate they are
sample_data/                the JNTUH syllabus PDF for trying it out
```

## 5. API

| Method & path | Purpose |
|---|---|
| `GET /api/health` | server and database check |
| `GET /api/market` | number of postings, period, and the job roles you can compare with |
| `GET /api/models` | training data, algorithm comparison, test metrics, confusion matrices, rules, clusters |
| `POST /api/analyses` | multipart: `file` (PDF), `course_name`, `institution` (optional), `job_field` = `it`, `all` or a job role, `entry_level` = true/false |
| `GET /api/analyses` | previous analyses with their summary |
| `GET /api/analyses/{id}` | progress, or the full result: summary, skills, recommendations, courses |
| `POST /api/analyses/{id}/retry` | re-run a failed analysis |
| `DELETE /api/analyses/{id}` | remove from history |

Interactive docs: <http://localhost:8000/docs>.

## 6. Configuration (`backend/.env`, all optional)

`DATABASE_URL` (SQLite by default; PostgreSQL works), `MAX_UPLOAD_MB` (20), `OCR_ENABLED`,
`TESSERACT_CMD` (path to Tesseract for scanned PDFs), `ARTIFACTS_DIR`. See `.env.example`.

**Scanned PDFs:** read automatically with the built-in OCR (RapidOCR, installed by `requirements.txt`, models
included, about 3 seconds per page). If Tesseract OCR is installed and on PATH (or `TESSERACT_CMD` is set) it is
used instead. A PDF with neither text nor images is reported as blank or damaged.
**Related-skill matching:** `pip install -r requirements-optional.txt` adds a Sentence-Transformers
model. Without it a TF-IDF fallback is used.

## 7. Tests

```powershell
cd backend;  pytest -q          # 42 tests: PDF reading + OCR, structure, skill extraction, job data, ML models, full workflow
cd frontend; npm test           # 6 tests: upload form, history, results page, filters, progress/failure, ML page
```

## 8. Docker

`docker compose up --build`, then open <http://localhost:8080> (PostgreSQL + backend with Tesseract +
frontend). This has not been run in the development environment.

## 9. Limitations

- **Skill list:** skills are matched against a list of 188 computing and engineering skills, so
  non-technical syllabi get little coverage.
- **Rule-based structure detection:** works well for JNTU-style syllabi. Other formats may need new
  rules in `data/parsing_rules.json`.
- **Not exhaustive:** a skill missing from the PDF may still be taught. Treat recommendations as
  input for the faculty's own judgement.
- **Dated job data:** the job data is from 2024–2025, not live. Add newer CSV files to `data/market/` and
  retrain to refresh it.
- **Weak labels:** Model 1's labels come from job-title rules and Model 2's from job-ad headings, so both
  contain noise. Model 2 in particular is about 59% accurate on three classes, well above the
  majority-class baseline but far from perfect. The metrics page shows the exact numbers.
- **Single worker:** heavy work runs one task at a time in a background thread, which suits a
  department-level tool rather than many simultaneous users.
