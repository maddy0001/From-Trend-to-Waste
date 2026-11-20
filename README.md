# From Trend to Waste

**A data-driven analysis of fast-fashion consumer behaviour and excess-inventory risk**

M.Tech course project · 21CSE668T Business Data Analytics · Department of Data Science and Business Systems, SRM Institute of Science and Technology, Kattankulathur (November 2026)

**Team:** Madhusuthanan G · Pharos Sophy Samuel T J · Sricharan K
**Guide:** Dr. S. Srividhya

---

## What this project is about

Fast fashion's environmental cost comes from making more clothes than people need. We look at where that excess shows up in a retailer's own numbers: **stock that is delivered to shops but never sold**.

All retail data comes from **one retailer, Nuna Lie** (a mid-sized Italian fast-fashion chain), through the public research dataset **VISUELLE 2.0**. It covers 5,355 products, 110 shops and 6 seasons (2017–2019), and holds weekly sales, deliveries, prices, discounts and 3.18 million loyalty-card purchases. Our own consumer survey (122 responses) covers what happens to clothes after they are bought.

The analysis follows three moments in a product's life:

| Moment | Question | Methods |
|---|---|---|
| Before launch | Will this product end with a lot of leftover stock? | Logistic regression (odds ratios), decision tree, gradient boosting |
| After 2 weeks of sales | How much will sell in weeks 3–12, and how much will be left over? | Naive baseline, lifecycle profile, Poisson regression, gradient boosting (± Google Trends) |
| End of season | Which sales patterns exist? When do markdowns happen? Who buys? What does it cost? | k-means on sales curves, markdown detection, RFM segments + CLTV, financial statements |

Models are always **trained on 2017–2018 and tested on the unseen 2019 seasons**.

### Key results

| Finding | Number |
|---|---|
| Delivered stock still unsold after 12 weeks | **31.6%** (736,546 of 2,334,430 units) |
| Products whose best week is in weeks 1–3 | **70%** |
| Median first markdown | **week 8**, when demand is down to 53% of peak |
| Pre-launch risk model / week-2 model (ROC-AUC) | **0.69 / 0.71** |
| Week-2 forecast error (WAPE), gradient boosting vs naive | **39% vs 60%** |
| Loyal "Frequent, active" segment | 15% of customers buy **50%** of units |
| Margin on goods sold vs if unsold stock is lost (cost = 50% of price) | **44% → 19%** |
| **Week-2 rule** replayed on 2019: unsold stock / revenue / gross profit | **−11.9% / −0.7% / +8.7%** |

**The week-2 rule:** two weeks after launch, forecast each product's season. If predicted sell-through is below 60%, cut its remaining deliveries by 30% and mark it down 20% from week 4. A randomised pilot needs only 8 flagged products per group to prove it.

---

## Repository contents

```
From-Trend-to-Waste/
├── src/                    Analysis pipeline (Python)
│   ├── run_all.py            runs every step below in order
│   ├── config.py             paths, seasons, thresholds (60% sell-through, 15% markdown, ...)
│   ├── vis_forecast.py       leftover stock, week-2 forecasting, what-if simulation
│   ├── vis_lifecycle.py      demand lifecycles (k-means), markdown timing
│   ├── vis_risk_model.py     pre-launch excess-stock risk model
│   ├── vis_customers.py      RFM customer segments, CLTV, buying-frequency bands
│   ├── trends.py             seasonality, demand trendline, Google Trends test
│   ├── pilot_design.py       randomised pilot + power analysis
│   ├── vis_financials.py     financial statements built from the data
│   ├── survey_analysis.py    survey descriptives, conjoint, shopper segments
│   ├── data_quality.py       data overview, 18-check cleaning log, EDA tables
│   ├── review2_data.py       data for the mid-project (Review 2) dashboard
│   ├── literature.py, env_factors.py, utils.py     helpers and fixed facts
│   ├── build_report.py, deck_series.py, export_pdf.py,
│   │   dashboard_screenshots.py, review2_screens.py  report/deck/PDF/screenshot builders
│   └── colab_pack.py         builds the private data pack for Google Colab
├── dashboard/              Streamlit apps
│   ├── app.py                early-warning dashboard (7 tabs)
│   ├── demo_app.py           guided 9-step demo for presenting
│   └── review2_app.py        Review 2 dashboard
├── outputs/                Results produced by the pipeline (committed, so dashboards work out of the box)
│   ├── tables/               CSV/JSON results behind every number
│   ├── figures/              all charts (PNG), incl. dashboard screenshots
│   ├── models/               saved risk model (vis_risk_logit.joblib)
│   └── review2/              Review 2 dashboard data
├── survey/conjoint_design.csv   the 8 garment profiles used in the conjoint survey
├── deliverables/           Final report, slide decks (PPTX + PDF), summary, Colab notebooks
├── requirements.txt
└── README.md
```

**Not in the repository (on purpose):** the raw VISUELLE 2.0 files (`visuelle2/`), the survey responses and processed tables (`data/`), and the Colab data pack. The dataset licence does not allow redistribution, and the survey responses are personal data. They are listed in `.gitignore`.

---

## Requirements

| Tool | Needed for | Get it |
|---|---|---|
| **Python 3.13** (3.12–3.14 should also work) | everything | https://www.python.org/downloads/ (on Windows, tick **"Add python.exe to PATH"** during install) |
| **Git** | cloning / pushing | https://git-scm.com/download/win |
| ~8 GB RAM | full pipeline (peaks around 2.5 GB) | |
| Google Chrome, LibreOffice, Node.js | *optional*: dashboard screenshots and PDF copies | skipped automatically if missing |

---

## Quick start on Windows (PowerShell)

> All commands are for **PowerShell**. Open it from the Start menu, or in VS Code via *Terminal → New Terminal*.
> The path `C:\Users\THE GOD\...` contains a space, so always keep it in **quotes**.

### 1. Go to the project folder

```powershell
cd "C:\Users\THE GOD\Documents\Github\From-Trend-to-Waste"
```

(If you are cloning from GitHub instead: `cd "C:\Users\THE GOD\Documents\Github"` then `git clone https://github.com/<your-username>/From-Trend-to-Waste.git` and `cd From-Trend-to-Waste`.)

### 2. Create a virtual environment and install the libraries

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

You should now see `(.venv)` at the start of the prompt.

If PowerShell says *"running scripts is disabled on this system"*, run this once and then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 3. Open a dashboard (works straight away, no raw data needed)

```powershell
python -X utf8 -m streamlit run dashboard\app.py
```

Your browser opens at http://localhost:8501. Press `Ctrl + C` in the terminal to stop it.

Other dashboards:

```powershell
python -X utf8 -m streamlit run dashboard\review2_app.py   # Review 2 dashboard
python -X utf8 -m streamlit run dashboard\demo_app.py      # guided demo: needs the full pipeline run first (step 5)
```

> **Why `-X utf8`?** Windows uses a different default text encoding than macOS/Linux. This flag makes Python read and write files as UTF-8, as on the Mac where the project was built. Use it for every command here, or set it permanently with `setx PYTHONUTF8 1` (then open a new terminal).

### 4. Get the raw data (only for re-running the analysis)

The dashboards above use the saved results in `outputs/`. To recompute everything from scratch, you need the raw data in the project root:

```
From-Trend-to-Waste/
├── visuelle2/            sales.csv, restocks.csv, price_discount_series.csv,
│                         customer_data.csv, vis2_gtrends_data.csv
└── data/raw/survey/      survey_responses_<date>.csv  (Google Forms -> Download CSV)
```

- **Team members:** the private data pack `deliverables\colab\From_Trend_to_Waste_colab_pack.zip` (shared privately, never on GitHub) holds all of these. Put it in `deliverables\colab\`, then unpack only the data folders:

  ```powershell
  python -c "import zipfile; z = zipfile.ZipFile(r'deliverables\colab\From_Trend_to_Waste_colab_pack.zip'); [z.extract(n) for n in z.namelist() if n.startswith(('visuelle2/', 'data/'))]"
  ```

- **Everyone else:** request VISUELLE 2.0 through its request form at https://humaticslab.github.io/forecasting/visuelle, and put the five CSV files above in `visuelle2\`. Without survey responses the survey steps are skipped.

### 5. Run the full pipeline

```powershell
python -X utf8 -m src.run_all
```

This takes about 1–3 minutes. It recomputes every table, figure and model in `outputs/`, and writes processed tables to `data/processed/`, which the demo app and the dashboard's *Data & EDA* tab need. Steps that need tools you don't have are skipped with a message: dashboard screenshots (Google Chrome), slide-deck rebuild and report text (not part of this repo), PDF copies (LibreOffice). The finished report and decks are already in `deliverables/`.

You can also run a single step, for example:

```powershell
python -X utf8 -m src.vis_forecast       # leftover stock, forecasting, simulation
python -X utf8 -m src.vis_risk_model     # pre-launch risk model
python -X utf8 -m src.survey_analysis    # survey (expects survey responses in data\raw\survey\)
```

### 6. Next time

```powershell
cd "C:\Users\THE GOD\Documents\Github\From-Trend-to-Waste"
.\.venv\Scripts\Activate.ps1
python -X utf8 -m streamlit run dashboard\app.py
```

---

## macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run dashboard/app.py
python -m src.run_all               # needs visuelle2/ and data/raw/survey/ (see step 4)
```

---

## The dashboards

| App | What it shows |
|---|---|
| `dashboard/app.py` | **Early-warning dashboard**, one tab per decision maker: Overview · Data & EDA · 1 Pre-launch risk (buyers) · 2 Week-2 watchlist (planners) · 3 Customers & survey (marketing) · 4 Impact simulator (management) · 5 Financial statements (finance) |
| `dashboard/demo_app.py` | **Guided demo** for presenting: the project as a 9-step story with one interactive chart per question; turn on *Presenter notes* in the sidebar |
| `dashboard/review2_app.py` | **Review 2** (mid-project) dashboard |

All dashboards read precomputed files from `outputs/` (and `data/processed/` for the demo and the EDA tab), so they open in seconds.

## Google Colab

`deliverables/colab/From_Trend_to_Waste_Colab.ipynb` reruns every algorithm step by step with explanations (`..._with_results.ipynb` shows the outputs). Upload the private data pack to your Google Drive, open the notebook in Colab, and choose *Runtime → Run all*. To rebuild the pack from local data: `python -X utf8 -m src.colab_pack`.

## Deliverables

| File | What |
|---|---|
| `deliverables/From_Trend_to_Waste_Final_Report.pdf` | Final project report (76 pages, SRM format) |
| `deliverables/From_Trend_to_Waste_Report.pdf` | Report generated from the pipeline |
| `deliverables/From_Trend_to_Waste_Results.pptx` / `.pdf` | Results deck: 31 slides + appendix, speaker notes with likely questions |
| `deliverables/From_Trend_to_Waste_Review2.pptx` / `.pdf` | Review 2 deck |
| `deliverables/Project_Summary.pdf` | Two-page summary |
| `deliverables/Review2_Presenter_Guide.pdf` | Presenter guide for Review 2 |

---

## Troubleshooting (Windows)

| Problem | Fix |
|---|---|
| `python` is not recognized, or the Microsoft Store opens | Reinstall Python from python.org with **"Add python.exe to PATH"** ticked, or use `py` instead of `python` (e.g. `py -m venv .venv`). |
| `running scripts is disabled on this system` | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `UnicodeEncodeError` / `UnicodeDecodeError` | Add `-X utf8` after `python`, or run `setx PYTHONUTF8 1` and open a new terminal. |
| `cd` fails on `THE GOD` | Put the path in quotes: `cd "C:\Users\THE GOD\Documents\Github\From-Trend-to-Waste"` |
| Demo app says *"Project outputs are missing"* | Run the full pipeline once (steps 4–5). |
| Data & EDA tab is empty | Same: it needs `data/processed/`, which the pipeline creates. |
| Warning about scikit-learn version when loading the model | Install the pinned version: `pip install scikit-learn==1.9.1` |
| Port 8501 already in use | Stop the other Streamlit window, or add `--server.port 8502`. |
| `FileNotFoundError: ... visuelle2\sales.csv` | The raw data is missing: see step 4. |

## Data, ethics and limits

- VISUELLE 2.0 is used under its request-form terms and is **never committed or redistributed**. Customer IDs are anonymous numbers, and results are reported for groups only.
- The survey was anonymous, with informed consent, age in bands and aggregate reporting only. Raw responses stay out of the repo.
- Results show association, not causation. They come from one retailer (2017–2019). Prices in the dataset are normalised, so money is in index units, and unit cost is assumed (40–60% of full price), not observed.
- The survey is a convenience sample (122 people, mostly aged 18–24 in India) describing stated behaviour.

## Data source

VISUELLE 2.0 — Skenderi et al. (2022), *The multi-modal universe of fast-fashion: the VISUELLE 2.0 benchmark*, Humatics Lab, University of Verona. https://humaticslab.github.io/forecasting/visuelle
