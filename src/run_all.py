"""Re-run the whole pipeline from the raw files, in dependency order (VISUELLE 2.0 + our consumer survey).

    python -m src.run_all

The claims register is rebuilt from scratch on every full run, so it only ever lists numbers this run produced.
"""
import shutil
import subprocess
import time

from . import (build_report, dashboard_screenshots, data_quality, deck_series, export_pdf, literature, pilot_design,
               review2_data, review2_screens, survey_analysis, trends, vis_customers, vis_financials, vis_forecast,
               vis_lifecycle, vis_risk_model)
from .config import REPORTS, ROOT
from .utils import CLAIMS

STEPS = [
    ("Stock, week-2 forecasting, what-if simulation", vis_forecast.main),
    ("Demand lifecycles and markdown timing", vis_lifecycle.main),
    ("Pre-launch excess-stock risk model", vis_risk_model.main),
    ("Customer segments, CLTV, frequency bands", vis_customers.main),
    ("Market trends, seasonality, Google Trends value", trends.main),
    ("Pilot experiment design (power analysis)", pilot_design.main),
    ("Financial statements built from the data", vis_financials.main),
    ("Survey analysis (real responses only)", survey_analysis.run),
    ("Data overview, cleaning log, EDA tables", data_quality.main),
    ("Review 2 (mid-project) dashboard data", review2_data.main),
    ("Literature facts -> claims register", literature.main),
    ("Report, viva notes, summary, Review 2 guide", build_report.build),
    ("Deck chart data", deck_series.main),
]


def main():
    CLAIMS.unlink(missing_ok=True)
    for name, fn in STEPS:
        if fn is build_report.build and not all((REPORTS / t).exists() for t in build_report.TEMPLATES):
            print(f"\n=== {name} ===\nreport templates not found in reports/: skip (see README)")
            continue
        t = time.time()
        print(f"\n=== {name} ===")
        fn()
        print(f"--- done in {time.time() - t:.0f}s")
    print("\n=== Dashboard screenshots for the deck ===")
    dashboard_screenshots.main()
    print("\n=== Review 2 dashboard screenshots ===")
    review2_screens.main()
    if not (ROOT / "deck" / "build_deck.js").exists():
        print("\ndeck/ folder not found: skip deck build (the finished decks are in deliverables/)")
    elif shutil.which("node"):
        print("\n=== Decks (node deck/build_deck.js, deck/build_review2_deck.js) ===")
        subprocess.run(["node", "deck/build_deck.js"], check=True)
        subprocess.run(["node", "deck/build_review2_deck.js"], check=True)
    else:
        print("node not found: skip deck build (install Node.js, then `npm install` in deck/)")
    print("\n=== PDF copies in deliverables/ ===")
    export_pdf.main()


if __name__ == "__main__":
    main()
