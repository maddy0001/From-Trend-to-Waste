"""Paths and analysis constants shared by every module (VISUELLE 2.0 + our consumer survey)."""
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VIS_RAW = ROOT / "visuelle2"
SURVEY_RAW = ROOT / "data" / "raw" / "survey"

PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
MODELS = ROOT / "outputs" / "models"
REPORTS = ROOT / "reports"

for p in (PROCESSED, FIG, TAB, MODELS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

# --- VISUELLE 2.0 (Nuna Lie) -------------------------------------------------
K_WEEKS = 12                       # each product is observed for its first 12 weeks in every shop
TRAIN_SEASONS = ["SS17", "AW17", "SS18", "AW18"]
TEST_SEASONS = ["SS19", "AW19"]    # never seen when fitting models
SEASON_ORDER = ["SS17", "AW17", "SS18", "AW18", "SS19", "AW19"]
MIN_UNITS_LIFECYCLE = 30           # minimum 12-week units for a stable sales-curve shape

# Markdown = discount of at least 15% off the full price (sensitivity: 10% and 20%)
MARKDOWN_THRESHOLD = 0.15
MARKDOWN_SENSITIVITY = (0.10, 0.15, 0.20)
# Excess-stock outcome: a product that sells less than 60% of its delivered stock in 12 weeks
HIGH_LEFTOVER_SELL_THROUGH = 0.60

# Customer purchases (customer_data.csv): window with full product coverage, before COVID shop closures.
# The dataset holds only SS17-AW19 products, so purchases after February 2020 are not representative.
CUSTOMER_START = date(2016, 12, 1)
CUSTOMER_END = date(2020, 3, 1)    # exclusive
CUSTOMER_MAX_SHOPS = 10            # IDs buying in more shops are shop or generic cards, not people

# Bands shared by the customer analysis and the survey (Q5 buying frequency, Q1 age)
FREQ_BANDS = [0, 1.5, 4.5, 12, 1e9]   # purchase days per year
FREQ_LABELS = ["Once a year or less", "2-4 times a year", "5-11 times a year", "Monthly or more"]
AGE_LABELS = ["Under 18", "18-24", "25-34", "35-44", "45-54", "55+"]

RANDOM_STATE = 42
