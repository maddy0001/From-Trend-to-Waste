"""Small helpers: figure style, saving outputs, and the claims register."""
import csv
import json
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import FIG, REPORTS, TAB

PALETTE = {
    "ink": "#2B2B2B",
    "muted": "#8A8580",
    "accent": "#C4663A",   # terracotta — the deck's accent colour
    "teal": "#2F6F73",
    "sand": "#D9C7A7",
    "grid": "#E6E2DC",
}
SERIES = ["#C4663A", "#2F6F73", "#8C6BB1", "#D9A441", "#5B8E4F", "#8A8580"]

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": PALETTE["muted"],
    "axes.labelcolor": PALETTE["ink"],
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.8,
    "xtick.color": PALETTE["ink"],
    "ytick.color": PALETTE["ink"],
    "axes.prop_cycle": plt.cycler(color=SERIES),
})


def savefig(fig, name: str):
    path = FIG / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_table(df, name: str):
    """Save a polars or pandas frame as CSV in outputs/tables."""
    path = TAB / f"{name}.csv"
    if hasattr(df, "write_csv"):
        df.write_csv(path)
    else:
        df.to_csv(path, index=False)
    return path


def save_json(obj, name: str):
    path = TAB / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


CLAIMS = REPORTS / "claims_register.csv"
_FIELDS = ["claim_id", "claim", "value", "source_or_script", "date_verified"]


def register_claim(claim_id: str, claim: str, value, source: str):
    """Upsert one row in reports/claims_register.csv so every reported number is traceable."""
    rows = {}
    if CLAIMS.exists():
        with CLAIMS.open() as f:
            rows = {r["claim_id"]: r for r in csv.DictReader(f)}
    rows[claim_id] = {
        "claim_id": claim_id,
        "claim": claim,
        "value": value if isinstance(value, str) else f"{value:.6g}" if isinstance(value, float) else str(value),
        "source_or_script": source,
        "date_verified": date.today().isoformat(),
    }
    with CLAIMS.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
