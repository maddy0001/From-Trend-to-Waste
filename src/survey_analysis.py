"""Phase 5 — analyse the real Google Forms export of survey/questionnaire.md.

Usage:  python -m src.survey_analysis [path/to/responses.csv]
Default: the newest CSV in data/raw/survey/. Files whose name starts with FAKE_ are refused
(they exist only to test this code — see tests/).

Columns are found by the question codes at the start of each title (Q1., Q8a., Q21. ...),
which is how Google Forms names export columns (grid rows appear as "title [row]").

The live form's answer options were edited during collection (from 1 October 2026; see
survey/questionnaire.md). Both versions are mapped onto common bands below (HARMONISE). A value
that is not in the maps is never dropped silently: it is listed under "unmapped" in
survey_quality.json and a warning is printed.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import FIG, RANDOM_STATE, ROOT, SURVEY_RAW, TAB
from .utils import PALETTE, SERIES, plt, register_claim

# ----------------------------------------------------------------------------- answer maps
# Midpoints used for averages (both form versions)
WEARS = {"Fewer than 5 times": 3, "5-10 times": 7.5, "11-20 times": 15.5, "11-30 times": 20,
         "21-50 times": 35.5, "31-50 times": 40, "More than 50 times": 60}
KEEP_YEARS = {"Less than 6 months": 0.25, "6-12 months": 0.75, "1-2 years": 1.5, "2-3 years": 2.5, "More than 3 years": 4}
UNWORN = {"Less than 10%": 0.05, "10-25%": 0.175, "26-50%": 0.38, "More than 50%": 0.625}
FREQ5 = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Always": 4}

# Common bands across the original (v1) and edited (v2) form
HARMONISE = {
    "Q1": {"Under 18": "Under 18", "18-24": "18-24", "25-34": "25-34", "35-44": "35-44",
           "45-54": "45+", "55+": "45+", "45 or above": "45+"},
    "Q2": {"Student": "Student", "Working full-time": "Working full-time", "Employed full-time": "Working full-time",
           "Working part-time / freelance": "Part-time / self-employed", "Employed part-time": "Part-time / self-employed",
           "Self-employed / freelancer": "Part-time / self-employed", "Homemaker": "Not in paid work",
           "Not working / retired": "Not in paid work", "Unemployed / seeking work": "Not in paid work",
           "Retired": "Not in paid work", "Prefer not to say": "Prefer not to say"},
    "Q3": {"Metro city": "Metro city", "Other city / town": "Other city / town", "Tier-2 city": "Other city / town",
           "Town / semi-urban area": "Other city / town", "Village / rural area": "Rural", "Rural area": "Rural"},
    "Q4": {"Less than Rs 1,000": "Under Rs 1,000", "Rs 1,000 - 2,999": "Rs 1,000-2,999", "Rs 3,000 - 4,999": "Rs 3,000-4,999",
           "Rs 5,000 or more": "Rs 5,000+", "Rs 5,000 - 9,999": "Rs 5,000+", "Rs 10,000 or more": "Rs 5,000+"},
    # same four bands as the Nuna Lie customer analysis (src/config.py FREQ_LABELS)
    "Q5": {"Weekly or more": "Monthly or more", "2-3 times a month": "Monthly or more", "About once a month": "Monthly or more",
           "Monthly or more": "Monthly or more", "5-11 times a year": "5-11 times a year", "2-4 times a year": "2-4 times a year",
           "Once a year or less": "Once a year or less", "Less than once a year": "Once a year or less"},
    "Q6": {"Mostly online": "Mostly online", "Social media / live-shopping links": "Mostly online",
           "Mostly in stores": "Mostly in stores", "About equally online and in stores": "Both equally"},
    "Q10": {"Fewer than 5 times": "Fewer than 5", "5-10 times": "5-10", "11-20 times": "11-50", "11-30 times": "11-50",
            "21-50 times": "11-50", "31-50 times": "11-50", "More than 50 times": "More than 50"},
}
ORDER = {
    "Q1": ["Under 18", "18-24", "25-34", "35-44", "45+"],
    "Q5": ["Once a year or less", "2-4 times a year", "5-11 times a year", "Monthly or more"],
    "Q10": ["Fewer than 5", "5-10", "11-50", "More than 50"],
    "Q11": list(KEEP_YEARS), "Q12": list(UNWORN),
}
FORM_EDIT = pd.Timestamp("2026-10-01")  # first response with edited options: 1 Oct 2026, 00:05 IST
DISPOSAL = {"Q13": "Sell / resell", "Q14": "Donate", "Q15": "Hand down", "Q16": "Recycle / take-back",
            "Q17": "Throw away", "Q18": "Keep unused"}
MOTIVES = {"Q8a": "Trend", "Q8b": "Discount", "Q8c": "Novelty", "Q8d": "Quality", "Q8e": "Environment"}
ATTITUDES = {"Q19a": "Knows clothes use water/energy", "Q19b": "Fast fashion harms environment",
             "Q19c": "Would buy fewer if they lasted longer", "Q19d": "Would pay more for longer-lasting",
             "Q19e": "Brands should produce fewer items"}
CONJOINT = [f"Q{21 + i}" for i in range(8)]
ATTRS = ["price_high", "trendy", "long_life", "certified"]
ATTR_NAMES = {"price_high": "Price Rs 1,499 (vs Rs 799)", "trendy": "Trendy (vs timeless)",
              "long_life": "Lasts 3+ years (vs 1 season)", "certified": "Certified sustainable (vs no label)"}
WRAP_YEARS = {"WRAP 2012 (UK)": 2.2, "WRAP later update (UK)": 3.3}
PRICE_GAP_RS = 700  # Rs 1,499 - Rs 799
BOOT = 2000


def find_col(df, code):
    pat = re.compile(rf"(^|\[)\s*{re.escape(code)}\.")
    hits = [c for c in df.columns if pat.search(c)]
    if not hits:
        raise KeyError(f"No column for {code}. Question titles must start with '{code}.'")
    return hits[0]


def first_int(x):
    m = re.match(r"\s*(\d+)", str(x))
    return float(m.group(1)) if m else np.nan


def parse_time(s):
    """Google Forms: '2026/09/27 9:14:48 PM GMT+5:30' -> local timestamp (time-zone text dropped)."""
    return pd.to_datetime(s.astype(str).str.replace(r"\s*GMT.*$", "", regex=True), format="mixed", errors="coerce")


def load(path: Path, allow_fake=False):
    """Returns (cleaned respondent table, quality dict)."""
    if path.name.startswith("FAKE_") and not allow_fake:
        raise ValueError("Refusing to analyse a FAKE_ test fixture as real survey data.")
    raw = pd.read_csv(path, keep_default_na=False, na_values=[""])  # keep a literal 'None' answer (Q9)
    q = {"rows_in_file": int(len(raw))}
    consent = raw[find_col(raw, "Q0")].astype(str).str.startswith("Yes")
    q["consented"] = int(consent.sum())
    q["did_not_consent"] = int((~consent).sum())
    raw = raw[consent].copy()
    body_cols = [c for c in raw.columns if c != "Timestamp" and not c.startswith("Q20.")]
    q["duplicate_answer_rows"] = int(raw[body_cols].duplicated(keep=False).sum())

    d = pd.DataFrame(index=raw.index)
    if "Timestamp" in raw:
        t = parse_time(raw["Timestamp"])
        d["submitted"] = t
        d["form_version"] = np.where(t < FORM_EDIT, "v1 original", "v2 edited")
        q["first_response"] = str(t.min())
        q["last_response"] = str(t.max())
    else:
        d["form_version"] = "unknown"
    q["form_versions"] = d.form_version.value_counts().to_dict()

    unmapped = {}
    for code in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q9", "Q10", "Q11", "Q12"):
        d[code + "_raw"] = raw[find_col(raw, code)]
    for code, mp in HARMONISE.items():
        d[code] = d[code + "_raw"].map(mp)
        bad = sorted(set(d.loc[d[code].isna() & d[code + "_raw"].notna(), code + "_raw"]))
        if bad:
            unmapped[code] = bad
    for code, mp in (("Q11", KEEP_YEARS), ("Q12", UNWORN)):
        d[code] = d[code + "_raw"]
        bad = sorted(set(d.loc[~d[code].isin(list(mp)) & d[code].notna(), code]))
        if bad:
            unmapped[code] = bad
    d["Q9"] = d["Q9_raw"]
    q["unmapped"] = unmapped
    if unmapped:
        print("WARNING: answer values not in the harmonisation maps:", unmapped)

    q7 = raw[find_col(raw, "Q7")].fillna("").astype(str)
    d["buys_fast_fashion"] = q7.str.contains("Fast-fashion brands|mix of fast-fashion", case=False, regex=True)
    d["buys_premium"] = q7.str.contains("premium", case=False)
    d["buys_local"] = q7.str.contains("Local shops", case=False)
    d["buys_second_hand"] = q7.str.contains("second-hand|thrift", case=False, regex=True)

    for code in list(MOTIVES) + list(ATTITUDES):
        d[code] = raw[find_col(raw, code)].map(first_int)
    for code in DISPOSAL:
        d[code] = raw[find_col(raw, code)].map(FREQ5)
    for code in CONJOINT:
        d[code] = raw[find_col(raw, code)].map(first_int)
    d["wears"] = d["Q10_raw"].map(WEARS)
    d["keep_years"] = d["Q11"].map(KEEP_YEARS)
    d["unworn_share"] = d["Q12"].map(UNWORN)
    q["q9_unanswered"] = int(d["Q9"].isna().sum())
    q20 = [c for c in raw.columns if c.startswith("Q20.")]  # optional comment box (never displayed)
    q["free_text_comments_not_shown"] = int(raw[q20[0]].notna().sum()) if q20 else 0
    cj = d[CONJOINT]
    q["conjoint_incomplete"] = int(cj.isna().any(axis=1).sum())
    q["conjoint_all_same_rating_excluded"] = int(((cj.nunique(axis=1) == 1) & cj.notna().all(axis=1)).sum())
    n = len(d)
    q["margin_of_error_95_for_a_share"] = float(1.96 * np.sqrt(0.25 / n)) if n else None
    return d.reset_index(drop=True), q


# ----------------------------------------------------------------------------- conjoint
def conjoint(d, design, seed=RANDOM_STATE):
    """Rating-based conjoint on an orthogonal 2^(4-1) design (8 profiles per respondent).
    Respondent fixed effects via within-respondent demeaning; cluster-robust CIs by respondent.
    Straight-liners (all 8 ratings equal) carry no preference information and are excluded."""
    cj = d[CONJOINT]
    keep = cj.notna().all(axis=1) & (cj.nunique(axis=1) > 1)
    Y = cj[keep].to_numpy(float)                      # respondents x 8, columns = P1..P8
    Yd = Y - Y.mean(1, keepdims=True)
    long = pd.DataFrame(Yd, columns=design.profile).assign(rid=np.arange(len(Y))).melt(
        id_vars="rid", var_name="profile", value_name="rating_dm").merge(design, on="profile")
    m = smf.ols("rating_dm ~ " + " + ".join(ATTRS), data=long).fit(cov_type="cluster", cov_kwds={"groups": long.rid})
    pooled = pd.DataFrame({"attribute": ATTRS, "label": [ATTR_NAMES[a] for a in ATTRS],
                           "part_worth": m.params[ATTRS].values, "ci_low": m.conf_int().loc[ATTRS, 0].values,
                           "ci_high": m.conf_int().loc[ATTRS, 1].values, "p_value": m.pvalues[ATTRS].values})

    X = np.column_stack([np.ones(8), design[ATTRS].to_numpy(float)])
    P = np.linalg.pinv(X)                              # 5 x 8
    ind = Yd @ P.T                                     # individual part-worths (same as per-person OLS)
    imp = np.abs(ind[:, 1:]) / np.abs(ind[:, 1:]).sum(1, keepdims=True)
    importance = pd.Series(imp.mean(0), index=["Price", "Style (trendy)", "Lifespan", "Sustainability label"])

    def wtp_from(beta):  # beta columns: intercept, price, trendy, long_life, certified
        per_rs = -beta[..., 1] / PRICE_GAP_RS
        with np.errstate(divide="ignore", invalid="ignore"):
            return {a: np.where(per_rs > 0, beta[..., j] / per_rs, np.nan) for j, a in ((2, "trendy"), (3, "long_life"), (4, "certified"))}

    beta = Yd.mean(0) @ P.T
    wtp = {a: float(v) for a, v in wtp_from(beta).items()}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(Yd), (BOOT, len(Yd)))
    bb = Yd[idx].mean(1) @ P.T
    wb = wtp_from(bb)
    wtp_ci = {a: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for a, v in wb.items()}
    return pooled, importance, wtp, wtp_ci, int(len(Y))


# ----------------------------------------------------------------------------- segments
def name_segments(z):
    names = {}
    for sid, row in z.iterrows():
        tops = [m for m in row.sort_values(ascending=False).index if row[m] > 0.25][:2]
        names[sid] = " & ".join(tops) + "-driven" if tops else "Low-involvement"
    seen = {}
    for sid in names:
        seen[names[sid]] = seen.get(names[sid], 0) + 1
        if seen[names[sid]] > 1:
            names[sid] = f"{names[sid]} ({seen[names[sid]]})"
    return names


# ----------------------------------------------------------------------------- link to VISUELLE customers
def bridge_with_vis(d):
    """Group-level comparison (never person-level) with Nuna Lie loyalty customers, by the same buying-frequency bands."""
    out = {}
    pf = TAB / "vis_customers_by_frequency.csv"
    if pf.exists():
        vf_ = pd.read_csv(pf).set_index("freq_band")[["customers", "short_lived_share", "markdown_share"]]
        vf_.columns = ["vis_customers", "vis_short_lived_share", "vis_markdown_share"]
        sv = d.groupby("Q5").agg(respondents=("Q5", "size"), trend_influence=("Q8a", "mean"),
                                 trend_4or5=("Q8a", lambda s: (s >= 4).mean()), discount_influence=("Q8b", "mean"),
                                 buys_fast_fashion=("buys_fast_fashion", "mean"))
        out["frequency"] = (sv.join(vf_, how="left").reindex([b for b in ORDER["Q5"] if b in sv.index])
                              .reset_index().rename(columns={"Q5": "freq_band"}))
    return out


def rank_corr(d, code, col):
    order = {b: i for i, b in enumerate(ORDER[code])}
    x = d[code].map(order)
    ok = x.notna() & d[col].notna()
    r, p = spearmanr(x[ok], d.loc[ok, col])
    return float(r), float(p)


# ----------------------------------------------------------------------------- figures
def figures(d, desc, pooled, importance, wtp, wtp_ci, fig_dir, n, n_conj):
    # 1. Disposal: 100% stacked bars Never..Always
    levels = ["Never", "Rarely", "Sometimes", "Often", "Always"]
    cols = ["#E6E2DC", "#D9C7A7", "#C9BFAF", PALETTE["teal"], "#1F4D50"]
    shares = pd.DataFrame({DISPOSAL[q]: d[q].value_counts(normalize=True).reindex(range(5), fill_value=0).values
                           for q in DISPOSAL}, index=levels).T
    shares = shares.loc[shares[["Often", "Always"]].sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    left = np.zeros(len(shares))
    for lv, c in zip(levels, cols):
        ax.barh(shares.index, shares[lv] * 100, left=left, color=c, label=lv)
        left += shares[lv].values * 100
    for i, v in enumerate(shares[["Often", "Always"]].sum(axis=1)):
        ax.text(101, i, f"{v:.0%} often/always", va="center", fontsize=9)
    ax.set_xlim(0, 130)
    ax.set_xticks(range(0, 101, 20))
    ax.spines["bottom"].set_bounds(0, 100)
    ax.set_xlabel("% of respondents")
    ax.set_title(f"What happens to clothes people stop using (n={n})")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.4, -0.18))
    fig.savefig(fig_dir / "survey_disposal.png")
    plt.close(fig)

    # 2. Conjoint: importance + part-worths with 95% CI
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4), gridspec_kw={"width_ratios": [1, 1.1]})
    fig.subplots_adjust(wspace=0.95)
    imp = importance.sort_values()
    axes[0].barh(imp.index, imp.values * 100, color=PALETTE["accent"])
    for i, v in enumerate(imp.values):
        axes[0].text(v * 100 + 0.8, i, f"{v:.0%}", va="center", fontsize=9)
    axes[0].set_xlabel("Average relative importance (%)")
    axes[0].set_title("What drives the choice?")
    pw = pooled.iloc[::-1]
    ycoords = np.arange(len(pw))
    colrs = [PALETTE["teal"] if v > 0 else PALETTE["accent"] for v in pw.part_worth]
    axes[1].hlines(ycoords, pw.ci_low, pw.ci_high, color=colrs, lw=2.5)
    axes[1].scatter(pw.part_worth, ycoords, color=colrs, zorder=3)
    axes[1].axvline(0, color=PALETTE["muted"], lw=1)
    axes[1].set_yticks(ycoords)
    axes[1].set_yticklabels(pw.label, fontsize=9)
    axes[1].set_xlabel("Change in rating (1-10 scale), 95% CI")
    axes[1].set_title("Effect of each feature on the rating")
    fig.suptitle(f"Conjoint analysis (n={n_conj})", fontweight="bold", y=1.03)
    fig.savefig(fig_dir / "survey_conjoint_importance.png")
    plt.close(fig)

    # 3. Garment life: how long kept (with WRAP) and how many wears
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3))
    k = d.Q11.value_counts(normalize=True).reindex(ORDER["Q11"], fill_value=0)
    axes[0].bar(range(len(k)), k.values * 100, color=PALETTE["teal"])
    axes[0].set_xticks(range(len(k)))
    axes[0].set_xticklabels(["<6 m", "6-12 m", "1-2 y", "2-3 y", ">3 y"])
    axes[0].set_ylabel("% of respondents")
    axes[0].set_title(f"How long a garment is kept (mean ≈ {desc['mean_keep_years']:.1f} years)")
    axes[0].text(0.02, 0.95, "WRAP UK average:\n2.2-3.3 years", transform=axes[0].transAxes, ha="left", va="top", fontsize=8.5, color=PALETTE["muted"])
    w = d.Q10.value_counts(normalize=True).reindex(ORDER["Q10"], fill_value=0)
    axes[1].bar(range(len(w)), w.values * 100, color=[PALETTE["accent"], PALETTE["accent"], PALETTE["sand"], PALETTE["sand"]])
    axes[1].set_xticks(range(len(w)))
    axes[1].set_xticklabels(["<5 wears", "5-10", "11-50", ">50"])
    axes[1].set_title("Wears before a fast-fashion item is dropped")
    axes[1].set_ylabel("% of respondents")
    fig.savefig(fig_dir / "survey_lifetime.png")
    plt.close(fig)

    # 4. Attitudes: % agree
    a = pd.Series(desc["attitude_share_agree_4or5"]).sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.barh(a.index, a.values * 100, color=PALETTE["teal"])
    for i, v in enumerate(a.values):
        ax.text(v * 100 + 1, i, f"{v:.0%}", va="center", fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_xlabel("% agree or strongly agree (4-5)")
    ax.set_title(f"Attitudes (n={n})")
    fig.savefig(fig_dir / "survey_attitudes.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def run(csv_path=None, out_dir: Path = TAB, fig_dir: Path = FIG, allow_fake=False):
    if csv_path is None:
        files = sorted(SURVEY_RAW.glob("*.csv"), key=lambda p: p.stat().st_mtime)
        files = [f for f in files if not f.name.startswith("FAKE_")]
        if not files:
            print("No survey export found in data/raw/survey/ — collect real responses first.")
            return None
        csv_path = files[-1]
    csv_path = Path(csv_path)
    d, quality = load(csv_path, allow_fake=allow_fake)
    design = pd.read_csv(ROOT / "survey" / "conjoint_design.csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    n = len(d)

    def share(s, values):
        return float(s.isin(values).sum() / s.notna().sum()) if s.notna().any() else float("nan")

    pooled, importance, wtp, wtp_ci, n_conj = conjoint(d, design)
    disp_often = {DISPOSAL[q]: float((d[q] >= 3).mean()) for q in DISPOSAL}
    motives = {MOTIVES[q]: float(d[q].mean()) for q in MOTIVES}
    desc = {
        "source_file": csv_path.name, "respondents": n,
        "distributions": {c: d[c].value_counts(normalize=True).round(4).to_dict()
                          for c in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q9", "Q10", "Q11", "Q12")},
        "share_age_18_24": share(d.Q1, ["18-24"]), "share_students": share(d.Q2, ["Student"]),
        "share_buys_fast_fashion": float(d.buys_fast_fashion.mean()),
        "share_buys_second_hand": float(d.buys_second_hand.mean()),
        "motive_means_1to5": motives,
        "motive_share_4or5": {MOTIVES[q]: float((d[q] >= 4).mean()) for q in MOTIVES},
        "top_motive": max(motives, key=motives.get),
        "attitude_means_1to5": {ATTITUDES[q]: float(d[q].mean()) for q in ATTITUDES},
        "attitude_share_agree_4or5": {ATTITUDES[q]: float((d[q] >= 4).mean()) for q in ATTITUDES},
        "mean_keep_years": float(d.keep_years.mean()), "median_keep_years": float(d.keep_years.median()),
        "share_keep_under_1_year": share(d.Q11, ["Less than 6 months", "6-12 months"]),
        "share_keep_over_3_years": share(d.Q11, ["More than 3 years"]),
        "wrap_benchmarks_years": WRAP_YEARS,
        "mean_wears_fast_fashion": float(d.wears.mean()),
        "share_wears_under_10": share(d.Q10, ["Fewer than 5", "5-10"]),
        "mean_unworn_share": float(d.unworn_share.mean()),
        "share_unworn_over_25pct": share(d.Q12, ["26-50%", "More than 50%"]),
        "share_unplanned_3plus_of_answered": share(d.Q9, ["3-5", "6 or more"]),
        "disposal_share_often_or_always": disp_often,
        "disposal_share_never": {DISPOSAL[q]: float((d[q] == 0).mean()) for q in DISPOSAL},
        "top_disposal": max(disp_often, key=disp_often.get),
        "conjoint_respondents": n_conj, "conjoint_importance": importance.to_dict(),
        "conjoint_part_worths": pooled.set_index("attribute")[["part_worth", "ci_low", "ci_high", "p_value"]].to_dict("index"),
        "wtp_rs": wtp, "wtp_rs_ci95": wtp_ci, "bootstrap_reps": BOOT,
        "by_form_version": d.groupby("form_version").agg(respondents=("Q1", "size"), keep_years=("keep_years", "mean"),
                                                         trend=("Q8a", "mean"), quality=("Q8d", "mean")).round(3).to_dict("index"),
    }
    # Model-free check: same profile, but 3+ years for Rs 1,499 vs one season for Rs 799
    cj = d[CONJOINT].copy()
    cj.columns = list(design.profile)
    cj = cj[cj.notna().all(axis=1) & (cj.nunique(axis=1) > 1)]
    desc["direct_long_life_vs_cheap"] = {}
    for hi, lo, what in (("P6", "P1", "timeless, no label"), ("P8", "P3", "trendy, certified")):
        diff = cj[hi] - cj[lo]
        desc["direct_long_life_vs_cheap"][f"{hi}_vs_{lo}"] = {
            "profiles": what, "mean_rating_difference": float(diff.mean()), "share_rated_long_life_higher": float((diff > 0).mean()),
            "share_equal": float((diff == 0).mean()), "share_rated_cheap_higher": float((diff < 0).mean())}
    desc["mean_rating_by_profile"] = {k_: float(v) for k_, v in cj.mean().items()}
    desc["rank_corr_age_vs_trend"] = dict(zip(("rho", "p_value"), rank_corr(d, "Q1", "Q8a")))
    desc["rank_corr_frequency_vs_trend"] = dict(zip(("rho", "p_value"), rank_corr(d, "Q5", "Q8a")))
    desc["rank_corr_frequency_vs_discount"] = dict(zip(("rho", "p_value"), rank_corr(d, "Q5", "Q8b")))
    desc["rank_corr_age_vs_discount"] = dict(zip(("rho", "p_value"), rank_corr(d, "Q1", "Q8b")))
    v2 = d[d.form_version == "v2 edited"]
    if len(v2) >= 30:  # robustness: same tests inside the larger, edited-form group only
        desc["robustness_v2_only"] = {"respondents": int(len(v2)),
                                      "rank_corr_frequency_vs_trend": dict(zip(("rho", "p_value"), rank_corr(v2, "Q5", "Q8a"))),
                                      "rank_corr_age_vs_trend": dict(zip(("rho", "p_value"), rank_corr(v2, "Q1", "Q8a")))}
    (out_dir / "survey_quality.json").write_text(json.dumps(quality, indent=2, default=str))
    (out_dir / "survey_descriptives.json").write_text(json.dumps(desc, indent=2, default=str))
    pooled.to_csv(out_dir / "survey_conjoint_partworths.csv", index=False)

    # Respondent segments on the five purchase motives
    M = d[list(MOTIVES)].fillna(d[list(MOTIVES)].mean())
    X = StandardScaler().fit_transform(M)
    k = 3 if n >= 30 else 2
    lab = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(X)
    z = pd.DataFrame(X, columns=list(MOTIVES.values())).assign(segment=lab).groupby("segment").mean()
    names = name_segments(z)
    seg = d.assign(segment=lab).groupby("segment").agg(
        respondents=("Q1", "size"), **{MOTIVES[c]: (c, "mean") for c in MOTIVES},
        keep_years=("keep_years", "mean"), wears=("wears", "mean"), unworn_share=("unworn_share", "mean"),
        throw_away_often=("Q17", lambda s: (s >= 3).mean()), buys_fast_fashion=("buys_fast_fashion", "mean"),
        age_18_24=("Q1", lambda s: (s == "18-24").mean())).reset_index()
    seg.insert(1, "name", seg.segment.map(names))
    seg["share"] = seg.respondents / n
    seg.sort_values("respondents", ascending=False).to_csv(out_dir / "survey_segments.csv", index=False)

    br = bridge_with_vis(d)
    for key, tab in br.items():
        tab.to_csv(out_dir / f"survey_vis_bridge_by_{key}.csv", index=False)

    figures(d, desc, pooled, importance, wtp, wtp_ci, fig_dir, n, n_conj)

    if out_dir == TAB and not allow_fake:  # real data only: record headline numbers
        src = "outputs/tables/survey_descriptives.json"
        rc = register_claim
        rc("sv_n", "Survey: respondents who consented", n, "outputs/tables/survey_quality.json")
        rc("sv_versions", "Survey: responses by form version", json.dumps(quality["form_versions"]), "outputs/tables/survey_quality.json")
        rc("sv_moe", "Survey: 95% margin of error for a share", quality["margin_of_error_95_for_a_share"], "outputs/tables/survey_quality.json")
        rc("sv_age_18_24", "Survey: share aged 18-24", desc["share_age_18_24"], src)
        rc("sv_keep_mean", "Survey: mean years a garment is kept (band midpoints)", desc["mean_keep_years"], src)
        rc("sv_keep_under_1y", "Survey: share keeping a garment under 1 year", desc["share_keep_under_1_year"], src)
        rc("sv_wears_under_10", "Survey: share wearing a fast-fashion item 10 times or fewer", desc["share_wears_under_10"], src)
        rc("sv_unworn_mean", "Survey: mean share of wardrobe unworn in 12 months", desc["mean_unworn_share"], src)
        rc("sv_unworn_over_25", "Survey: share with more than 25% of wardrobe unworn", desc["share_unworn_over_25pct"], src)
        for name, v in disp_often.items():
            rc("sv_disp_" + re.sub(r"\W+", "_", name.lower()).strip("_"), f"Survey: share who {name.lower()} often/always", v, src)
        for q_, name in ATTITUDES.items():
            rc("sv_att_" + q_.lower(), f"Survey: agree (4-5) '{name}'", desc["attitude_share_agree_4or5"][name], src)
        for name, v in importance.items():
            rc("sv_imp_" + re.sub(r"\W+", "_", name.lower()).strip("_"), f"Survey conjoint: importance of {name}", float(v), src)
        for a in ("long_life", "certified", "trendy"):
            rc("sv_wtp_" + a, f"Survey conjoint: willingness to pay (Rs) for {ATTR_NAMES[a]}",
               f"{wtp[a]:.0f} (95% CI {wtp_ci[a][0]:.0f} to {wtp_ci[a][1]:.0f})", src)
        rc("sv_rho_age_trend", "Survey: Spearman, age band vs trend influence", desc["rank_corr_age_vs_trend"]["rho"], src)
        rc("sv_rho_freq_discount", "Survey: Spearman, buying frequency vs discount influence", desc["rank_corr_frequency_vs_discount"]["rho"], src)
        rc("sv_rho_freq_trend", "Survey: Spearman, buying frequency vs trend influence", desc["rank_corr_frequency_vs_trend"]["rho"], src)
        dl = desc["direct_long_life_vs_cheap"]["P6_vs_P1"]
        rc("sv_direct_life", "Survey: mean rating gain, long-lasting Rs 1,499 vs one-season Rs 799 (timeless, no label)", dl["mean_rating_difference"], src)
        rc("sv_direct_life_share", "Survey: share rating the long-lasting Rs 1,499 option higher (timeless, no label)", dl["share_rated_long_life_higher"], src)
        rc("sv_disp_top", "Survey: most common disposal route (often/always)", desc["top_disposal"], src)
        rc("sv_top_motive", "Survey: strongest purchase motive (mean 1-5)", desc["top_motive"], src)
        rc("sv_conj_n", "Survey: respondents in the conjoint (straight-liners excluded)", n_conj, src)
        rc("sv_students", "Survey: share students", desc["share_students"], src)
        rc("sv_keep_over_3y", "Survey: share keeping a garment over 3 years", desc["share_keep_over_3_years"], src)
        rc("sv_unplanned_3plus", "Survey: share (of Q9 answers) buying 3+ unplanned items in 3 months", desc["share_unplanned_3plus_of_answered"], src)
        for name, v in desc["disposal_share_never"].items():
            rc("sv_never_" + re.sub(r"\W+", "_", name.lower()).strip("_"), f"Survey: share who never {name.lower()}", v, src)
        for name, v in desc["motive_means_1to5"].items():
            rc("sv_motive_" + name.lower(), f"Survey: mean influence of {name.lower()} on buying (1-5)", v, src)
        for a in ATTRS:
            pw_ = desc["conjoint_part_worths"][a]
            rc("sv_pw_" + a, f"Survey conjoint: effect of {ATTR_NAMES[a]} on 1-10 rating",
               f"{pw_['part_worth']:.2f} (95% CI {pw_['ci_low']:.2f} to {pw_['ci_high']:.2f})", src)
        rc("sv_p_age_trend", "Survey: p-value, age band vs trend influence", desc["rank_corr_age_vs_trend"]["p_value"], src)
        rc("sv_p_freq_trend", "Survey: p-value, buying frequency vs trend influence", desc["rank_corr_frequency_vs_trend"]["p_value"], src)
        rb = desc["robustness_v2_only"]["rank_corr_frequency_vs_trend"]
        rc("sv_rho_freq_trend_v2", "Survey: Spearman frequency vs trend, edited-form responses only", f"{rb['rho']:.3f} (p {rb['p_value']:.3f})", src)
        for ver, r_ in desc["by_form_version"].items():
            rc("sv_trend_" + ver.split()[0], f"Survey: mean trend influence, form {ver}", r_["trend"], src)
        rc("sv_direct_life_lower", "Survey: share rating the one-season Rs 799 option higher (timeless, no label)", dl["share_rated_cheap_higher"], src)
        for r_ in seg.itertuples():
            rc("sv_seg_" + re.sub(r"\W+", "_", r_.name.lower()).strip("_"), f"Survey segment '{r_.name}': people / keep years / often throw away / buys fast fashion",
               f"{r_.respondents} / {r_.keep_years:.2f} / {r_.throw_away_often:.3f} / {r_.buys_fast_fashion:.3f}", "outputs/tables/survey_segments.csv")
        for r_ in br.get("frequency", pd.DataFrame()).itertuples():
            rc("sv_bridge_" + re.sub(r"\W+", "_", r_.freq_band.lower()).strip("_"),
               f"Survey vs Nuna Lie customers, '{r_.freq_band}': survey discount influence / Nuna Lie markdown share",
               f"{r_.discount_influence:.2f} / {r_.vis_markdown_share:.4f}", "outputs/tables/survey_vis_bridge_by_frequency.csv")
    print(json.dumps({"quality": quality, **{k_: v for k_, v in desc.items() if k_ != "distributions"}}, indent=1, default=str))
    return desc


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
