"""Data overview, cleaning log and EDA tables (rubric: data preprocessing + EDA dashboard) — VISUELLE 2.0 + survey.

Writes
  outputs/tables/data_overview.csv      one row per file: rows, columns, period, unit, use
  outputs/tables/data_quality_log.csv   per dataset: missing values, duplicates, outliers, inconsistencies,
                                        scope filters -> what we found (counted here) and how the pipeline handles it
  data/processed/eda_*.parquet          compact per-product / per-customer / per-respondent tables for the dashboard's
                                        Data & EDA tab (local only: derived from licensed data, git-ignored)
Every count is computed from the data in this script; "how handled" describes the code in the named module.
Run after vis_lifecycle, vis_customers and survey_analysis (it reads their outputs).
"""
import json

import numpy as np
import pandas as pd

from .config import CUSTOMER_END, CUSTOMER_MAX_SHOPS, CUSTOMER_START, K_WEEKS, MIN_UNITS_LIFECYCLE, PROCESSED, TAB, VIS_RAW
from .utils import register_claim, save_table

K = K_WEEKS
V = "VISUELLE 2.0"


def vis_checks():
    s = pd.read_csv(VIS_RAW / "sales.csv").drop(columns="Unnamed: 0", errors="ignore")
    p = pd.read_csv(VIS_RAW / "price_discount_series.csv")
    r = pd.read_csv(VIS_RAW / "restocks.csv")
    c = pd.read_csv(VIS_RAW / "customer_data.csv", parse_dates=["data"])
    g = pd.read_csv(VIS_RAW / "vis2_gtrends_data.csv")
    wk = [str(k) for k in range(K)]
    neg = int((s[wk] < 0).sum().sum())
    disc = p[wk]
    rel = pd.to_datetime(s.set_index(["external_code", "retail"]).release_date)
    rd = r.assign(rdate=[pd.Timestamp.fromisocalendar(y, w, 1) for y, w in zip(r.year, r.week)])
    rd = rd.join(rel.rename("release"), on=["external_code", "retail"])
    k = (rd.rdate - rd.release).dt.days // 7
    win = (c.data >= pd.Timestamp(CUSTOMER_START)) & (c.data < pd.Timestamp(CUSTOMER_END))
    cw = c[win]
    shops = cw.groupby("customer").retail.nunique()
    cards = shops[shops > CUSTOMER_MAX_SHOPS].index
    known_prod = c.external_code.isin(s.external_code)
    lc = json.loads((TAB / "vis_lifecycle_summary.json").read_text())
    gcols = [x for x in g.columns if x != "date"]
    rows = [
        (V, "Missing values", f"sales: {int(s[wk].isna().sum().sum())} empty cells; discounts: {int(disc.isna().sum().sum())} empty cell; "
                              f"customer purchases: {int(c.isna().sum().sum())} empty cells",
         "The one empty discount is read as no discount (vis_forecast.py)"),
        (V, "Missing values", f"Google Trends: {int(g[gcols].isna().sum().sum())} empty cells in {len(gcols)} search terms",
         "A product gets no Google Trends feature if its term has under 8 weeks of data before release"),
        (V, "Duplicates", f"duplicate product-shop rows: sales {int(s.duplicated(['external_code', 'retail']).sum())}, prices "
                          f"{int(p.duplicated(['external_code', 'retail']).sum())}, restocks {int(r.duplicated().sum())}", "None found"),
        (V, "Duplicates", f"{int(c.duplicated().sum()):,} identical customer purchase line{'' if c.duplicated().sum() == 1 else 's'} "
                          "(same customer, shop, product, second)",
         "Kept: each line is a separate scan of the same item at the till"),
        (V, "Outliers", f"{neg:,} negative weekly sales cells ({neg / s[wk].size:.1%}): returns",
         "Clipped to 0, since we model demand, not returns (vis_forecast.py)"),
        (V, "Outliers", f"discounts outside 0-100%: {int(((disc < 0) | (disc > 1)).sum().sum())}", "None found"),
        (V, "Outliers", f"{len(cards)} customer IDs buy in more than {CUSTOMER_MAX_SHOPS} shops "
                        f"({int(cw[cw.customer.isin(cards)].qty.sum()):,} units, up to {int(shops.max())} shops)",
         "Excluded: shop or generic cards, not real customers (vis_customers.py)"),
        (V, "Inconsistencies", f"{(~known_prod).mean():.1%} of purchase lines are for products not in the sales file; "
                               f"customers bought in {c.retail.nunique()} shops vs {s.retail.nunique()} in the sales file",
         "Kept for frequency and spend; product-based measures (short-lived, markdown) use matched lines only"),
        (V, "Date alignment", f"{int((k < 0).sum()):,} deliveries ({int(r.qty[k < 0].sum()):,} units) arrive before the product's release date",
         "Counted as opening stock in week 1 (vis_forecast.py)"),
        (V, "Scope filters", f"{int(r.qty[k >= K].sum()):,} units delivered after week 12",
         "Excluded: the analysis window is each product's first 12 weeks"),
        (V, "Scope filters", f"{int((~win).sum()):,} purchase lines after February 2020 (the data holds only SS17-AW19 products; "
                             f"COVID closures follow)", "Excluded from the customer analysis"),
        (V, "Scope filters", f"{lc['products_total'] - lc['products_in_sample']:,} products sold fewer than {MIN_UNITS_LIFECYCLE} units in 12 weeks",
         f"Left out of lifecycle clustering only (unstable curve shape); {lc['products_in_sample']:,} products clustered"),
    ]
    return rows, s, r, c, g, neg, len(cards)


def survey_checks():
    q = TAB / "survey_quality.json"
    if not q.exists():
        return [], None
    q = json.loads(q.read_text())
    v = q["form_versions"]
    rows = [
        ("Survey", "Consent", f"{q['rows_in_file']} responses, {q['consented']} consented", "Only consenting responses are analysed"),
        ("Survey", "Duplicates", f"{q['duplicate_answer_rows']} identical answer rows", "None found"),
        ("Survey", "Missing values", f"Q9 (unplanned purchases) blank for {q['q9_unanswered']} people",
         "Q9 shares reported for those who answered"),
        ("Survey", "Inconsistencies", f"answer options edited mid-collection: {v.get('v1 original', 0)} original + {v.get('v2 edited', 0)} edited responses",
         f"Both mapped to common bands; unmapped answers: {len(q['unmapped'])} (survey_analysis.py)"),
        ("Survey", "Outliers", f"{q['conjoint_all_same_rating_excluded']} respondent gave all 8 conjoint options the same rating",
         "Excluded from the conjoint only (no preference information)"),
        ("Survey", "Privacy", f"{q['free_text_comments_not_shown']} free-text comments", "Never shown; no names or emails collected"),
    ]
    try:  # EDA table (local only)
        from .survey_analysis import SURVEY_RAW, load
        f = sorted(SURVEY_RAW.glob("*.csv"), key=lambda p: p.stat().st_mtime)[-1]
        d, _ = load(f)
        cols = [x for x in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "keep_years", "wears", "unworn_share",
                            "Q8a", "Q8b", "Q8c", "Q8d", "Q8e"] if x in d.columns]
        d[cols].rename(columns={"Q1": "age_band", "Q2": "occupation", "Q3": "location", "Q4": "monthly_spend",
                                "Q5": "buying_frequency", "Q6": "channel", "Q8a": "motive_trend", "Q8b": "motive_discount",
                                "Q8c": "motive_novelty", "Q8d": "motive_quality", "Q8e": "motive_environment"}) \
            .to_parquet(PROCESSED / "eda_survey.parquet", index=False)
    except (FileNotFoundError, IndexError, KeyError):
        pass
    return rows, q


def eda_tables():
    p = pd.read_parquet(PROCESSED / "vis_products.parquet")
    p[["external_code", "season", "category", "color", "fabric", "price", "n_shops", "delivered_12w", "sold_12w", "leftover_12w",
       "sell_through", "short_lived", "high_leftover", "cluster_name", "peak_week", "weeks_peak_to_half", "first_markdown_week"]] \
        .assign(first_markdown_week=lambda d: d.first_markdown_week + 1) \
        .to_parquet(PROCESSED / "eda_visuelle_products.parquet", index=False)
    ct = PROCESSED / "vis_customer_table.parquet"
    if ct.exists():
        c = pd.read_parquet(ct, columns=["freq_band", "segment_id", "recency_days", "purchase_days", "purchases_per_year",
                                         "units", "spend", "items_per_basket", "shops", "short_lived_share", "markdown_share"])
        seg = pd.read_csv(TAB / "vis_customer_segments.csv").set_index("segment_id").segment
        c.assign(segment=c.segment_id.map(seg)).drop(columns="segment_id").to_parquet(PROCESSED / "eda_customers.parquet", index=False)


def main():
    vis_rows, s, r, c, g, neg, n_cards = vis_checks()
    sv_rows, sv = survey_checks()
    log = pd.DataFrame(vis_rows + sv_rows, columns=["dataset", "check", "what_we_found", "how_handled"])
    save_table(log, "data_quality_log")
    p = pd.read_csv(VIS_RAW / "price_discount_series.csv")
    overview = pd.DataFrame([
        ("VISUELLE sales (sales.csv)", f"{len(s):,}", s.shape[1], "2017-2019, 6 seasons", "one product in one shop, 12 weekly sales",
         "lifecycles, stock, forecasting, simulation"),
        ("VISUELLE prices (price_discount_series.csv)", f"{len(p):,}", p.shape[1], "2017-2019", "one product in one shop, 12 weekly discounts",
         "markdowns, revenue, financial statement"),
        ("VISUELLE deliveries (restocks.csv)", f"{len(r):,}", r.shape[1], f"{r.year.min()}-{r.year.max()}", "one delivery",
         "stock available, leftover"),
        ("VISUELLE customers (customer_data.csv)", f"{len(c):,}", c.shape[1], f"{c.data.min():%b %Y} - {c.data.max():%b %Y}",
         "one purchase line (customer, shop, product, time)", "segments, CLTV, trends"),
        ("VISUELLE Google Trends (vis2_gtrends_data.csv)", f"{len(g):,}", g.shape[1], f"{g.date.min()[:7]} - {g.date.max()[:7]}",
         "one week, search interest per term", "market-interest test"),
        ("Our survey (Google Form)", f"{sv['consented']}" if sv else "-", "-",
         f"{sv['first_response'][:10]} to {sv['last_response'][:10]}" if sv else "-", "one respondent (29 questions)",
         "use, disposal, conjoint"),
    ], columns=["dataset", "rows", "columns", "period", "one row is", "used_for"])
    save_table(overview, "data_overview")
    eda_tables()

    src = "outputs/tables/data_quality_log.csv"
    register_claim("dq_vis_negative_cells", "VISUELLE negative weekly sales cells clipped to 0", neg, src)
    register_claim("dq_cards_excluded", "Customer IDs excluded as shop or generic cards (more than 10 shops)", n_cards, src)
    register_claim("dq_checks", "Data-quality checks recorded in the cleaning log", len(log), src)
    print(log.to_string())


if __name__ == "__main__":
    main()
