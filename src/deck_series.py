"""Export the data series behind the deck's native PowerPoint charts -> outputs/tables/deck_series.json (VISUELLE + survey)."""
import json

import numpy as np
import pandas as pd

from .config import K_WEEKS, MARKDOWN_THRESHOLD, PROCESSED, SEASON_ORDER, TAB, VIS_RAW

K = K_WEEKS


def main():
    p = pd.read_parquet(PROCESSED / "vis_products.parquet")
    lc = p[p.in_lifecycle_sample]
    U = lc[[f"u{k}" for k in range(K)]].to_numpy()
    S = U / U.sum(1, keepdims=True)
    shares = json.loads((TAB / "vis_lifecycle_summary.json").read_text())["cluster_shares"]
    order = sorted(shares, key=lambda n: -S[(lc.cluster_name == n).values].mean(0)[:3].sum())  # most front-loaded first
    clusters = [{"name": n, "share": shares[n], "values": [round(float(v) * 100, 2) for v in S[(lc.cluster_name == n).values].mean(0)]}
                for n in order]
    md = lc[[f"d{k}" for k in range(K)]].to_numpy() >= MARKDOWN_THRESHOLD
    rel = U / U.max(1, keepdims=True)
    demand = {"all": {"demand_pct_of_peak": [round(float(v) * 100, 1) for v in rel.mean(0)],
                      "pct_on_markdown": [round(float(v) * 100, 1) for v in md.mean(0)]}}

    risk = pd.read_csv(TAB / "vis_risk_model_metrics.csv")
    orr = pd.read_csv(TAB / "vis_risk_odds_ratios.csv")
    sig = orr[(orr.p_value < 0.05) & ~orr.term.str.contains("rare")]

    def label(t):
        k, v = t.split(" = ", 1)
        if k == "season_type":
            return "Autumn-winter" if v == "AW" else "Spring-summer"
        if k == "price_q":
            return "Priciest quarter" if "priciest" in v else f"Price {v}"
        if k == "depth_q":
            return "Deepest initial buy" if "deepest" in v else f"Depth {v}"
        return v.capitalize() + (" (fabric)" if k == "fabric" else " (colour)" if k == "color" else "")
    odds = {"up": [{"label": label(r.term), "or": round(r.odds_ratio, 2)} for r in sig[sig.odds_ratio > 1].itertuples()],
            "down": [{"label": label(r.term), "or": round(r.odds_ratio, 2)} for r in sig[sig.odds_ratio < 1].itertuples()]}

    seg = pd.read_csv(TAB / "vis_customer_segments.csv").sort_values("share_of_units", ascending=False)
    freq = pd.read_csv(TAB / "vis_customers_by_frequency.csv")
    seas = pd.read_csv(TAB / "vis_leftover_by_season.csv").set_index("season").loc[SEASON_ORDER]
    fc = pd.read_csv(TAB / "vis_forecast_metrics.csv")
    fc = fc[fc.group == "All products"].sort_values("weekly_wape", ascending=False)
    grid = pd.read_csv(TAB / "vis_simulation_grid.csv")
    g = grid[(grid.flag_threshold_pred_sell_through == 0.6) & (grid.uplift_estimate == "FE (conservative)")]
    a = g[g.policy == "A: cut later deliveries"].sort_values("delivery_cut")
    ab = g[(g.policy == "A+B") & (g.markdown_level == 0.2)].sort_values("delivery_cut")
    fin = pd.read_csv(TAB / "vis_financial_statement.csv").set_index("season").loc[SEASON_ORDER]
    si = pd.read_csv(TAB / "vis_seasonal_index.csv").set_index("month").seasonal_index
    mnames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ps = pd.read_csv(TAB / "pilot_sample_size.csv")
    fb = lambda b: b.replace(" times a year", "/yr").replace("Once a year or less", "≤1/yr").replace("Monthly or more", "Monthly+")

    out = {
        "clusters": clusters, "markdown_vs_demand": demand,
        "risk_auc": [{"label": f"{r.model}", "stage": r.stage, "auc": round(r.roc_auc, 3)} for r in risk.itertuples()],
        "odds": odds,
        "segments": {"labels": seg.segment.tolist(), "pct_customers": (seg.share_of_customers * 100).round(1).tolist(),
                     "pct_units": (seg.share_of_units * 100).round(1).tolist(), "cltv": seg.cltv_index.round(2).tolist()},
        "freq": {"labels": [fb(x) for x in freq.freq_band], "markdown": (freq.markdown_share * 100).round(1).tolist(),
                 "short_lived": (freq.short_lived_share * 100).round(1).tolist(),
                 "pct_customers": (freq.share_of_customers * 100).round(1).tolist(),
                 "pct_units": (freq.share_of_units * 100).round(1).tolist()},
        "seasons": {"labels": seas.index.tolist(), "sold_k": (seas.sold / 1000).round(1).tolist(),
                    "unsold_k": (seas.leftover / 1000).round(1).tolist(),
                    "unsold_pct": (seas.leftover_share * 100).round(0).tolist()},
        "forecast": {"labels": fc.model.tolist(), "wape": (fc.weekly_wape * 100).round(1).tolist()},
        "simulation": {"cuts": (a.delivery_cut * 100).round(0).tolist(),
                       "A_leftover": (a.leftover_change * 100).round(1).tolist(),
                       "AB_leftover": (ab.leftover_change * 100).round(1).tolist(),
                       "A_revenue": (a.revenue_change * 100).round(2).tolist(),
                       "AB_revenue": (ab.revenue_change * 100).round(2).tolist()},
        "financials": {"seasons": SEASON_ORDER, "stock_to_sales": (fin.stock_to_sales * 100).round(1).tolist(),
                       "markdown_rate": (fin.markdown_rate * 100).round(1).tolist(),
                       "gm50": (fin.gross_margin_cost50 * 100).round(1).tolist(),
                       "mu50": (fin.margin_after_unsold_cost50 * 100).round(1).tolist()},
        "seasonal_index": {"labels": mnames, "values": [round(float(si[m]) * 100, 0) for m in range(1, 13)]},
        "pilot": {"labels": [f"{int(f * 100)}% of simulated effect" for f in ps[ps.design == "Randomise shops"].effect_as_share_of_simulated],
                  "shops": ps[ps.design == "Randomise shops"].units_per_arm.astype(int).tolist(),
                  "products": ps[ps.design == "Randomise flagged products"].units_per_arm.astype(int).tolist()},
    }
    # Market search interest (Google Trends, shipped with VISUELLE 2.0) for the three most common categories
    cats = (pd.read_csv(VIS_RAW / "sales.csv", usecols=["external_code", "category"]).drop_duplicates("external_code")
            .category.value_counts())
    top = sorted(cats.index[:3], key=lambda c: -cats[c])
    gt = pd.read_csv(VIS_RAW / "vis2_gtrends_data.csv", parse_dates=["date"]).set_index("date").sort_index()
    gm = gt.loc["2016-01-01":"2019-12-31", top].resample("MS").mean()
    tr = json.loads((TAB / "trends_summary.json").read_text())["google_trends"]
    out["gtrends"] = {"labels": gm.index.strftime("%b %y").tolist(),
                      "series": [{"name": c, "values": gm[c].round(1).tolist()} for c in top],
                      "wape_with": [round(tr["prelaunch_total_sales"]["with Google Trends"]["wape"] * 100, 1),
                                    round(tr["week2_weekly_wape"]["with Google Trends"] * 100, 1)],
                      "wape_without": [round(tr["prelaunch_total_sales"]["without Google Trends"]["wape"] * 100, 1),
                                       round(tr["week2_weekly_wape"]["without Google Trends"] * 100, 1)]}

    log = pd.read_csv(TAB / "data_quality_log.csv")  # rows for the data-cleaning slide (full log: dashboard Data & EDA tab)
    pick = [("VISUELLE 2.0", "Missing values", "sales"), ("VISUELLE 2.0", "Duplicates", "customer"),
            ("VISUELLE 2.0", "Outliers", "negative"), ("VISUELLE 2.0", "Outliers", "customer IDs"),
            ("VISUELLE 2.0", "Inconsistencies", ""), ("VISUELLE 2.0", "Date alignment", ""),
            ("VISUELLE 2.0", "Scope filters", "after week 12"), ("VISUELLE 2.0", "Scope filters", "February 2020"),
            ("Survey", "Inconsistencies", "")]
    rows = []
    for d, c, t in pick:
        m = log[(log.dataset == d) & (log.check == c) & log.what_we_found.str.contains(t, regex=False)]
        if len(m):
            rows.append(m.iloc[0][["dataset", "check", "what_we_found", "how_handled"]].tolist())
    out["data_quality"] = {"rows": rows, "n_checks": len(log)}

    sv_path = TAB / "survey_descriptives.json"
    if sv_path.exists():  # real responses only (src/survey_analysis.py refuses fixtures)
        sv = json.loads(sv_path.read_text())
        often, never = sv["disposal_share_often_or_always"], sv["disposal_share_never"]
        routes = sorted(often, key=often.get, reverse=True)
        pw = pd.read_csv(TAB / "survey_conjoint_partworths.csv").sort_values("part_worth")
        bf = pd.read_csv(TAB / "survey_vis_bridge_by_frequency.csv")
        out["survey"] = {
            # whole percents rounded in Python, so chart labels match the report's text
            "disposal": {"labels": routes, "often": [round(often[r] * 100) for r in routes],
                         "never": [round(never[r] * 100) for r in routes]},
            "partworths": {"labels": pw.attribute.map({"price_high": "Rs 1,499 vs Rs 799", "trendy": "Trendy vs timeless",
                                                       "certified": "Sustainability label", "long_life": "Lasts 3+ years vs 1 season"}).tolist(),
                           "values": pw.part_worth.round(2).tolist()},
            "freq": {"labels": bf.freq_band.tolist(), "trend": bf.trend_influence.round(2).tolist(),
                     "discount": bf.discount_influence.round(2).tolist(), "n": bf.respondents.astype(int).tolist(),
                     "vis_markdown": (bf.vis_markdown_share * 100).round(1).tolist()},
        }
    (TAB / "deck_series.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote outputs/tables/deck_series.json")


if __name__ == "__main__":
    main()
