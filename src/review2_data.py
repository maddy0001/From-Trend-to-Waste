"""Data for the Review 2 (mid-project) dashboard and 8-slide deck -> outputs/review2/review2_data.json

All retail data is VISUELLE 2.0 (Nuna Lie). Five exploratory questions, each answered by one chart:
  Q1 What does a product's 12-week life look like?          (lifecycle clusters, by category)
  Q2 Do markdowns follow demand, or lag behind it?           (demand vs % of products on markdown, by week)
  Q3 Who buys: how do frequent and occasional customers differ? (loyalty-card customers, by buying frequency)
  Q4 How much delivered stock is left unsold?               (by category and season)
  Q5 Does pushing more stock per shop leave more unsold?     (also: weeks 1-2 sales vs the season's deliveries)
Aggregates go to outputs/review2/; the per-product table for the Q5 scatter stays in data/processed (licensed data).
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import K_WEEKS, MARKDOWN_THRESHOLD, PROCESSED, ROOT, TAB
from .utils import register_claim
from .vis_forecast import load as vis_load

OUT = ROOT / "outputs" / "review2"
K = K_WEEKS


def curves():
    """Q1: average share of 12-week sales by week for each lifecycle cluster; Q2: demand (% of own peak) and % of
    products on markdown by week. Both for all products and for the six largest categories."""
    p = pd.read_parquet(PROCESSED / "vis_products.parquet")
    p = p[p.in_lifecycle_sample]
    top = p.category.value_counts().index[:6].tolist()
    U = p[[f"u{k}" for k in range(K)]].to_numpy()
    shares, rel = U / U.sum(1, keepdims=True), U / U.max(1, keepdims=True)
    md = p[[f"d{k}" for k in range(K)]].to_numpy() >= MARKDOWN_THRESHOLD
    lc_rows, md_rows, groups = [], [], []
    for g in ["All products"] + top:
        m = np.ones(len(p), bool) if g == "All products" else (p.category == g).values
        groups.append({"group": g, "products": int(m.sum()), "short_lived_share": float(p.short_lived[m].mean())})
        for name in sorted(p.cluster_name.unique()):
            k = m & (p.cluster_name == name).values
            for w in range(K):
                lc_rows.append({"group": g, "cluster": name, "week": w + 1, "products": int(k.sum()),
                                "share_of_products": float(k.sum() / m.sum()), "pct_of_12w_sales": float(shares[k, w].mean() * 100)})
        for t, k in (("All products", m), ("Short-lived", m & p.short_lived.values), ("Other", m & ~p.short_lived.values)):
            for w in range(K):
                md_rows.append({"group": g, "type": t, "week": w + 1, "products": int(k.sum()),
                                "demand_pct_of_peak": float(rel[k, w].mean() * 100) if k.any() else None,
                                "pct_on_markdown": float(md[k, w].mean() * 100) if k.any() else None})
    return pd.DataFrame(lc_rows), pd.DataFrame(md_rows), pd.DataFrame(groups)


def vis_tables():
    meta, S, D, _, _ = vis_load()
    df = meta[["external_code", "season", "category"]].assign(
        sold=S.sum(1), delivered=D.sum(1), sold_w12=S[:, :2].sum(1), shops=1,
        leftover=np.clip(D.sum(1) - S.sum(1), 0, None))  # leftover per product-shop, as in vis_forecast.py
    prod = df.groupby("external_code").agg(season=("season", "first"), category=("category", "first"),
                                           sold=("sold", "sum"), delivered=("delivered", "sum"), sold_w12=("sold_w12", "sum"),
                                           shops=("shops", "sum"), leftover=("leftover", "sum")).reset_index()
    prod = prod[prod.delivered > 0].copy()
    prod["deliveries_per_shop"] = prod.delivered / prod.shops
    prod["early_share"] = prod.sold_w12 / prod.delivered           # weeks 1-2 sales / the season's 12-week deliveries
    prod["sell_through_12w"] = (prod.sold / prod.delivered).clip(upper=1)
    prod["leftover_share"] = prod.leftover / prod.delivered
    prod.to_parquet(PROCESSED / "r2_vis_products.parquet", index=False)
    cat = df.groupby(["season", "category"], as_index=False)[["delivered", "sold", "leftover"]].sum()
    rel = {}
    for x in ("deliveries_per_shop", "early_share"):
        rho, p = spearmanr(prod[x], prod.sell_through_12w)
        d = prod.assign(dec=pd.qcut(prod[x].rank(method="first"), 10, labels=range(1, 11)).astype(int)).groupby("dec").agg(
            x=(x, "median"), sell_through=("sell_through_12w", "mean"), products=("external_code", "size")).reset_index()
        rel[x] = {"rho": float(rho), "p": float(p), "deciles": d.to_dict("records")}
    return cat, rel, len(prod)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lcc, mdc, groups = curves()
    cat, rel, n_prod = vis_tables()
    vis = json.loads((TAB / "vis_summary.json").read_text())
    lc = json.loads((TAB / "vis_lifecycle_summary.json").read_text())
    cs = json.loads((TAB / "vis_customers_summary.json").read_text())
    freq = pd.read_csv(TAB / "vis_customers_by_frequency.csv")
    by_cat = cat.groupby("category")[["delivered", "leftover"]].sum()
    by_cat = by_cat[by_cat.delivered >= by_cat.delivered.quantile(0.25)]
    left_cat = (by_cat.leftover / by_cat.delivered).sort_values()
    sv = json.loads((TAB / "survey_quality.json").read_text()) if (TAB / "survey_quality.json").exists() else {}
    am = lc["all_products_markdown"]
    data = {
        "lifecycle_curves": lcc.to_dict("records"), "markdown_curves": mdc.to_dict("records"), "groups": groups.to_dict("records"),
        "by_freq": freq.to_dict("records"), "vis_category_season": cat.to_dict("records"), "relationships": rel,
        "stats": {
            "products": vis["stock"]["products"], "shops": vis["stock"]["shops"], "lc_products": lc["products_in_sample"],
            "lc_k": lc["k_chosen"], "cluster_shares": lc["cluster_shares"],
            "peak_first3": am["share_peak_in_first_3_weeks"], "first_md_week": am["median_first_markdown_week"],
            "weeks_peak_to_md": am["median_weeks_peak_to_markdown"], "demand_at_md": am["median_demand_at_markdown_vs_peak"],
            "md_after_half": am["marked_down_after_demand_halved"], "sl_md_after_half": lc["short_lived_marked_down_after_demand_halved"],
            "sl_share": lc["share_short_lived"],
            "customers": cs["customers"], "purchase_lines": cs["purchase_lines_raw"], "top20_units": cs["share_units_top20pct_customers"],
            "top10_units": cs["share_units_top10pct_customers"], "md_lo": cs["markdown_share_lowest_band"],
            "md_hi": cs["markdown_share_highest_band"], "sl_lo": cs["short_lived_share_lowest_band"],
            "sl_hi": cs["short_lived_share_highest_band"], "rho_md": cs["spearman_frequency_vs_markdown_share"]["rho"],
            "one_day": cs["share_one_purchase_day"],
            "vis_leftover": vis["stock"]["leftover_share_of_delivered"], "stockout": vis["stock"]["share_shop_weeks_ending_at_zero_stock"],
            "cat_left_lo": float(left_cat.iloc[0]), "cat_left_lo_name": left_cat.index[0],
            "cat_left_hi": float(left_cat.iloc[-1]), "cat_left_hi_name": left_cat.index[-1],
            "q5_n": n_prod,
            "dps_rho": rel["deliveries_per_shop"]["rho"], "early_rho": rel["early_share"]["rho"],
            "dps_d1_st": rel["deliveries_per_shop"]["deciles"][0]["sell_through"],
            "dps_d10_st": rel["deliveries_per_shop"]["deciles"][-1]["sell_through"],
            "early_d1_st": rel["early_share"]["deciles"][0]["sell_through"],
            "early_d10_st": rel["early_share"]["deciles"][-1]["sell_through"],
            "survey_n": sv.get("consented"),
        },
    }
    log = pd.read_csv(TAB / "data_quality_log.csv")
    pick = [("VISUELLE 2.0", "Missing values", "sales"), ("VISUELLE 2.0", "Outliers", "negative"),
            ("VISUELLE 2.0", "Outliers", "customer IDs"), ("VISUELLE 2.0", "Inconsistencies", ""),
            ("VISUELLE 2.0", "Date alignment", ""), ("VISUELLE 2.0", "Scope filters", "after week 12"),
            ("Survey", "Inconsistencies", "")]
    data["cleaning"] = [log[(log.dataset == d) & (log.check == c) & log.what_we_found.str.contains(t, regex=False)].iloc[0]
                        [["dataset", "check", "what_we_found", "how_handled"]].tolist() for d, c, t in pick]
    data["n_checks"] = len(log)
    data["overview"] = pd.read_csv(TAB / "data_overview.csv").to_dict("records")
    (OUT / "review2_data.json").write_text(json.dumps(data, indent=1, default=float))
    src = "outputs/review2/review2_data.json"
    st_ = data["stats"]
    register_claim("r2_dps_rho", "Spearman, deliveries per shop vs 12-week sell-through", st_["dps_rho"], src)
    register_claim("r2_dps_deciles", "Mean sell-through, fewest vs most deliveries per shop (bottom/top 10%)",
                   f"{st_['dps_d1_st']:.3f} vs {st_['dps_d10_st']:.3f}", src)
    register_claim("r2_early_rho", "Spearman, weeks 1-2 sales / season deliveries vs 12-week sell-through", st_["early_rho"], src)
    register_claim("r2_early_deciles", "Mean sell-through, slowest vs fastest early sellers (bottom/top 10%)",
                   f"{st_['early_d1_st']:.3f} vs {st_['early_d10_st']:.3f}", src)
    register_claim("r2_cat_leftover_range", "Leftover share range across categories (larger 75%)",
                   f"{left_cat.iloc[0]:.3f} ({left_cat.index[0]}) to {left_cat.iloc[-1]:.3f} ({left_cat.index[-1]})", src)
    print(json.dumps(data["stats"], indent=1, default=float))


if __name__ == "__main__":
    main()
