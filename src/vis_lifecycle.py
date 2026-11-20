"""Demand lifecycles and markdown timing (end of season) — VISUELLE 2.0, Nuna Lie.

Unit: product (all shops added up), first 12 weeks after release.
  - K-means on each product's normalised 12-week sales curve (k = 3..7 by silhouette).
  - Short-lived rule (fixed before modelling, src/vis_forecast.short_lived_flag): weekly sales fall below 25% of
    the peak within 4 weeks of it, and weeks 7-12 carry under 20% of 12-week units.
  - Markdown: product's average discount across shops at least 15% (sensitivity 10%, 20%).
Writes data/processed/vis_products.parquet (one row per product; reused by the risk model, customers, dashboards).
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from . import vis_forecast as vf
from .config import (HIGH_LEFTOVER_SELL_THROUGH, K_WEEKS, MARKDOWN_SENSITIVITY, MARKDOWN_THRESHOLD,
                     MIN_UNITS_LIFECYCLE, PROCESSED, RANDOM_STATE)
from .utils import PALETTE, SERIES, plt, register_claim, save_json, save_table, savefig

K = K_WEEKS


def weeks_below(U, frac):
    """Weeks from the peak until weekly units first fall below frac x peak (NaN if never, within 12 weeks)."""
    pk, peak = U.argmax(1), U.max(1)
    out = np.full(len(U), np.nan)
    for i in range(len(U)):
        hit = np.nonzero(U[i, pk[i] + 1:] < frac * peak[i])[0]
        if len(hit):
            out[i] = hit[0] + 1
    return out


def name_clusters(centers):
    """Name clusters from their centre shapes (every centre is a share of 12-week sales by week)."""
    names = {}
    for c, v in enumerate(centers):
        first3, last6 = v[:3].sum(), v[6:].sum()
        later_peak = int(v[2:].argmax()) + 2
        if last6 >= 0.45:
            names[c] = "Late bloomer"
        elif first3 >= 0.42:
            names[c] = "Fast fade"
        elif 3 <= later_peak <= 8 and v[later_peak] > 1.05 * v[2]:
            names[c] = "Mid-season peak"
        else:
            names[c] = "Steady decline"
    seen = {}
    for c in sorted(names):
        if names[c] in seen:
            names[c] = f"{names[c]} {seen[names[c]] + 1}"
        seen[names[c]] = seen.get(names[c], 0) + 1
    return names


def products():
    """Product table with 12-week sales (u*), deliveries (dl*) and average discount (d*) per week."""
    meta, S, D, Dsc, _ = vf.load()
    prod, Sp, Dp, dscp, idx, codes = vf.product_table(meta, S, D, Dsc)
    prod["high_leftover"] = prod.sell_through < HIGH_LEFTOVER_SELL_THROUGH
    for k in range(K):
        prod[f"u{k}"], prod[f"dl{k}"], prod[f"d{k}"] = Sp[:, k], Dp[:, k], dscp[:, k]
    return prod, Sp, dscp


def main():
    prod, Sp, dscp = products()
    keep = Sp.sum(1) >= MIN_UNITS_LIFECYCLE
    U, n = Sp[keep], int(keep.sum())
    shares = U / U.sum(1, keepdims=True)

    rng = np.random.default_rng(RANDOM_STATE)
    samp = rng.choice(n, size=min(5000, n), replace=False)
    scores, models = {}, {}
    for k in range(3, 8):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(shares)
        scores[k], models[k] = float(silhouette_score(shares[samp], km.labels_[samp])), km
    k_best = max(scores, key=scores.get)
    for k in (4, 5):  # prefer an interpretable 4-5 clusters if within 0.02 of the best silhouette
        if scores[k] >= scores[k_best] - 0.02:
            k_best = k
            break
    km = models[k_best]
    names = name_clusters(km.cluster_centers_)
    prod["cluster_name"] = None
    prod.loc[keep, "cluster_name"] = [names[c] for c in km.labels_]
    prod["in_lifecycle_sample"] = keep
    prod["peak_week"] = Sp.argmax(1) + 1
    prod["weeks_peak_to_half"] = weeks_below(Sp, 0.5)
    prod["share_last6"] = Sp[:, 6:].sum(1) / np.maximum(Sp.sum(1), 1)
    prod["marked_down_12w"] = ~np.isnan(prod.first_markdown_week)
    prod.to_parquet(PROCESSED / "vis_products.parquet", index=False)

    lc = prod[keep].copy()
    by_cluster = (lc.groupby("cluster_name").agg(
        products=("external_code", "size"), units=("sold_12w", "sum"), median_peak_week=("peak_week", "median"),
        mean_share_last6=("share_last6", "mean"), median_weeks_peak_to_half=("weeks_peak_to_half", "median"),
        share_marked_down=("marked_down_12w", "mean"), share_short_lived=("short_lived", "mean"),
        median_sell_through=("sell_through", "median"))
        .assign(share_products=lambda d: d.products / n).sort_values("median_peak_week").reset_index())
    save_table(by_cluster, "vis_lifecycle_by_cluster")

    # Markdown timing vs demand: demand (weekly units / own peak) in the first markdown week
    md = lc[lc.marked_down_12w].copy()
    Um = md[[f"u{k}" for k in range(K)]].to_numpy()
    fm = md.first_markdown_week.to_numpy().astype(int)
    md["demand_at_markdown_vs_peak"] = Um[np.arange(len(Um)), fm] / np.maximum(Um.max(1), 1)
    md["weeks_peak_to_markdown"] = fm + 1 - md.peak_week
    timing = (md.groupby("short_lived").agg(
        products_marked_down=("external_code", "size"), median_peak_week=("peak_week", "median"),
        median_first_markdown_week=("first_markdown_week", lambda s: s.median() + 1),
        median_weeks_peak_to_markdown=("weeks_peak_to_markdown", "median"),
        median_demand_at_markdown_vs_peak=("demand_at_markdown_vs_peak", "median"),
        share_marked_down_after_demand_halved=("demand_at_markdown_vs_peak", lambda s: (s < 0.5).mean()))
        .reset_index())
    save_table(timing, "vis_markdown_timing")

    sens = {}
    for thr in MARKDOWN_SENSITIVITY:
        m = dscp[keep] >= thr
        sens[f"{int(thr * 100)}pct"] = {"share_products_marked_down_12w": float(m.any(1).mean()),
                                        "share_units_sold_on_markdown": float((U * m).sum() / U.sum())}
    sl = lc[lc.short_lived]
    t_sl = timing.set_index("short_lived")
    md_all = {"marked_down_after_demand_halved": float((md.demand_at_markdown_vs_peak < 0.5).mean()),
              "median_demand_at_markdown_vs_peak": float(md.demand_at_markdown_vs_peak.median()),
              "median_first_markdown_week": float(md.first_markdown_week.median() + 1),
              "median_weeks_peak_to_markdown": float(md.weeks_peak_to_markdown.median()),
              "median_peak_week": float(lc.peak_week.median()),
              "share_peak_in_first_3_weeks": float((lc.peak_week <= 3).mean())}
    summary = {
        "products_total": int(len(prod)), "products_in_sample": n, "units_in_sample": float(U.sum()),
        "min_units": MIN_UNITS_LIFECYCLE, "k_chosen": k_best, "silhouette_by_k": scores,
        "cluster_names": {int(k): v for k, v in names.items()},
        "cluster_shares": {r.cluster_name: float(r.share_products) for r in by_cluster.itertuples()},
        "short_lived_definition": "falls below 25% of peak within 4 weeks of it AND weeks 7-12 < 20% of 12-week units",
        "share_short_lived": float(lc.short_lived.mean()), "short_lived_products": int(lc.short_lived.sum()),
        "short_lived_median_weeks_peak_to_half": float(sl.weeks_peak_to_half.median()),
        "short_lived_median_peak_week": float(sl.peak_week.median()),
        "other_median_weeks_peak_to_half": float(lc[~lc.short_lived].weeks_peak_to_half.median()),
        "share_marked_down_12w": float(lc.marked_down_12w.mean()),
        "share_units_on_markdown": sens[f"{int(MARKDOWN_THRESHOLD * 100)}pct"]["share_units_sold_on_markdown"],
        "markdown_sensitivity": sens,
        "short_lived_marked_down_after_demand_halved": float(t_sl.loc[True, "share_marked_down_after_demand_halved"]),
        "other_marked_down_after_demand_halved": float(t_sl.loc[False, "share_marked_down_after_demand_halved"]),
        "short_lived_median_weeks_peak_to_markdown": float(t_sl.loc[True, "median_weeks_peak_to_markdown"]),
        "other_median_weeks_peak_to_markdown": float(t_sl.loc[False, "median_weeks_peak_to_markdown"]),
        "short_lived_median_first_markdown_week": float(t_sl.loc[True, "median_first_markdown_week"]),
        "all_products_markdown": md_all,
        "short_lived_sell_through_median": float(sl.sell_through.median()),
        "other_sell_through_median": float(lc[~lc.short_lived].sell_through.median()),
    }
    save_json(summary, "vis_lifecycle_summary")
    print(by_cluster.to_string())
    print(timing.to_string())

    src = "outputs/tables/vis_lifecycle_summary.json"
    rc = register_claim
    rc("lc_n", "Products in the lifecycle sample (at least 30 units in 12 weeks)", n, src)
    rc("lc_k", "Number of lifecycle clusters (k-means, silhouette)", k_best, src)
    rc("lc_short_lived_share", "Share of products that are short-lived", summary["share_short_lived"], src)
    rc("lc_short_lived_n", "Short-lived products", summary["short_lived_products"], src)
    rc("lc_sl_weeks_to_half", "Short-lived products: median weeks from peak to half of peak", summary["short_lived_median_weeks_peak_to_half"], src)
    rc("lc_md_share", "Share of products marked down (average discount >= 15%) within 12 weeks", summary["share_marked_down_12w"], src)
    rc("lc_units_on_md", "Share of 12-week units sold in weeks with >= 15% average discount", summary["share_units_on_markdown"], src)
    rc("lc_sl_md_after_half", "Short-lived products marked down only after demand fell below half of peak", summary["short_lived_marked_down_after_demand_halved"], src)
    rc("lc_other_md_after_half", "Other products marked down only after demand fell below half of peak", summary["other_marked_down_after_demand_halved"], src)
    rc("lc_md_after_half", "All marked-down products: share cut only after demand fell below half of peak", md_all["marked_down_after_demand_halved"], src)
    rc("lc_demand_at_md", "All marked-down products: median weekly demand at first markdown, as share of peak", md_all["median_demand_at_markdown_vs_peak"], src)
    rc("lc_first_md_week", "All marked-down products: median first markdown week", md_all["median_first_markdown_week"], src)
    rc("lc_peak_first3", "Share of products whose sales peak in weeks 1-3", md_all["share_peak_in_first_3_weeks"], src)
    rc("lc_sl_weeks_peak_to_md", "Short-lived products: median weeks from peak to first markdown", summary["short_lived_median_weeks_peak_to_markdown"], src)
    plots(lc, km, names, scores)


def plots(lc, km, names, scores):
    weeks = np.arange(1, K + 1)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    order = np.argsort([-c[:3].sum() for c in km.cluster_centers_])  # most front-loaded first
    for i, c in enumerate(order):
        share = (lc.cluster_name == names[c]).mean()
        ax.plot(weeks, km.cluster_centers_[c] * 100, lw=2.4, color=SERIES[i % len(SERIES)], label=f"{names[c]} ({share:.0%} of products)")
    ax.set_xlabel("Week since release")
    ax.set_ylabel("% of the product's 12-week sales")
    ax.set_xticks(weeks)
    ax.set_title("Nuna Lie products fall into distinct demand lifecycles")
    ax.legend(frameon=False, fontsize=8.5)
    savefig(fig, "vis_lifecycle_clusters")

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ks = sorted(scores)
    ax.plot(ks, [scores[k] for k in ks], marker="o", color=PALETTE["teal"])
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Choosing k")
    savefig(fig, "vis_cluster_silhouette")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for flag, col, lab in ((True, PALETTE["accent"], "Short-lived products"), (False, PALETTE["teal"], "Other products")):
        d = lc[lc.short_lived == flag]
        U = d[[f"u{k}" for k in range(K)]].to_numpy()
        Dm = d[[f"d{k}" for k in range(K)]].to_numpy() >= MARKDOWN_THRESHOLD
        ax.plot(weeks, (U / U.max(1, keepdims=True)).mean(0) * 100, lw=2.4, color=col, label=f"{lab}: demand (% of own peak)")
        ax.plot(weeks, Dm.mean(0) * 100, lw=1.8, ls="--", color=col, label=f"{lab}: % on markdown")
    ax.set_xlabel("Week since release")
    ax.set_ylabel("%")
    ax.set_xticks(weeks)
    ax.set_title("Markdowns rise as demand fades")
    ax.legend(frameon=False, fontsize=8.5)
    savefig(fig, "vis_markdown_vs_demand")


if __name__ == "__main__":
    main()
