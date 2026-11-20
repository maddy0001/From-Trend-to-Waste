"""Customers: RFM segmentation, CLTV, and who buys short-lived / marked-down products — VISUELLE 2.0 customer_data.

Data: loyalty-card purchases (customer, shop, product, timestamp, quantity), joined to the sales and price files by
product code + shop + date. Window: 1 Dec 2016 - 29 Feb 2020 (the dataset holds only SS17-AW19 products; COVID
closures follow). IDs that buy in more than 10 shops are shop or generic cards and are excluded.
  - Markdown purchase: bought in a week (of the product's first 12 weeks in that shop) with a discount >= 15%.
    Purchases outside that window have no recorded discount and are left out of the markdown share.
  - Spend uses the dataset's normalised prices (index units, not euros).
  - Frequency = purchase days per year since the customer's first purchase (at least one year).
  - CLTV = last-12-month spend x expected lifetime, lifetime = 1 / (1 - 12-month retention), capped at 5 years.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import vis_forecast as vf
from .config import (CUSTOMER_END, CUSTOMER_MAX_SHOPS, CUSTOMER_START, FREQ_BANDS, FREQ_LABELS, MARKDOWN_THRESHOLD,
                     PROCESSED, RANDOM_STATE, VIS_RAW)
from .utils import PALETTE, plt, register_claim, save_json, save_table, savefig

END = pd.Timestamp(CUSTOMER_END)
Y1 = (pd.Timestamp("2018-03-01"), pd.Timestamp("2019-03-01"))   # retention: active in Y1 ...
Y2 = (pd.Timestamp("2019-03-01"), END)                          # ... and bought again in Y2


def purchases():
    c = pd.read_csv(VIS_RAW / "customer_data.csv", parse_dates=["data"])
    n_raw = len(c)
    c = c[(c.data >= pd.Timestamp(CUSTOMER_START)) & (c.data < END)]
    n_window = len(c)
    shops = c.groupby("customer").retail.nunique()
    cards = shops[shops > CUSTOMER_MAX_SHOPS].index
    excluded_units = int(c[c.customer.isin(cards)].qty.sum())
    c = c[~c.customer.isin(cards)].copy()
    meta, S, D, Dsc, _ = vf.load()
    key = meta.reset_index()[["index", "external_code", "retail", "release_date", "price"]]
    key["release_date"] = pd.to_datetime(key.release_date)
    c = c.merge(key, on=["external_code", "retail"], how="left")
    prod = pd.read_parquet(PROCESSED / "vis_products.parquet")[["external_code", "price", "short_lived", "high_leftover"]]
    c = c.merge(prod.rename(columns={"price": "product_price"}), on="external_code", how="left")
    c["unit_price"] = c.price.fillna(c.product_price)
    c["week"] = (c.data.dt.normalize() - c.release_date).dt.days // 7
    inwin = c["index"].notna() & (c.week >= 0) & (c.week < vf.K)
    disc = np.full(len(c), np.nan)
    disc[inwin.values] = Dsc[c.loc[inwin, "index"].astype(int).values, c.loc[inwin, "week"].astype(int).values]
    c["markdown"] = np.where(np.isnan(disc), np.nan, disc >= MARKDOWN_THRESHOLD)
    c["day"] = c.data.dt.normalize()
    notes = {"purchase_lines_raw": n_raw, "purchase_lines_in_window": n_window, "card_ids_excluded": int(len(cards)),
             "units_on_excluded_cards": excluded_units, "purchase_lines_used": int(len(c)),
             "share_lines_matched_to_shop_sales": float(c["index"].notna().mean()),
             "share_lines_with_known_discount": float(inwin.mean()),
             "share_lines_with_known_product": float(c.short_lived.notna().mean())}
    return c, notes


def customer_table(c):
    c = c.assign(spend=c.qty * c.unit_price,
                 sl_units=np.where(c.short_lived.isna(), np.nan, c.qty * c.short_lived.fillna(False)),
                 hl_units=np.where(c.high_leftover.isna(), np.nan, c.qty * c.high_leftover.fillna(False)),
                 known_units=np.where(c.short_lived.isna(), 0, c.qty),
                 last12=c.data >= END - pd.DateOffset(years=1),
                 y1=(c.data >= Y1[0]) & (c.data < Y1[1]), y2=(c.data >= Y2[0]) & (c.data < Y2[1]))
    g = c.groupby("customer")
    t = pd.DataFrame({
        "first": g.data.min(), "last": g.data.max(), "purchase_days": g.day.nunique(), "units": g.qty.sum(),
        "spend": g.spend.sum(), "shops": g.retail.nunique(), "known_units": g.known_units.sum(),
        "short_lived_units": g.sl_units.sum(), "high_leftover_units": g.hl_units.sum(),
        "markdown_lines": g.markdown.sum(), "known_discount_lines": g.markdown.count(),
        "spend_last12m": c[c.last12].groupby("customer").spend.sum(),
        "active_y1": g.y1.any(), "active_y2": g.y2.any()}).fillna({"spend_last12m": 0})
    t["recency_days"] = (END - t["last"]).dt.days
    t["tenure_years"] = np.maximum((END - t["first"]).dt.days / 365.25, 1.0)
    t["purchases_per_year"] = t.purchase_days / t.tenure_years
    t["items_per_basket"] = t.units / t.purchase_days
    t["spend_per_basket"] = t.spend / t.purchase_days
    t["short_lived_units"] = t.short_lived_units.astype(float)
    t["high_leftover_units"] = t.high_leftover_units.astype(float)
    t["markdown_lines"] = t.markdown_lines.astype(float)
    t["short_lived_share"] = np.where(t.known_units >= 3, t.short_lived_units / t.known_units.clip(lower=1), np.nan)
    t["high_leftover_share"] = np.where(t.known_units >= 3, t.high_leftover_units / t.known_units.clip(lower=1), np.nan)
    t["markdown_share"] = np.where(t.known_discount_lines >= 3, t.markdown_lines / t.known_discount_lines.clip(lower=1), np.nan)
    t["freq_band"] = pd.cut(t.purchases_per_year, FREQ_BANDS, labels=FREQ_LABELS, right=False)
    return t.reset_index().sort_values("customer").reset_index(drop=True)


def segment(t):
    X = StandardScaler().fit_transform(np.column_stack([
        np.log1p(t.recency_days), np.log1p(t.purchase_days), np.log1p(t.spend_per_basket * 1000), np.log1p(t.items_per_basket)]))
    rng = np.random.default_rng(RANDOM_STATE)
    samp = rng.choice(len(X), size=20000, replace=False)
    scores, models = {}, {}
    for k in range(3, 8):
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, batch_size=8192, n_init=5).fit(X)
        scores[k], models[k] = float(silhouette_score(X[samp], km.labels_[samp])), km
    k_best = max(scores, key=scores.get)
    for k in (4, 5, 6):  # prefer 4-6 segments when within 0.02 of the best: easier to act on
        if scores[k] >= scores[k_best] - 0.02:
            k_best = k
            break
    return models[k_best].labels_, scores, k_best


def name_segments(prof):
    """Readable names from each segment's median purchase days and recency (not from assumptions)."""
    names = {}
    for sid, r in prof.iterrows():
        f = "Frequent" if r.purchase_days_median >= 6 else "Repeat" if r.purchase_days_median >= 2 else "One-time"
        rec = "active" if r.recency_median <= 180 else "lapsing" if r.recency_median <= 540 else "lapsed"
        names[sid] = f"{f}, {rec}"
    dup = pd.Series(names).duplicated(keep=False)
    for sid in prof.index:  # same frequency and recency: tell them apart by basket size
        if dup[sid]:
            names[sid] += ", multi-item baskets" if prof.loc[sid, "items_per_basket_mean"] >= 2 else ", single items"
    return names


def main():
    c, notes = purchases()
    t = customer_table(c)
    labels, scores, k = segment(t)
    t["segment_id"] = labels
    t.to_parquet(PROCESSED / "vis_customer_table.parquet", index=False)
    n = len(t)

    g = t.groupby("segment_id")
    prof = pd.DataFrame({
        "customers": g.size(), "purchase_days_median": g.purchase_days.median(), "recency_median": g.recency_days.median(),
        "purchases_per_year_median": g.purchases_per_year.median(), "items_per_basket_mean": g.items_per_basket.mean(),
        "spend_per_basket_median": g.spend_per_basket.median(), "spend_per_customer_mean": g.spend.mean(),
        "short_lived_share_mean": g.short_lived_share.mean(), "markdown_share_mean": g.markdown_share.mean(),
        "high_leftover_share_mean": g.high_leftover_share.mean(),
        "retention_12m": t[t.active_y1].groupby("segment_id").active_y2.mean(),
        "spend_last12m_mean": g.spend_last12m.mean(),
        "share_of_units": g.units.sum() / t.units.sum(), "share_of_spend": g.spend.sum() / t.spend.sum(),
    })
    prof["share_of_customers"] = prof.customers / n
    prof["expected_lifetime_years"] = (1 / (1 - prof.retention_12m.fillna(0).clip(upper=0.8))).clip(upper=5)
    prof["cltv_index"] = prof.spend_last12m_mean * prof.expected_lifetime_years
    names = name_segments(prof)
    prof.insert(0, "segment", prof.index.map(names))
    prof = prof.sort_values("cltv_index", ascending=False).reset_index()
    save_table(prof, "vis_customer_segments")

    fb = t.groupby("freq_band", observed=True).agg(
        customers=("customer", "size"), units=("units", "sum"), short_lived_share=("short_lived_share", "mean"),
        markdown_share=("markdown_share", "mean"), high_leftover_share=("high_leftover_share", "mean"),
        spend_per_customer=("spend", "mean")).reindex(FREQ_LABELS)
    fb["share_of_customers"] = fb.customers / n
    fb["share_of_units"] = fb.units / t.units.sum()
    fb = fb.reset_index()
    save_table(fb, "vis_customers_by_frequency")

    ok_sl, ok_md = t.short_lived_share.notna(), t.markdown_share.notna()
    f64 = lambda x: np.asarray(x, dtype=float)
    rho_sl, p_sl = spearmanr(f64(t.purchases_per_year[ok_sl]), f64(t.short_lived_share[ok_sl]))
    rho_md, p_md = spearmanr(f64(t.purchases_per_year[ok_md]), f64(t.markdown_share[ok_md]))
    rho_hl, p_hl = spearmanr(f64(t.purchases_per_year[ok_sl]), f64(t.high_leftover_share[ok_sl]))
    u = t.units.sort_values(ascending=False).to_numpy()
    cum = u.cumsum() / u.sum()
    top = prof.iloc[0]
    summary = {**notes, "customers": n, "k": k, "silhouette_by_k": scores,
               "share_units_top10pct_customers": float(cum[int(n * 0.10) - 1]),
               "share_units_top20pct_customers": float(cum[int(n * 0.20) - 1]),
               "share_one_purchase_day": float((t.purchase_days == 1).mean()),
               "median_purchase_days": float(t.purchase_days.median()),
               "markdown_share_all_known_lines": float(c.markdown.mean()),
               "short_lived_share_all_known_units": float((c.qty * c.short_lived.fillna(False))[c.short_lived.notna()].sum()
                                                          / c.qty[c.short_lived.notna()].sum()),
               "spearman_frequency_vs_short_lived_share": {"rho": float(rho_sl), "p": float(p_sl)},
               "spearman_frequency_vs_markdown_share": {"rho": float(rho_md), "p": float(p_md)},
               "spearman_frequency_vs_high_leftover_share": {"rho": float(rho_hl), "p": float(p_hl)},
               "top_cltv_segment": top.segment, "top_cltv_segment_share_customers": float(top.share_of_customers),
               "top_cltv_segment_share_units": float(top.share_of_units), "top_cltv_segment_share_spend": float(top.share_of_spend),
               "markdown_share_lowest_band": float(fb.markdown_share.iloc[0]), "markdown_share_highest_band": float(fb.markdown_share.iloc[-1]),
               "short_lived_share_lowest_band": float(fb.short_lived_share.iloc[0]),
               "short_lived_share_highest_band": float(fb.short_lived_share.iloc[-1])}
    save_json(summary, "vis_customers_summary")
    print(prof.round(3).to_string())
    print(fb.round(3).to_string())
    print({k_: v for k_, v in summary.items() if not isinstance(v, dict)})

    src = "outputs/tables/vis_customers_summary.json"
    rc = register_claim
    rc("cust_n", "Loyalty-card customers analysed (Dec 2016 - Feb 2020)", n, src)
    rc("cust_cards_excluded", "Card IDs excluded (bought in more than 10 shops: shop or generic cards)", notes["card_ids_excluded"], src)
    rc("cust_k", "Customer segments (k-means on RFM features)", k, src)
    rc("cust_top10_units", "Share of units bought by the top 10% of customers", summary["share_units_top10pct_customers"], src)
    rc("cust_top20_units", "Share of units bought by the top 20% of customers", summary["share_units_top20pct_customers"], src)
    rc("cust_one_day", "Share of customers with a single purchase day", summary["share_one_purchase_day"], src)
    rc("cust_md_share", "Share of purchases (with known discount) made on markdown", summary["markdown_share_all_known_lines"], src)
    rc("cust_rho_md", "Spearman: purchase frequency vs markdown share", float(rho_md), src)
    rc("cust_rho_sl", "Spearman: purchase frequency vs short-lived share", float(rho_sl), src)
    rc("cust_md_lo_hi", "Markdown share: once a year or less vs monthly or more",
       f"{summary['markdown_share_lowest_band']:.3f} vs {summary['markdown_share_highest_band']:.3f}", src)
    rc("cust_sl_lo_hi", "Short-lived share: once a year or less vs monthly or more",
       f"{summary['short_lived_share_lowest_band']:.3f} vs {summary['short_lived_share_highest_band']:.3f}", src)
    rc("cust_top_segment", "Highest-CLTV segment: share of customers / units / spend",
       f"{top.segment}: {top.share_of_customers:.3f} / {top.share_of_units:.3f} / {top.share_of_spend:.3f}", src)
    plots(prof, fb)


def plots(prof, fb):
    p = prof.sort_values("share_of_units", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
    y = np.arange(len(p))
    axes[0].barh(y - 0.2, p.share_of_customers * 100, height=0.4, color=PALETTE["sand"], label="% of customers")
    axes[0].barh(y + 0.2, p.share_of_units * 100, height=0.4, color=PALETTE["accent"], label="% of units bought")
    for i, v in enumerate(p.share_of_units):
        axes[0].text(v * 100 + 0.5, i + 0.2, f"{v:.0%}", va="center", fontsize=8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(p.segment, fontsize=9)
    axes[0].set_xlabel("%")
    axes[0].set_title("Customer segments: share of customers vs share of purchases")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[1].barh(y, p.cltv_index, color=PALETTE["teal"])
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("CLTV (index units: prices are normalised)")
    axes[1].set_title("Customer lifetime value")
    savefig(fig, "vis_customer_segments")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(fb))
    ax.bar(x - 0.2, fb.markdown_share * 100, width=0.4, color=PALETTE["teal"], label="Bought on markdown")
    ax.bar(x + 0.2, fb.short_lived_share * 100, width=0.4, color=PALETTE["accent"], label="Short-lived products")
    for i, (a, b) in enumerate(zip(fb.markdown_share, fb.short_lived_share)):
        ax.text(i - 0.2, a * 100 + 0.6, f"{a:.0%}", ha="center", fontsize=8)
        ax.text(i + 0.2, b * 100 + 0.6, f"{b:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(fb.freq_band, fontsize=9)
    ax.set_ylabel("% of the customer's purchases (mean)")
    ax.set_title("Frequent customers buy more on markdown")
    ax.legend(frameon=False, fontsize=8.5)
    savefig(fig, "vis_customers_by_frequency")


if __name__ == "__main__":
    main()
