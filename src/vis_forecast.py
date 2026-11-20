"""Phase 6 — VISUELLE 2.0: stock, forecasting after 2 weeks, unsold-stock prediction,
markdown timing, markdown uplift, and the what-if policy simulation.

Unit of analysis: product x shop (106,850 rows) for stock and simulation; product (5,355)
for forecasting. Seasons 2017-2018 train, 2019 (SS19, AW19) test — never seen in training.
Deliveries: restocks.csv, placed on a week index relative to release (pre-release lumped
into week 0; deliveries after week 12 are outside the observed selling window).
Sales: negative weekly values (returns, 0.3% of cells) are clipped to 0 as demand.
"""
from datetime import date

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

from .config import K_WEEKS, MARKDOWN_THRESHOLD, RANDOM_STATE, TEST_SEASONS, TRAIN_SEASONS, VIS_RAW
from .env_factors import CO2E_PER_KG, GARMENT_KG, units_to_impact
from .utils import PALETTE, SERIES, plt, register_claim, save_json, save_table, savefig

SCRIPT = "src/vis_forecast.py"
K = K_WEEKS
ORIGIN = 2                      # weeks observed before forecasting
TRAIN = TRAIN_SEASONS
TEST = TEST_SEASONS
MD_LEVEL = MARKDOWN_THRESHOLD   # product counted as marked down when mean discount >= 15%


# ----------------------------------------------------------------------------- data
def load():
    s = pl.read_csv(VIS_RAW / "sales.csv").drop("").with_columns(pl.col("release_date").str.to_date())
    p = pl.read_csv(VIS_RAW / "price_discount_series.csv").rename({str(k): f"d{k}" for k in range(K)})
    s = s.join(p, on=["external_code", "retail"], how="left")
    S_raw = s.select([str(k) for k in range(K)]).to_numpy()
    n_returns = int((S_raw < 0).sum())
    S = np.nan_to_num(S_raw).clip(0)
    Dsc = np.nan_to_num(s.select([f"d{k}" for k in range(K)]).to_numpy())

    r = pl.read_csv(VIS_RAW / "restocks.csv").with_columns(
        pl.struct(["year", "week"]).map_elements(lambda x: date.fromisocalendar(x["year"], x["week"], 1),
                                                 return_dtype=pl.Date).alias("rdate"))
    key = s.select("external_code", "retail").with_row_index("i")
    r = (r.join(s.select("external_code", "retail", "release_date"), on=["external_code", "retail"])
          .with_columns(((pl.col("rdate") - pl.col("release_date")).dt.total_days() // 7).alias("k"))
          .join(key, on=["external_code", "retail"]))
    D = np.zeros((s.height, K))
    rr = r.filter(pl.col("k") < K).with_columns(pl.col("k").clip(0, None)).select("i", "k", "qty").to_numpy()
    np.add.at(D, (rr[:, 0], rr[:, 1]), rr[:, 2])
    late_deliveries = int(r.filter(pl.col("k") >= K)["qty"].sum())
    meta = s.select("external_code", "retail", "season", "category", "color", "fabric", "release_date", "price")
    return meta.to_pandas(), S, D, Dsc, {"negative_sales_cells_clipped": n_returns,
                                          "units_delivered_after_week12_excluded": late_deliveries}


def stock_path(D, S):
    """Start-of-week available stock and end-of-week stock for each product-shop."""
    cumD = D.cumsum(1)
    sold_before = np.concatenate([np.zeros((len(S), 1)), S.cumsum(1)[:, :-1]], 1)
    avail = cumD - sold_before
    return avail, avail - S


def short_lived_flag(U):
    """Short-lived product (rule fixed before modelling): weekly sales fall below 25% of the peak
    within 4 weeks of the peak, and the last 6 of the 12 weeks carry < 20% of 12-week units."""
    peak_w, peak = U.argmax(1), U.max(1)
    tot = U.sum(1)
    within = np.zeros(len(U), bool)
    for i in range(len(U)):
        after = U[i, peak_w[i] + 1: peak_w[i] + 5]
        within[i] = (after < 0.25 * peak[i]).any()
    last6 = U[:, 6:].sum(1) / np.where(tot > 0, tot, 1)
    return within & (last6 < 0.20) & (tot > 0)


def gtrend_features(prod):
    g = pd.read_csv(VIS_RAW / "vis2_gtrends_data.csv", parse_dates=["date"]).set_index("date").sort_index()
    out = {}
    for attr in ("category", "color", "fabric"):
        lvl, slope = [], []
        for term, rel in zip(prod[attr], prod.release_date):
            win = g.loc[(g.index < pd.Timestamp(rel)) & (g.index >= pd.Timestamp(rel) - pd.Timedelta(weeks=12)), term]
            if len(win) < 8:
                lvl.append(np.nan)
                slope.append(np.nan)
                continue
            lvl.append(win.iloc[-4:].mean())
            slope.append(win.iloc[-4:].mean() - win.iloc[:4].mean())
        out[f"gt_{attr}_level"] = lvl
        out[f"gt_{attr}_slope"] = slope
    return pd.DataFrame(out, index=prod.index)


def product_table(meta, S, D, Dsc):
    df = meta.copy()
    idx = df.groupby("external_code").indices
    codes = np.array(sorted(idx))
    Sp = np.stack([S[idx[c]].sum(0) for c in codes])
    Dp = np.stack([D[idx[c]].sum(0) for c in codes])
    dscp = np.stack([Dsc[idx[c]].mean(0) for c in codes])
    first = df.groupby("external_code").first().loc[codes]
    prod = pd.DataFrame({
        "external_code": codes, "season": first.season.values, "category": first.category.values,
        "color": first.color.values, "fabric": first.fabric.values,
        "release_date": pd.to_datetime(first.release_date.values),
        "price": df.groupby("external_code").price.mean().loc[codes].values,
        "n_shops": df.groupby("external_code").size().loc[codes].values,
    })
    prod["release_month"] = prod.release_date.dt.month
    prod["stock_by_wk2"] = Dp[:, :ORIGIN].sum(1)
    prod["delivered_12w"] = Dp.sum(1)
    prod["sold_12w"] = Sp.sum(1)
    prod["leftover_12w"] = (prod.delivered_12w - prod.sold_12w).clip(lower=0)
    prod["sell_through"] = np.where(prod.delivered_12w > 0, prod.sold_12w / prod.delivered_12w, np.nan)
    prod["short_lived"] = short_lived_flag(Sp)
    prod["disc_wk01"] = dscp[:, :ORIGIN].mean(1)
    md = dscp >= MD_LEVEL
    prod["first_markdown_week"] = np.where(md.any(1), md.argmax(1), np.nan)
    prod = pd.concat([prod, gtrend_features(prod)], axis=1)
    return prod, Sp, Dp, dscp, idx, codes


# ----------------------------------------------------------------------------- forecasting
def horizon_rows(prod, Sp):
    """Long format: one row per product x horizon week (weeks 3..12 = index 2..11)."""
    rows = []
    for h in range(ORIGIN, K):
        d = prod[["external_code", "season", "category", "color", "fabric", "price", "n_shops", "release_month",
                  "stock_by_wk2", "disc_wk01", "short_lived"] + [c for c in prod if c.startswith("gt_")]].copy()
        d["h"] = h
        d["w1"], d["w2"] = Sp[:, 0], Sp[:, 1]
        d["y"] = Sp[:, h]
        rows.append(d)
    X = pd.concat(rows, ignore_index=True)
    X["log_w1"], X["log_w2"] = np.log1p(X.w1), np.log1p(X.w2)
    X["log_w2_vs_w1"] = np.log1p(X.w2) - np.log1p(X.w1)
    X["log_stock_wk2"] = np.log1p(X.stock_by_wk2)
    X["log_w12_per_shop"] = np.log1p((X.w1 + X.w2) / X.n_shops)
    for c in ("category", "color", "fabric"):
        X[c] = X[c].astype("category")
    return X


NUMF = ["h", "log_w1", "log_w2", "log_w2_vs_w1", "log_stock_wk2", "log_w12_per_shop", "price", "n_shops",
        "release_month", "disc_wk01", "gt_category_level", "gt_category_slope", "gt_color_level",
        "gt_color_slope", "gt_fabric_level", "gt_fabric_slope"]
CATF = ["category", "color", "fabric"]


def fit_forecasters(X):
    tr, te = X[X.season.isin(TRAIN)].copy(), X[X.season.isin(TEST)].copy()
    preds = {}
    # 1. Naive: week-2 sales carried forward
    preds["Naive (last week repeats)"] = te.w2.values.astype(float)
    # 2. Category lifecycle profile: ratio of sums, week h vs weeks 1-2, estimated on training seasons
    base = tr.w1 + tr.w2
    prof = (tr.assign(base=base).groupby(["category", "h"], observed=True)
              .apply(lambda g: g.y.sum() / max(g.base.sum(), 1), include_groups=False))
    glob = tr.assign(base=base).groupby("h").apply(lambda g: g.y.sum() / g.base.sum(), include_groups=False)
    ratio = [prof.get((c, h), glob[h]) for c, h in zip(te.category, te.h)]
    preds["Category lifecycle profile"] = (te.w1 + te.w2).values * np.array(ratio)
    # 3. Poisson regression (a linear model on the log scale, suited to counts), one-hot categories
    Xl_tr = pd.get_dummies(tr[NUMF + CATF], columns=CATF + ["h"], dtype=float)
    Xl_te = pd.get_dummies(te[NUMF + CATF], columns=CATF + ["h"], dtype=float).reindex(columns=Xl_tr.columns, fill_value=0)
    med = Xl_tr.median()
    sc = StandardScaler().fit(Xl_tr.fillna(med))
    pois = PoissonRegressor(alpha=1e-3, max_iter=2000).fit(sc.transform(Xl_tr.fillna(med)), tr.y)
    preds["Poisson regression"] = pois.predict(sc.transform(Xl_te.fillna(med)))
    # 4. Gradient boosting with Poisson loss (native categorical support)
    gbm = HistGradientBoostingRegressor(loss="poisson", max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                                        categorical_features="from_dtype",
                                        random_state=RANDOM_STATE)
    gbm.fit(tr[NUMF + CATF], tr.y)
    preds["Gradient boosting + Google Trends"] = gbm.predict(te[NUMF + CATF])
    # 5. Ablation: identical model without the Google Trends (market-interest) features
    no_gt = [f for f in NUMF if not f.startswith("gt_")]
    gbm_ngt = HistGradientBoostingRegressor(loss="poisson", max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                                            categorical_features="from_dtype", random_state=RANDOM_STATE)
    gbm_ngt.fit(tr[no_gt + CATF], tr.y)
    preds["Gradient boosting (no Google Trends)"] = gbm_ngt.predict(te[no_gt + CATF])
    return tr, te, preds, gbm


def forecast_metrics(te, preds):
    rows = []
    y = te.y.values
    for name, f in preds.items():
        e = f - y
        tot = te.assign(f=f).groupby("external_code").agg(f=("f", "sum"), y=("y", "sum"))
        for grp, m in (("All products", np.ones(len(te), bool)), ("Short-lived", te.short_lived.values),
                       ("Other", ~te.short_lived.values)):
            rows.append({"model": name, "group": grp,
                         "weekly_wape": np.abs(e[m]).sum() / y[m].sum(),
                         "weekly_mae": np.abs(e[m]).mean(),
                         "bias": e[m].sum() / y[m].sum()})
        rows[-3]["remaining_total_wape"] = (tot.f - tot.y).abs().sum() / tot.y.sum()
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- simulation
def simulate(D, S, Dsc, price, flag_rows, r_cut=0.0, md_level=None, md_week=3, beta=0.882):
    """Replay one season at product-shop level under a policy.
    Demand = observed sales (lower bound; stock-outs are rare) scaled by the markdown uplift.
    Policy A: for flagged product-shops, deliveries after the 2 observed weeks are cut by r_cut.
    Policy B: for flagged product-shops, discount raised to at least md_level from week md_week+1.
    """
    D2 = D.copy()
    if r_cut:
        D2[flag_rows, ORIGIN:] *= (1 - r_cut)
    dsc2 = Dsc.copy()
    if md_level is not None:
        sub = dsc2[flag_rows, md_week:]
        dsc2[flag_rows, md_week:] = np.maximum(sub, md_level)
    demand = S * np.exp(beta * (dsc2 - Dsc))
    stock = np.zeros(len(S))
    sold = np.zeros_like(S)
    for k in range(K):
        stock += D2[:, k]
        sold[:, k] = np.minimum(demand[:, k], np.maximum(stock, 0))
        stock -= sold[:, k]
    revenue = (price[:, None] * (1 - dsc2) * sold).sum()
    return {"delivered": D2.sum(), "sold": sold.sum(), "leftover": np.maximum(stock, 0).sum(), "revenue": revenue,
            "purchase_value_full_price": (price[:, None] * D2).sum()}


def run_simulations(meta, S, D, Dsc, prod_te_pred, betas):
    test_rows = meta.season.isin(TEST).values
    Dt, St, Dsct = D[test_rows], S[test_rows], Dsc[test_rows]
    price = meta.price.values[test_rows]
    codes_t = meta.external_code.values[test_rows]
    base = simulate(Dt, St, Dsct, price, np.zeros(len(St), bool))
    rows = []
    for tau in (0.6, 0.7):
        flagged_codes = set(prod_te_pred.loc[prod_te_pred.pred_sell_through < tau, "external_code"])
        flag = np.isin(codes_t, list(flagged_codes))
        for beta_name, beta in betas.items():
            for policy in ("A: cut later deliveries", "B: earlier markdown", "A+B"):
                for r in (0.1, 0.2, 0.3, 0.4):
                    for mdl in (0.2, 0.3):
                        if policy == "A: cut later deliveries" and mdl != 0.2:
                            continue
                        if policy == "B: earlier markdown" and r != 0.1:
                            continue
                        kw = {}
                        if policy.startswith("A"):
                            kw["r_cut"] = r
                        if "B" in policy:
                            kw["md_level"] = mdl
                        res = simulate(Dt, St, Dsct, price, flag, beta=beta, **kw)
                        rows.append({
                            "flag_threshold_pred_sell_through": tau, "share_products_flagged": len(flagged_codes) / prod_te_pred.shape[0],
                            "uplift_estimate": beta_name, "policy": policy,
                            "delivery_cut": r if policy.startswith("A") else 0.0,
                            "markdown_level": mdl if "B" in policy else 0.0,
                            "leftover_units": res["leftover"], "leftover_change": res["leftover"] / base["leftover"] - 1,
                            "sold_change": res["sold"] / base["sold"] - 1,
                            "revenue_change": res["revenue"] / base["revenue"] - 1,
                            "delivered_change": res["delivered"] / base["delivered"] - 1,
                            "units_leftover_avoided": base["leftover"] - res["leftover"],
                            # Gross profit rises if unit cost / full price exceeds this ratio
                            # (revenue given up per unit of full-price value not bought).
                            "breakeven_cost_to_price": max(base["revenue"] - res["revenue"], 0) / max(
                                base["purchase_value_full_price"] - res["purchase_value_full_price"], 1e-9)
                            if res["purchase_value_full_price"] < base["purchase_value_full_price"] else np.nan,
                        })
    return base, pd.DataFrame(rows).drop_duplicates(
        subset=["flag_threshold_pred_sell_through", "uplift_estimate", "policy", "delivery_cut", "markdown_level"])


def estimate_uplift(S, Dsc):
    """Markdown uplift: log(1+sales) on discount with product-shop and week fixed effects
    (two-way demeaning of the balanced 12-week panel), plus a first-difference check."""
    y, x = np.log1p(S), Dsc
    dm = lambda a: a - a.mean(1, keepdims=True) - a.mean(0, keepdims=True) + a.mean()
    yd, xd = dm(y), dm(x)
    beta = (xd * yd).sum() / (xd * xd).sum()
    g = (xd * (yd - beta * xd)).sum(1)
    se = np.sqrt((g ** 2).sum()) / (xd * xd).sum()
    dy, dx = np.diff(y, axis=1), np.diff(x, axis=1)
    m = dx != 0
    b_fd = (dx[m] * dy[m]).sum() / (dx[m] ** 2).sum()
    return {"beta_fe": float(beta), "se_fe_cluster": float(se), "beta_fd": float(b_fd),
            "uplift_per_10pp_fe": float(np.exp(beta * 0.1) - 1), "uplift_per_10pp_fd": float(np.exp(b_fd * 0.1) - 1)}


# ----------------------------------------------------------------------------- main
def main():
    meta, S, D, Dsc, notes = load()
    avail, end = stock_path(D, S)
    stockout = end <= 0
    prod, Sp, Dp, dscp, idx, codes = product_table(meta, S, D, Dsc)

    # Stock facts
    stock = {
        **notes,
        "product_shops": int(len(S)), "products": int(len(prod)), "shops": int(meta.retail.nunique()),
        "seasons": sorted(meta.season.unique().tolist()),
        "share_shop_weeks_ending_at_zero_stock": float(stockout.mean()),
        "share_product_shops_ever_stocked_out_12w": float(stockout.any(1).mean()),
        "units_delivered_12w": float(D.sum()), "units_sold_12w": float(S.sum()),
        "leftover_units_12w_product_shop": float(np.maximum(end[:, -1], 0).sum()),
        "leftover_share_of_delivered": float(np.maximum(end[:, -1], 0).sum() / D.sum()),
        "median_product_sell_through": float(prod.sell_through.median()),
        "share_products_sell_through_below_60pct": float((prod.sell_through < 0.6).mean()),
        "share_products_short_lived": float(prod.short_lived.mean()),
    }
    # Did short-lived products fade with stock still on the shelf?
    sl_codes = set(prod.loc[prod.short_lived, "external_code"])
    sl_rows = meta.external_code.isin(sl_codes).values
    post_peak = np.zeros_like(stockout)
    pk = Sp.argmax(1)
    code_pos = {c: i for i, c in enumerate(codes)}
    pk_rows = np.array([pk[code_pos[c]] for c in meta.external_code])
    for k in range(K):
        post_peak[:, k] = k > pk_rows
    stock["short_lived_post_peak_shop_weeks_with_stock_available"] = float(
        1 - stockout[sl_rows][post_peak[sl_rows]].mean())
    stock["short_lived_leftover_share"] = float(
        prod.loc[prod.short_lived, "leftover_12w"].sum() / prod.loc[prod.short_lived, "delivered_12w"].sum())
    stock["other_leftover_share"] = float(
        prod.loc[~prod.short_lived, "leftover_12w"].sum() / prod.loc[~prod.short_lived, "delivered_12w"].sum())
    by_season = (prod.groupby("season").agg(products=("external_code", "size"), delivered=("delivered_12w", "sum"),
                                            sold=("sold_12w", "sum"), leftover=("leftover_12w", "sum"),
                                            median_sell_through=("sell_through", "median"))
                 .assign(leftover_share=lambda d: d.leftover / d.delivered).reset_index())
    save_table(by_season, "vis_leftover_by_season")

    # Published early-warning metric (Zhang, Wang, Lin & Hult 2025, J. Retailing): time-to-peak-sales vs overstock
    from scipy.stats import spearmanr
    ok = prod.dropna(subset=["sell_through"])
    rho, pv = spearmanr(Sp[ok.index].argmax(1) + 1, 1 - ok.sell_through)
    stock["spearman_time_to_peak_vs_leftover_share"] = float(rho)
    stock["spearman_time_to_peak_p_value"] = float(pv)

    # Markdown timing vs demand (product level)
    has_md = ~np.isnan(prod.first_markdown_week.values)
    fm = prod.first_markdown_week.values[has_md].astype(int)
    Spm = Sp[has_md]
    dem_at_md = Spm[np.arange(len(Spm)), fm] / np.maximum(Spm.max(1), 1)
    slm = prod.short_lived.values[has_md]
    stock["share_products_marked_down_15pct_12w"] = float(has_md.mean())
    stock["median_first_markdown_week_1based"] = float(np.median(fm) + 1)
    stock["short_lived_marked_down_after_demand_halved"] = float((dem_at_md[slm] < 0.5).mean())
    stock["other_marked_down_after_demand_halved"] = float((dem_at_md[~slm] < 0.5).mean())

    # Uplift
    upl = estimate_uplift(S, Dsc)

    # Forecasting
    X = horizon_rows(prod, Sp)
    tr, te, preds, gbm = fit_forecasters(X)
    fm_tab = forecast_metrics(te, preds)
    save_table(fm_tab, "vis_forecast_metrics")
    print(fm_tab.round(3).to_string())
    allp = fm_tab[fm_tab.group == "All products"].set_index("model")
    best = allp.weekly_wape.idxmin()
    naive_wape = allp.loc["Naive (last week repeats)", "weekly_wape"]

    # Unsold-stock prediction with the best model (given each product's actual delivery plan)
    te_best = te.assign(f=preds[best]).groupby("external_code").f.sum()
    pt = prod.set_index("external_code").loc[te_best.index]
    pred_sold = Sp[[code_pos[c] for c in te_best.index]][:, :ORIGIN].sum(1) + te_best.values
    pred = pd.DataFrame({"external_code": te_best.index, "pred_sold_12w": pred_sold,
                         "delivered_12w": pt.delivered_12w.values, "sold_12w": pt.sold_12w.values,
                         "leftover_12w": pt.leftover_12w.values, "category": pt.category.values,
                         "season": pt.season.values, "short_lived": pt.short_lived.values})
    pred["pred_leftover"] = (pred.delivered_12w - pred.pred_sold_12w).clip(lower=0)
    pred["pred_sell_through"] = pred.pred_sold_12w / pred.delivered_12w.replace(0, np.nan)
    pred["sell_through"] = pred.sold_12w / pred.delivered_12w.replace(0, np.nan)
    hi_true, hi_pred = pred.sell_through < 0.6, pred.pred_sell_through < 0.6
    leftover_eval = {
        "model": best, "test_products": int(len(pred)),
        "leftover_wape": float((pred.pred_leftover - pred.leftover_12w).abs().sum() / pred.leftover_12w.sum()),
        "leftover_total_actual": float(pred.leftover_12w.sum()), "leftover_total_predicted": float(pred.pred_leftover.sum()),
        "corr_pred_vs_actual_leftover": float(np.corrcoef(pred.pred_leftover, pred.leftover_12w)[0, 1]),
        "high_leftover_products_actual_share": float(hi_true.mean()),
        "high_leftover_precision": float((hi_true & hi_pred).sum() / max(hi_pred.sum(), 1)),
        "high_leftover_recall": float((hi_true & hi_pred).sum() / max(hi_true.sum(), 1)),
        "flagged_share_of_actual_leftover_units": float(pred.loc[hi_pred, "leftover_12w"].sum() / pred.leftover_12w.sum()),
        # Leftover UNITS scale with deliveries, so their correlation is not a test of the forecast:
        # a plan-only baseline (deliveries x average leftover share) scores higher. Judge the forecast
        # on sell-through (sold / delivered) and on the flag, against the other models.
        "corr_pred_vs_actual_sell_through": float(pred.pred_sell_through.corr(pred.sell_through)),
    }
    flag_rows = []
    trp = prod[prod.season.isin(TRAIN)]
    avg_left = 1 - trp.sold_12w.sum() / trp.delivered_12w.sum()
    for name, f in preds.items():
        tot = te.assign(f=f).groupby("external_code").f.sum()
        q = prod.set_index("external_code").loc[tot.index]
        ps = (Sp[[code_pos[c] for c in tot.index]][:, :ORIGIN].sum(1) + tot.values) / q.delivered_12w.replace(0, np.nan).values
        ht, hp = q.sell_through.values < 0.6, ps < 0.6
        flag_rows.append({"model": name,
                          "corr_sell_through": float(pd.Series(ps).corr(pd.Series(q.sell_through.values))),
                          "corr_leftover_units": float(pd.Series(np.clip(q.delivered_12w.values * (1 - ps), 0, None))
                                                       .corr(pd.Series(q.leftover_12w.values))),
                          "flag_recall": float((ht & hp).sum() / ht.sum()),
                          "flag_precision": float((ht & hp).sum() / max(hp.sum(), 1)),
                          "share_flagged": float(hp.mean())})
    flag_rows.append({"model": "Plan only (deliveries x average leftover share)", "corr_sell_through": np.nan,
                      "corr_leftover_units": float(np.corrcoef(pred.delivered_12w * avg_left, pred.leftover_12w)[0, 1]),
                      "flag_recall": np.nan, "flag_precision": np.nan, "share_flagged": np.nan})
    flag_tab = pd.DataFrame(flag_rows)
    save_table(flag_tab, "vis_leftover_flag_by_model")
    naive_flag = flag_tab[flag_tab.model == "Naive (last week repeats)"].iloc[0]
    leftover_eval.update({"naive_corr_sell_through": float(naive_flag.corr_sell_through),
                          "naive_flag_recall": float(naive_flag.flag_recall),
                          "naive_flag_precision": float(naive_flag.flag_precision),
                          "plan_only_corr_leftover_units": float(flag_tab.iloc[-1].corr_leftover_units)})

    # Markdown trigger from the week-2 forecast: first week forecast < 50% of peak-so-far
    trig = []
    fpath = te.assign(f=preds[best]).pivot_table(index="external_code", columns="h", values="f")
    for c, row in fpath.iterrows():
        obs = Sp[code_pos[c], :ORIGIN]
        path = np.concatenate([obs, row.values])
        peak = np.maximum.accumulate(path)
        below = np.nonzero((path < 0.5 * peak) & (np.arange(K) >= ORIGIN))[0]
        trig.append(below[0] + 1 if len(below) else np.nan)
    fpath["recommended_markdown_week"] = trig
    act = prod.set_index("external_code").loc[fpath.index, "first_markdown_week"] + 1
    pred = pred.merge(fpath[["recommended_markdown_week"]].reset_index(), on="external_code", how="left")
    pred["actual_first_markdown_week"] = pred.external_code.map(prod.set_index("external_code").first_markdown_week + 1)
    save_table(pred, "vis_test_leftover_predictions")
    # Weekly path per test product (weeks 3-12), for the demo app's forecast view
    weekly = te[["external_code", "h", "y"]].assign(forecast=preds[best], naive=preds["Naive (last week repeats)"])
    save_table(weekly.assign(week=weekly.h + 1).drop(columns="h").sort_values(["external_code", "week"]),
               "vis_test_weekly_forecast")
    trig_eval ={"median_recommended_week": float(np.nanmedian(trig)), "median_actual_first_markdown_week": float(act.median()),
                 "share_recommended_earlier_than_actual": float((fpath.recommended_markdown_week < act).mean())}

    # Simulation
    betas = {"FE (conservative)": upl["beta_fe"], "First-difference": upl["beta_fd"]}
    base, sim = run_simulations(meta, S, D, Dsc, pred, betas)
    save_table(sim, "vis_simulation_grid")
    main_row = sim[(sim.flag_threshold_pred_sell_through == 0.6) & (sim.uplift_estimate == "FE (conservative)")
                   & (sim.policy == "A+B") & (sim.delivery_cut == 0.3) & (sim.markdown_level == 0.2)].iloc[0]
    impact = units_to_impact(float(main_row.units_leftover_avoided))
    test_seasons_units = float(base["leftover"])

    summary = {"stock": stock, "uplift": upl, "forecast_best_model": best,
               "forecast_best_weekly_wape": float(allp.loc[best, "weekly_wape"]),
               "forecast_naive_weekly_wape": float(naive_wape),
               "forecast_best_remaining_total_wape": float(allp.loc[best, "remaining_total_wape"]),
               "forecast_best_bias": float(allp.loc[best, "bias"]),
               "leftover_prediction": leftover_eval, "markdown_trigger": trig_eval,
               "simulation_baseline_test_leftover_units": test_seasons_units,
               "simulation_main_scenario": main_row.to_dict(), "simulation_main_impact": impact,
               "env_factors": {"garment_kg": GARMENT_KG, "co2e_per_kg": CO2E_PER_KG}}
    save_json(summary, "vis_summary")
    for k, v in summary.items():
        print(k, ":", v)

    src = "outputs/tables/vis_summary.json"
    register_claim("vis_ttp_rho", "VISUELLE: Spearman correlation, time-to-peak-sales vs leftover share", stock["spearman_time_to_peak_vs_leftover_share"], src)
    register_claim("vis_leftover_share", "VISUELLE: share of delivered units unsold after 12 weeks", stock["leftover_share_of_delivered"], src)
    register_claim("vis_stockout_weeks", "VISUELLE: share of shop-weeks ending at zero stock", stock["share_shop_weeks_ending_at_zero_stock"], src)
    register_claim("vis_sl_stock_avail", "VISUELLE: short-lived products' post-peak shop-weeks with stock still available",
                   stock["short_lived_post_peak_shop_weeks_with_stock_available"], src)
    register_claim("vis_sl_leftover", "VISUELLE: leftover share, short-lived products", stock["short_lived_leftover_share"], src)
    register_claim("vis_other_leftover", "VISUELLE: leftover share, other products", stock["other_leftover_share"], src)
    register_claim("vis_uplift_fe", "Sales uplift per +10pp discount (fixed effects)", upl["uplift_per_10pp_fe"], src)
    register_claim("vis_uplift_fd", "Sales uplift per +10pp discount (first differences)", upl["uplift_per_10pp_fd"], src)
    register_claim("vis_fc_best", "Best 2-week forecaster", best, src)
    register_claim("vis_fc_wape", "Best forecaster weekly WAPE (test seasons)", summary["forecast_best_weekly_wape"], src)
    register_claim("vis_fc_naive_wape", "Naive forecaster weekly WAPE (test seasons)", float(naive_wape), src)
    register_claim("vis_fc_total_wape", "Best forecaster WAPE on remaining-season total", summary["forecast_best_remaining_total_wape"], src)
    register_claim("vis_left_recall", "High-leftover products caught at week 2 (recall)", leftover_eval["high_leftover_recall"], src)
    register_claim("vis_left_precision", "Flag precision for high-leftover products", leftover_eval["high_leftover_precision"], src)
    register_claim("vis_st_corr", "Week-2 predicted vs actual sell-through correlation (best model)", leftover_eval["corr_pred_vs_actual_sell_through"], src)
    register_claim("vis_st_corr_naive", "Week-2 predicted vs actual sell-through correlation (naive)", leftover_eval["naive_corr_sell_through"], src)
    register_claim("vis_left_recall_naive", "High-leftover products caught at week 2 (naive forecast)", leftover_eval["naive_flag_recall"], src)
    register_claim("vis_left_precision_naive", "Flag precision (naive forecast)", leftover_eval["naive_flag_precision"], src)
    register_claim("vis_sl_md_after_half", "VISUELLE short-lived products marked down after demand halved",
                   stock["short_lived_marked_down_after_demand_halved"], src)
    register_claim("sim_leftover_change", "Main policy (A+B, 30% cut, 20% markdown): change in leftover units", float(main_row.leftover_change), src)
    register_claim("sim_sold_change", "Main policy: change in units sold", float(main_row.sold_change), src)
    register_claim("sim_revenue_change", "Main policy: change in revenue (index)", float(main_row.revenue_change), src)
    register_claim("sim_units_avoided", "Main policy: leftover units avoided (2019 test seasons)", float(main_row.units_leftover_avoided), src)
    register_claim("sim_tonnes", "Main policy: textile avoided, tonnes (estimate range)",
                   f"{impact['tonnes_textile_low']:.0f}-{impact['tonnes_textile_high']:.0f}", src + " + src/env_factors.py")
    register_claim("sim_co2e", "Main policy: CO2e avoided, tonnes (estimate range)",
                   f"{impact['tonnes_co2e_low']:.0f}-{impact['tonnes_co2e_high']:.0f}", src + " + src/env_factors.py")

    plots(by_season, fm_tab, te, preds, best, Sp, prod, code_pos, sim, stock, dscp)


def plots(by_season, fm_tab, te, preds, best, Sp, prod, code_pos, sim, stock, dscp):
    order = ["SS17", "AW17", "SS18", "AW18", "SS19", "AW19"]
    b = by_season.set_index("season").loc[order]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(order, b.sold / 1000, color=PALETTE["teal"], label="Sold within 12 weeks")
    ax.bar(order, b.leftover / 1000, bottom=b.sold / 1000, color=PALETTE["accent"], label="Unsold after 12 weeks")
    for i, (s_, l_) in enumerate(zip(b.sold, b.leftover)):
        ax.text(i, (s_ + l_) / 1000 + 6, f"{l_ / (s_ + l_):.0%} unsold", ha="center", fontsize=8.5)
    ax.set_ylabel("Units delivered to shops (thousands)")
    ax.set_title("About a third of delivered stock is still unsold after 12 weeks")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, (b.sold + b.leftover).max() / 1000 * 1.15)
    savefig(fig, "vis_leftover_by_season")

    a = fm_tab[fm_tab.group == "All products"].sort_values("weekly_wape")
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    cols = [PALETTE["accent"] if m == best else PALETTE["sand"] for m in a.model]
    ax.barh(a.model, a.weekly_wape * 100, color=cols)
    for i, v in enumerate(a.weekly_wape):
        ax.text(v * 100 + 0.5, i, f"{v:.0%}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Weekly forecast error, WAPE (%) — lower is better")
    ax.set_title("Forecasting weeks 3-12 from the first 2 weeks (unseen 2019 seasons)")
    savefig(fig, "vis_forecast_models")

    # Example forecasts: 3 short-lived and 3 other test products with decent volume
    rng = np.random.default_rng(3)
    tp = prod[prod.season.isin(TEST) & (prod.sold_12w > 150)]
    picks = list(rng.choice(tp[tp.short_lived].external_code, 3, replace=False)) + \
            list(rng.choice(tp[~tp.short_lived].external_code, 3, replace=False))
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=True)
    for ax, c in zip(axes.ravel(), picks):
        act = Sp[code_pos[c]]
        f = te.assign(f=preds[best])[te.external_code == c].sort_values("h").f.values
        ax.plot(range(1, K + 1), act, color=PALETTE["ink"], lw=2, label="Actual")
        ax.plot(range(ORIGIN + 1, K + 1), f, color=PALETTE["accent"], lw=2, ls="--", label="Forecast at week 2")
        ax.axvspan(0.5, ORIGIN + 0.5, color=PALETTE["grid"], alpha=0.7)
        r = prod.set_index("external_code").loc[c]
        ax.set_title(f"{r.category} ({'short-lived' if r.short_lived else 'other'})", fontsize=9.5)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Week")
    savefig(fig, "vis_forecast_examples")

    s = sim[(sim.flag_threshold_pred_sell_through == 0.6) & (sim.uplift_estimate == "FE (conservative)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)
    for pol, col in (("A: cut later deliveries", PALETTE["teal"]), ("A+B", PALETTE["accent"])):
        d = s[(s.policy == pol) & (s.markdown_level.isin([0.0, 0.2]))].sort_values("delivery_cut")
        for ax, m, lab in zip(axes, ("leftover_change", "sold_change", "revenue_change"),
                              ("Unsold stock", "Units sold", "Revenue")):
            ax.plot(d.delivery_cut * 100, d[m] * 100, marker="o", color=col, lw=2, label=pol if pol != "A+B" else "A+B (with 20% markdown)")
            ax.set_title(f"Change in {lab.lower()} (%)")
            ax.axhline(0, color=PALETTE["muted"], lw=0.8)
    for ax in axes:
        ax.set_xticks([10, 20, 30, 40])
    fig.supxlabel("Cut to later deliveries for flagged items (%)", fontsize=10, y=-0.02)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("What-if on unseen 2019 seasons: flag at week 2, then order less and mark down earlier",
                 fontweight="bold", fontsize=11, y=1.03)
    savefig(fig, "vis_simulation")


if __name__ == "__main__":
    main()
