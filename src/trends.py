"""Market trends: weekly demand trendline and seasonality from Nuna Lie's loyalty-card purchases, and whether
market-interest data (Google Trends) improves VISUELLE sales forecasts before launch and after two weeks.

Coverage: the dataset holds only SS17-AW19 products, so weekly purchases are analysed from June 2017 (once the
previous season's products are in the data) to December 2019 (before SS20 products, which are not in the data,
and before the COVID shop closures). Growth in the series also reflects more customers joining the loyalty card.
"""
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from . import vis_forecast as vf
from .config import CUSTOMER_MAX_SHOPS, RANDOM_STATE, TAB, VIS_RAW
from .utils import PALETTE, plt, register_claim, save_json, save_table, savefig

SCRIPT = "src/trends.py"
START, END = pd.Timestamp("2017-06-01"), pd.Timestamp("2020-01-01")


def vis_demand():
    c = pd.read_csv(VIS_RAW / "customer_data.csv", parse_dates=["data"])
    shops = c.groupby("customer").retail.nunique()
    c = c[~c.customer.isin(shops[shops > CUSTOMER_MAX_SHOPS].index)]
    c = c[(c.data >= START) & (c.data < END)]
    w = c.set_index("data").qty.resample("W-MON", label="left", closed="left").sum().rename("units").reset_index()
    w = w.rename(columns={"data": "week"})
    w = w[(w.week >= START) & (w.week + pd.Timedelta(days=6) < END)].reset_index(drop=True)  # full weeks only
    w["ma4"] = w.units.rolling(4, center=True).mean()
    t = (w.week - w.week.min()).dt.days / 7
    slope, intercept = np.polyfit(t, w.units, 1)
    w["trendline"] = intercept + slope * t

    def window(y):  # same calendar weeks, June-December
        m = (w.week >= pd.Timestamp(f"{y}-06-01")) & (w.week < pd.Timestamp(f"{y}-12-24"))
        return w.loc[m, "units"].sum()
    yr = w[(w.week >= pd.Timestamp("2018-01-01")) & (w.week < pd.Timestamp("2019-12-24"))].copy()
    yr["month"] = yr.week.dt.month
    idx = (yr.groupby("month").units.mean() / yr.units.mean()).rename("seasonal_index")
    res = {
        "window": f"{w.week.min():%d %b %Y} - {w.week.max():%d %b %Y}", "weeks": int(len(w)),
        "trend_slope_units_per_week": float(slope), "trend_pct_per_year_of_mean": float(slope * 52 / w.units.mean()),
        "yoy_growth_jun_dec": float(window(2019) / window(2018) - 1),
        "seasonal_peak_month": int(idx.idxmax()), "seasonal_peak_index": float(idx.max()),
        "seasonal_trough_month": int(idx.idxmin()), "seasonal_trough_index": float(idx.min()),
    }
    save_table(w[["week", "units", "ma4", "trendline"]], "vis_weekly_demand")
    save_table(idx.reset_index(), "vis_seasonal_index")
    return w, idx, res


def gt_value():
    """Does market interest (Google Trends) help? (a) before launch: total 12-week sales per product;
    (b) after two weeks: from the week-2 forecasting comparison in vis_forecast."""
    meta, S, D, Dsc, _ = vf.load()
    prod, Sp, Dp, dscp, idx, codes = vf.product_table(meta, S, D, Dsc)
    prod["initial_stock"] = Dp[:, 0]
    prod["log_initial_stock"] = np.log1p(prod.initial_stock)
    prod["y"] = Sp.sum(1)
    for c in ("category", "color", "fabric"):
        prod[c] = prod[c].astype("category")
    gt = [c for c in prod if c.startswith("gt_")]
    base = ["price", "n_shops", "release_month", "log_initial_stock"]
    tr, te = prod[prod.season.isin(vf.TRAIN)], prod[prod.season.isin(vf.TEST)]
    out = {}
    for name, feats in (("with Google Trends", base + gt), ("without Google Trends", base)):
        m = HistGradientBoostingRegressor(loss="poisson", max_iter=500, learning_rate=0.05,
                                          categorical_features="from_dtype", random_state=RANDOM_STATE)
        m.fit(tr[feats + ["category", "color", "fabric"]], tr.y)
        f = m.predict(te[feats + ["category", "color", "fabric"]])
        out[name] = {"wape": float(np.abs(f - te.y).sum() / te.y.sum()),
                     "spearman": float(spearmanr(f, te.y)[0])}
    fm = pd.read_csv(TAB / "vis_forecast_metrics.csv")
    fm = fm[fm.group == "All products"].set_index("model")
    week2 = {"with Google Trends": float(fm.loc["Gradient boosting + Google Trends", "weekly_wape"]),
             "without Google Trends": float(fm.loc["Gradient boosting (no Google Trends)", "weekly_wape"])}
    return {"prelaunch_total_sales": out, "week2_weekly_wape": week2}


def main():
    w, idx, res = vis_demand()
    gv = gt_value()
    res["google_trends"] = gv
    save_json(res, "trends_summary")
    print(res)

    src = "outputs/tables/trends_summary.json"
    register_claim("trend_yoy", "Loyalty-card units, Jun-Dec 2019 vs same weeks 2018", res["yoy_growth_jun_dec"], src)
    register_claim("trend_peak_month", "Seasonal peak month (index, 2018-2019)", f"{res['seasonal_peak_month']} ({res['seasonal_peak_index']:.2f})", src)
    register_claim("trend_trough_month", "Seasonal trough month (index, 2018-2019)", f"{res['seasonal_trough_month']} ({res['seasonal_trough_index']:.2f})", src)
    pl_ = gv["prelaunch_total_sales"]
    register_claim("gt_prelaunch_with", "Pre-launch 12-week sales forecast WAPE, with Google Trends", pl_["with Google Trends"]["wape"], src)
    register_claim("gt_prelaunch_without", "Pre-launch 12-week sales forecast WAPE, without Google Trends", pl_["without Google Trends"]["wape"], src)
    register_claim("gt_week2_with", "Week-2 weekly WAPE, with Google Trends", gv["week2_weekly_wape"]["with Google Trends"], src)
    register_claim("gt_week2_without", "Week-2 weekly WAPE, without Google Trends", gv["week2_weekly_wape"]["without Google Trends"], src)

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    ax.plot(w.week, w.units / 1000, color=PALETTE["sand"], lw=1.2, label="Weekly units")
    ax.plot(w.week, w.ma4 / 1000, color=PALETTE["accent"], lw=2.2, label="4-week moving average")
    ax.plot(w.week, w.trendline / 1000, color=PALETTE["ink"], ls="--", lw=1.4, label="Linear trendline")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("Units bought per week (thousands)")
    ax.set_title("Nuna Lie loyalty-card purchases: trend and seasonality")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax2 = axes[1]
    ax2.bar(months, [idx[m] * 100 for m in range(1, 13)],
            color=[PALETTE["accent"] if idx[m] >= 1 else PALETTE["teal"] for m in range(1, 13)])
    ax2.axhline(100, color=PALETTE["ink"], lw=1)
    ax2.set_ylabel("Seasonal index (100 = average week)")
    ax2.set_title("Seasonality, 2018-2019")
    ax2.tick_params(axis="x", labelsize=8, rotation=45)
    savefig(fig, "vis_demand_trend")


if __name__ == "__main__":
    main()
