"""Early-warning dashboard (Module 1: Automation of Marketing Models) — VISUELLE 2.0 (Nuna Lie) + our survey.

Run:  streamlit run dashboard/app.py
Loads only precomputed artefacts from outputs/ — rerun `python -m src.run_all` to refresh.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.env_factors import CO2E_PER_KG, GARMENT_KG, units_to_impact  # noqa: E402

TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
PROC = ROOT / "data" / "processed"

st.set_page_config(page_title="From Trend to Waste — Early Warning", layout="wide")


@st.cache_data
def load_json(name):
    p = TAB / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_csv(name):
    p = TAB / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_eda(name, rename, plus_one=()):
    p = PROC / name
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    for c in plus_one:  # 0-based week index -> week 1..12
        df[c] = df[c] + 1
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].map({True: "Yes", False: "No"})
    return df[[c for c in rename if c in df.columns]].rename(columns=rename)


# Data & EDA tab: compact tables made by `python -m src.data_quality` (local only, derived from licensed data)
EDA = {
    "Products (VISUELLE 2.0, Nuna Lie)": ("eda_visuelle_products.parquet", {
        "season": "Season", "category": "Category", "color": "Colour", "fabric": "Fabric", "cluster_name": "Lifecycle cluster",
        "short_lived": "Short-lived", "high_leftover": "High leftover", "price": "Price (normalised)", "n_shops": "Shops",
        "delivered_12w": "Delivered (12 weeks)", "sold_12w": "Sold (12 weeks)", "leftover_12w": "Unsold after 12 weeks",
        "sell_through": "Sell-through", "peak_week": "Peak week", "weeks_peak_to_half": "Weeks from peak to half",
        "first_markdown_week": "First markdown week"}, (), "Sell-through"),
    "Customers (loyalty card, Dec 2016 - Feb 2020)": ("eda_customers.parquet", {
        "segment": "Segment", "freq_band": "Buying frequency", "purchase_days": "Purchase days", "purchases_per_year": "Purchase days per year",
        "recency_days": "Days since last purchase", "units": "Units bought", "spend": "Spend (index)", "items_per_basket": "Items per basket",
        "shops": "Shops visited", "short_lived_share": "Short-lived share of purchases", "markdown_share": "Share bought on markdown"},
        (), "Purchase days"),
    "Survey respondents": ("eda_survey.parquet", {
        "age_band": "Age band", "occupation": "Occupation", "location": "Location", "monthly_spend": "Monthly spend",
        "buying_frequency": "Buying frequency", "channel": "Channel", "keep_years": "Years a garment is kept",
        "wears": "Wears before dropping", "unworn_share": "Share of wardrobe unworn", "motive_trend": "Motive: trend",
        "motive_discount": "Motive: discount", "motive_novelty": "Motive: novelty", "motive_quality": "Motive: quality",
        "motive_environment": "Motive: environment"}, (), "Years a garment is kept"),
}


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "outputs" / "models" / "vis_risk_logit.joblib")


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


def who(users, decision):
    st.info(f"**Who uses it:** {users}  \n**Decision it supports:** {decision}")


NUM = st.column_config.NumberColumn
PCT = lambda label: NUM(label, format="%.0f%%")  # columns are multiplied by 100 before display
TERM = {"category": "Category", "color": "Colour", "fabric": "Fabric", "season_type": "Season", "price_q": "Price",
        "depth_q": "Initial units per shop"}


def nice_term(t):
    k, _, v = t.partition(" = ")
    if k == "season_type":
        return "Season: autumn-winter" if v == "AW" else "Season: spring-summer"
    return f"{TERM.get(k, k)}: {v}"


st.title("From Trend to Waste — early-warning dashboard")
st.caption("Decision support for fast-fashion buyers. Data: VISUELLE 2.0 (Nuna Lie, an Italian fast-fashion chain, 2017-2019: "
           "sales, deliveries, prices, loyalty-card customers) and our consumer survey. All figures come from the project's "
           "scripts and are listed in reports/claims_register.csv.")

lc, rs, vis, cs, fs = (load_json("vis_lifecycle_summary.json"), load_json("vis_risk_model_summary.json"),
                       load_json("vis_summary.json"), load_json("vis_customers_summary.json"), load_json("vis_financials_summary.json"))

tabs = st.tabs(["Overview", "Data & EDA", "1 · Pre-launch risk", "2 · Markdown watchlist", "3 · Customers & survey",
                "4 · Impact simulator", "5 · Financial statements"])

with tabs[0]:
    who("management, buying, finance and sustainability teams",
        "how big the excess-stock problem is, and whether acting early is worth it")
    am = lc["all_products_markdown"]
    sim = vis["simulation_main_scenario"]
    c = st.columns(4)
    c[0].metric("Stock unsold after 12 weeks", pct(vis["stock"]["leftover_share_of_delivered"], 1))
    c[1].metric("Products peaking in weeks 1-3", pct(am["share_peak_in_first_3_weeks"]))
    c[2].metric("Median first markdown week", f"week {am['median_first_markdown_week']:.0f}",
                help=f"demand then at {pct(am['median_demand_at_markdown_vs_peak'])} of peak")
    c[3].metric("Forecast error after 2 weeks (WAPE)", pct(vis["forecast_best_weekly_wape"]),
                delta=f"{(vis['forecast_best_weekly_wape'] - vis['forecast_naive_weekly_wape']) * 100:.0f} pts vs naive",
                delta_color="inverse")
    st.markdown("**Key insights**")
    st.markdown(
        f"1. **A third of stock is left over.** {pct(vis['stock']['leftover_share_of_delivered'], 1)} of delivered units were still "
        f"unsold after 12 weeks, although shelves rarely ran empty ({pct(vis['stock']['share_shop_weeks_ending_at_zero_stock'], 1)} of shop-weeks).\n"
        f"2. **Demand peaks early, markdowns come late.** {pct(am['share_peak_in_first_3_weeks'])} of products have their best week within "
        f"the first three weeks, but the first markdown comes in week {am['median_first_markdown_week']:.0f}; "
        f"{pct(am['marked_down_after_demand_halved'])} of marked-down products were cut only after demand had halved.\n"
        f"3. **Week 2 is the moment to act.** Before launch, risk is partly predictable (ROC-AUC {rs['auc_with_plan_gb']:.2f}); "
        f"at week 2 the sales forecast error is {pct(vis['forecast_best_weekly_wape'])} vs {pct(vis['forecast_naive_weekly_wape'])} "
        f"for the naive method.\n"
        f"4. **A loyal core buys most.** The top 20% of customers buy {pct(cs['share_units_top20pct_customers'])} of units, and "
        f"frequent customers buy more on markdown ({pct(cs['markdown_share_lowest_band'])} → {pct(cs['markdown_share_highest_band'])}).\n"
        f"5. **Acting early pays.** A week-2 rule would have cut unsold stock by {pct(-sim['leftover_change'], 1)} with revenue "
        f"{pct(sim['revenue_change'], 1)}, and raised gross profit by {pct(fs['policy_gross_profit_change']['cost50'], 1)} "
        f"(cost assumed at 50% of full price).")
    l, r = st.columns(2)
    l.image(str(FIG / "vis_lifecycle_clusters.png"), width="stretch")
    r.image(str(FIG / "vis_markdown_vs_demand.png"), width="stretch")
    l.image(str(FIG / "vis_leftover_by_season.png"), width="stretch")
    r.image(str(FIG / "vis_financial_statement.png"), width="stretch")
    st.image(str(FIG / "vis_demand_trend.png"), width="stretch")

with tabs[1]:
    st.subheader("Data overview, cleaning and exploratory analysis")
    who("analysts, and anyone checking the numbers", "whether the data can be trusted, and what it looks like before modelling")
    st.markdown("**1 · The datasets**")
    ov = load_csv("data_overview.csv")
    if ov is not None:
        st.dataframe(ov.rename(columns={"dataset": "Dataset", "rows": "Rows", "columns": "Columns", "period": "Period",
                                        "one row is": "One row is", "used_for": "Used for"}), hide_index=True, width="stretch")
    st.markdown("**2 · Cleaning log: missing values, duplicates, outliers, inconsistencies**")
    log = load_csv("data_quality_log.csv")
    if log is not None:
        f1, f2 = st.columns(2)
        ds = f1.multiselect("Dataset", list(log.dataset.unique()), default=list(log.dataset.unique()))
        ck = f2.multiselect("Type of check", list(log.check.unique()), default=list(log.check.unique()))
        st.dataframe(log[log.dataset.isin(ds) & log.check.isin(ck)].rename(columns={
            "dataset": "Dataset", "check": "Check", "what_we_found": "What we found", "how_handled": "How we handled it"}),
            hide_index=True, width="stretch")
    st.markdown("**3 · Explore the data** (pick a dataset, filter it, and choose a variable)")
    name = st.selectbox("Dataset to explore", list(EDA))
    fname, rename, plus_one, default_var = EDA[name]
    df = load_eda(fname, rename, plus_one)
    if df is None:
        st.info("EDA tables not built yet: run `python -m src.data_quality`.")
    else:
        cats = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        a, b, c3 = st.columns(3)
        fcol = a.selectbox("Filter by", ["(no filter)"] + cats)
        if fcol != "(no filter)":
            vals = sorted(df[fcol].dropna().astype(str).unique())
            keep = b.multiselect(f"{fcol} values", vals, default=vals)
            df = df[df[fcol].astype(str).isin(keep)]
        var = c3.selectbox("Variable", nums, index=nums.index(default_var) if default_var in nums else 0)
        x = df[var]
        k = st.columns(5)
        k[0].metric("Rows", f"{len(df):,}")
        k[1].metric("Missing", pct(x.isna().mean(), 1))
        k[2].metric("Mean", f"{x.mean():,.2f}")
        k[3].metric("Median", f"{x.median():,.2f}")
        k[4].metric("Std. deviation", f"{x.std():,.2f}")
        h1, h2 = st.columns(2)
        logx = h1.checkbox("Log scale (for skewed variables)", value=bool(x.dropna().min() >= 0 and x.skew() > 2))
        xs = np.log10(1 + x.dropna()) if logx else x.dropna()
        if len(xs):
            cnt, edges = np.histogram(xs, bins=min(40, max(5, xs.nunique())))
            mid = (edges[:-1] + edges[1:]) / 2
            hist = pd.DataFrame({("log10(1 + " + var + ")" if logx else var): np.round(mid, 3), "Count": cnt})
            h1.markdown(f"**Distribution of {var}**")
            h1.bar_chart(hist, x=hist.columns[0], y="Count", color="#C4663A")
        by = h2.selectbox("Compare the average by", cats, index=0) if cats else None
        if by:
            g = df.groupby(by, observed=True)[var].agg(["count", "mean", "median"]).reset_index()
            g = g.sort_values("mean", ascending=False).head(15)
            h2.bar_chart(g, x=by, y="mean", color="#2F6F73", horizontal=True)
            h2.dataframe(g.rename(columns={"count": "Rows", "mean": "Mean", "median": "Median"}).round(3),
                         hide_index=True, width="stretch")
        st.markdown("**Summary statistics** (all numeric variables, after the filter)")
        d = df[nums].describe().T
        d.insert(0, "missing", df[nums].isna().mean())
        st.dataframe(d.rename(columns={"missing": "Missing share", "count": "Count", "mean": "Mean", "std": "Std",
                                       "min": "Min", "25%": "Q1", "50%": "Median", "75%": "Q3", "max": "Max"}).round(3),
                     width="stretch")
        if len(nums) > 2:
            st.markdown("**Correlations** (Spearman rank, between numeric variables)")
            sample = df[nums].sample(min(len(df), 100_000), random_state=0)  # a sample keeps it fast on 1.2M customers
            st.dataframe(sample.corr(method="spearman").style.background_gradient(cmap="RdBu", vmin=-1, vmax=1)
                         .format("{:.2f}"), width="stretch")

with tabs[2]:
    st.subheader("Score a new product before launch")
    who("buyers and merchandise planners", "how deep the first order for a new product should be")
    st.write(f"Logistic regression trained on the 2017-2018 seasons and tested on 2019. It predicts the chance that a product ends "
             f"with **high leftover** (under 60% of its stock sold in 12 weeks). Test ROC-AUC {rs['auc_with_plan']:.2f}: a modest "
             "signal, so use it to decide how deep the first order is, then act on early sales.")
    m = load_model()
    qp = st.query_params  # optional ?category=...&initial_per_shop=12 to open with a planned product filled in
    defaults = {"category": "long sleeve", "color": "black", "fabric": "georgette", "season_type": "AW"}
    cols = st.columns(4)
    choice = {}
    labels = {"category": "Category", "color": "Colour", "fabric": "Fabric", "season_type": "Season (SS = spring-summer, AW = autumn-winter)"}
    for i, c_ in enumerate(m["cats"]):
        lv = m["levels"][c_]
        pick = qp.get(c_) if qp.get(c_) in lv else defaults.get(c_)
        choice[c_] = cols[i].selectbox(labels[c_], lv, index=lv.index(pick) if pick in lv else 0)

    def qnum(key, default, lo, hi):
        try:
            return min(max(float(qp.get(key, default)), lo), hi)
        except ValueError:
            return default
    c1, c2, c3, c4 = st.columns(4)
    tier = c1.slider("Price position within its category (0 = cheapest, 1 = priciest)", 0.0, 1.0, qnum("price_tier", 0.5, 0.0, 1.0))
    month = c2.selectbox("Release month", list(range(1, 13)), index=int(qnum("release_month", 9, 1, 12)) - 1,
                         format_func=lambda v: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][v - 1])
    shops = c3.slider("Number of shops", 1, 110, int(qnum("n_shops", m["defaults"]["n_shops"], 1, 110)))
    init = c4.slider("Initial units per shop (the first delivery)", 1.0, 20.0, round(qnum("initial_per_shop", m["defaults"]["initial_per_shop"], 1, 20), 1), 0.5)
    X = pd.DataFrame([{**choice, "price_tier": tier, "release_month": month, "log_shops": np.log(shops), "log_initial_per_shop": np.log1p(init)}])
    p = float(m["pipe"].predict_proba(X[m["cats"] + m["num"]])[0, 1])
    base = m["base_rate"]
    st.metric("Probability this product ends with high leftover", pct(p), delta=f"{(p / base - 1) * 100:+.0f}% vs average product",
              delta_color="inverse", help=f"average in the training seasons: {pct(base, 1)}")
    st.write("**Suggested action:** " + ("buy a shallow first order and plan replenishment only on strong week-1-2 sales."
                                         if p > base * 1.25 else "normal buying depth; check week-2 sales."))
    orr = load_csv("vis_risk_odds_ratios.csv")
    ref = rs["reference_levels"]
    st.markdown(f"**What raises or lowers the risk** (odds ratios vs a {ref['color']} {ref['category']} in {ref['fabric']}, "
                "spring-summer, middle price and depth; significant at 5%)")
    sig = orr[(orr.p_value < 0.05) & ~orr.term.str.contains("rare")]
    st.dataframe(pd.DataFrame({
        "Feature": sig.term.map(nice_term), "Effect": ["raises risk" if v > 1 else "lowers risk" for v in sig.odds_ratio],
        "Odds ratio": sig.odds_ratio, "95% CI": [f"{a:.2f}-{b:.2f}" for a, b in zip(sig.ci_low, sig.ci_high)],
        "p-value": ["< 0.001" if v < 0.001 else f"{v:.3f}" for v in sig.p_value]}),
        hide_index=True, width="stretch", column_config={"Odds ratio": NUM(format="%.2f×")})
    st.image(str(FIG / "vis_risk_odds_ratios.png"))

with tabs[3]:
    st.subheader("Week-2 watchlist (VISUELLE 2.0, unseen 2019 seasons)")
    who("merchandise planners and pricing", "which new products to stop replenishing and mark down early, two weeks after launch")
    pr = load_csv("vis_test_leftover_predictions.csv")
    thr = st.slider("Flag products whose predicted 12-week sell-through is below", 30, 90, 60, 5, format="%d%%") / 100
    season = st.multiselect("Season", sorted(pr.season.unique()), default=sorted(pr.season.unique()))
    w = pr[pr.season.isin(season)].copy()
    w["flag"] = w.pred_sell_through < thr
    a, b, c_ = st.columns(3)
    a.metric("Products flagged", f"{w.flag.sum():,} of {len(w):,}")
    b.metric("Predicted unsold units (flagged)", f"{w.loc[w.flag, 'pred_leftover'].sum():,.0f}")
    c_.metric("Actual unsold units (flagged)", f"{w.loc[w.flag, 'leftover_12w'].sum():,.0f}")
    show = w[w.flag].sort_values("pred_leftover", ascending=False)
    st.dataframe(pd.DataFrame({
        "Product": show.external_code.astype(str), "Season": show.season, "Category": show.category,
        "Delivered (12 wks)": show.delivered_12w, "Predicted sell-through": show.pred_sell_through * 100,
        "Predicted unsold": show.pred_leftover.round(0), "Mark down from week": show.recommended_markdown_week,
        "Actually marked down in week": show.actual_first_markdown_week, "Actual sell-through": show.sell_through * 100,
        "Actually unsold": show.leftover_12w}), hide_index=True, width="stretch",
        column_config={"Predicted sell-through": PCT("Predicted sell-through"), "Actual sell-through": PCT("Actual sell-through"),
                       "Predicted unsold": NUM(format="%d"), "Delivered (12 wks)": NUM(format="%d"), "Actually unsold": NUM(format="%d"),
                       "Mark down from week": NUM(format="%d"), "Actually marked down in week": NUM(format="%d")})
    st.caption("Action for each flagged product: cut its remaining deliveries. \"Mark down from week\" is the first week the "
               "forecast falls below half of the peak so far (None = the forecast never halves within 12 weeks). The actual "
               "columns are what really happened, shown so the forecast can be checked (None = not marked down within 12 weeks).")
    st.image(str(FIG / "vis_forecast_examples.png"))

with tabs[4]:
    st.subheader("Who buys? Loyalty-card customers (Dec 2016 - Feb 2020)")
    who("CRM / marketing and sustainability teams", "which customers to reach first with new products, and what consumers value")
    seg = load_csv("vis_customer_segments.csv")
    k = st.columns(4)
    k[0].metric("Customers", f"{cs['customers']:,}")
    k[1].metric("Units bought by the top 20%", pct(cs["share_units_top20pct_customers"]))
    k[2].metric("Bought on markdown", pct(cs["markdown_share_all_known_lines"]), help="purchases with a known discount")
    k[3].metric("Bought on a single day only", pct(cs["share_one_purchase_day"]))
    st.dataframe(pd.DataFrame({
        "Segment": seg.segment, "Share of customers": seg.share_of_customers * 100, "Share of units": seg.share_of_units * 100,
        "Share of spend": seg.share_of_spend * 100, "Purchase days (median)": seg.purchase_days_median,
        "Days since last purchase (median)": seg.recency_median, "Items per basket": seg.items_per_basket_mean,
        "Bought on markdown": seg.markdown_share_mean * 100, "CLTV (index)": seg.cltv_index}),
        hide_index=True, width="stretch",
        column_config={c_: PCT(c_) for c_ in ["Share of customers", "Share of units", "Share of spend", "Bought on markdown"]}
        | {"Purchase days (median)": NUM(format="%d"), "Days since last purchase (median)": NUM(format="%d"),
           "Items per basket": NUM(format="%.1f"), "CLTV (index)": NUM(format="%.2f")})
    l, r = st.columns(2)
    l.image(str(FIG / "vis_customer_segments.png"), width="stretch")
    r.image(str(FIG / "vis_customers_by_frequency.png"), width="stretch")
    sv = load_json("survey_descriptives.json")
    st.subheader("Consumer survey")
    if sv is None:
        st.info("Awaiting real survey responses (data/raw/survey/). Run `python -m src.survey_analysis` after collecting them.")
    else:
        st.caption(f"{sv['respondents']} real responses, a convenience sample in India ({sv['share_age_18_24']:.0%} aged 18-24). "
                   "Stated behaviour, compared with Nuna Lie's customers by group only.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Years a garment is kept", f"{sv['mean_keep_years']:.1f}", help="WRAP UK benchmark: 2.2-3.3 years")
        m2.metric("Wear a fast-fashion item ≤10 times", f"{sv['share_wears_under_10']:.0%}")
        m3.metric("Wardrobe unworn last year", f"{sv['mean_unworn_share']:.0%}")
        m4.metric("Never resell", f"{sv['disposal_share_never']['Sell / resell']:.0%}")
        st.image(str(FIG / "survey_lifetime.png"))
        st.image(str(FIG / "survey_disposal.png"))
        st.image(str(FIG / "survey_attitudes.png"))
        st.markdown("**What would they buy? Conjoint (price × style × lifespan × label)**")
        st.image(str(FIG / "survey_conjoint_importance.png"))
        st.write(f"Willingness to pay for a 3+ year lifespan: about Rs {sv['wtp_rs']['long_life']:,.0f} "
                 f"(95% CI Rs {sv['wtp_rs_ci95']['long_life'][0]:,.0f}-{sv['wtp_rs_ci95']['long_life'][1]:,.0f}). "
                 "Stated preference; values above the Rs 700 price gap tested are extrapolation.")
        st.markdown("**Shopper groups (k-means on purchase motives)**")
        sg = load_csv("survey_segments.csv")
        st.dataframe(pd.DataFrame({
            "Shopper group": sg.name, "People": sg.respondents, "Trend (1-5)": sg.Trend, "Novelty (1-5)": sg.Novelty,
            "Quality (1-5)": sg.Quality, "Environment (1-5)": sg.Environment, "Keeps clothes (years)": sg.keep_years,
            "Often throws away": sg.throw_away_often * 100, "Buys fast fashion": sg.buys_fast_fashion * 100}),
            hide_index=True, width="stretch",
            column_config={c_: NUM(format="%.1f") for c_ in ["Trend (1-5)", "Novelty (1-5)", "Quality (1-5)",
                                                              "Environment (1-5)", "Keeps clothes (years)"]}
            | {"Often throws away": PCT("Often throws away"), "Buys fast fashion": PCT("Buys fast fashion")})
        st.markdown("**Survey and Nuna Lie customers, by buying frequency (group level; different people and countries)**")
        bf = load_csv("survey_vis_bridge_by_frequency.csv")
        st.dataframe(pd.DataFrame({
            "Buys clothes": bf.freq_band, "Survey people": bf.respondents, "Survey: trend influence (1-5)": bf.trend_influence,
            "Survey: discount influence (1-5)": bf.discount_influence, "Nuna Lie customers": bf.vis_customers,
            "Nuna Lie: bought on markdown": bf.vis_markdown_share * 100, "Nuna Lie: short-lived products": bf.vis_short_lived_share * 100}),
            hide_index=True, width="stretch",
            column_config={"Survey: trend influence (1-5)": NUM(format="%.2f"), "Survey: discount influence (1-5)": NUM(format="%.2f"),
                           "Nuna Lie customers": NUM(format="localized"), "Nuna Lie: bought on markdown": PCT("Nuna Lie: bought on markdown"),
                           "Nuna Lie: short-lived products": NUM("Nuna Lie: short-lived products", format="%.1f%%")})

with tabs[5]:
    st.subheader("What if the retailer had acted at week 2? (VISUELLE 2.0, 2019 seasons)")
    who("management, finance and sustainability teams", "which week-2 rule to adopt: how much to cut and whether to mark down")
    grid = load_csv("vis_simulation_grid.csv")
    c1, c2, c3, c4 = st.columns(4)
    taus = sorted(grid.flag_threshold_pred_sell_through.unique())
    tau = c1.selectbox("Flag if predicted sell-through below", taus, index=taus.index(0.6) if 0.6 in taus else 0,
                       format_func=lambda v: f"{v:.0%}")
    pol = c2.selectbox("Policy", ["A+B", "A: cut later deliveries", "B: earlier markdown"],
                       format_func=lambda v: {"A+B": "A+B: cut deliveries and mark down"}.get(v, v))
    cut = c3.select_slider("Cut to later deliveries", [0.1, 0.2, 0.3, 0.4], value=0.3, disabled=pol.startswith("B"),
                           format_func=lambda v: f"{v:.0%}")
    md = c4.select_slider("Markdown level from week 4", [0.2, 0.3], value=0.2, disabled=pol.startswith("A:"),
                          format_func=lambda v: f"{v:.0%}")
    upl = st.radio("Markdown uplift estimate", sorted(grid.uplift_estimate.unique()), horizontal=True)
    q = grid[(grid.flag_threshold_pred_sell_through == tau) & (grid.policy == pol) & (grid.uplift_estimate == upl)]
    if not pol.startswith("B"):
        q = q[q.delivery_cut == cut]
    if not pol.startswith("A:"):
        q = q[q.markdown_level == md]
    row = q.iloc[0]
    k = st.columns(4)
    k[0].metric("Unsold stock", pct(row.leftover_change, 1))
    k[1].metric("Units sold", pct(row.sold_change, 1))
    k[2].metric("Revenue", pct(row.revenue_change, 1))
    k[3].metric("Units kept out of leftover", f"{row.units_leftover_avoided:,.0f}")
    imp = units_to_impact(row.units_leftover_avoided)
    st.write(f"≈ **{imp['tonnes_textile_low']:.0f}-{imp['tonnes_textile_high']:.0f} tonnes** of textiles kept out of the unsold pile and "
             f"**{imp['tonnes_co2e_low']:.0f}-{imp['tonnes_co2e_high']:.0f} t CO₂e** (estimate range: "
             f"{GARMENT_KG['low']}-{GARMENT_KG['high']} kg per garment × {CO2E_PER_KG['low']}-{CO2E_PER_KG['high']} kg CO₂e per kg).")
    if pd.notna(row.breakeven_cost_to_price):
        st.write(f"Gross profit rises if unit cost exceeds **{row.breakeven_cost_to_price:.0%}** of full price, which holds for any "
                 "real retailer. The Financial statements tab shows the profit effect for assumed unit costs.")
    st.caption("Demand = observed sales (stock-outs are rare: "
               f"{pct(vis['stock']['share_shop_weeks_ending_at_zero_stock'], 1)} of shop-weeks). Leftover stock's later "
               "clearance value is not modelled. Markdown uplift is estimated from observational data. We see stock delivered and sold, "
               "not production: tonnes assume the cut deliveries would not have been made.")
    st.image(str(FIG / "vis_simulation.png"))

with tabs[6]:
    st.subheader("Financial statements built from the data (Module 4)")
    who("finance and management", "how much excess stock costs, and what the week-2 rule is worth")
    st.caption("Nuna Lie publishes no accounts, so these statements are built from the dataset's deliveries, sales, prices and "
               "discounts (each product's first 12 weeks in every shop). Money is in index units (prices are normalised); ratios "
               "are unaffected. Unit cost is not in the data, so margins use an assumed cost as a share of full price.")
    st_ = load_csv("vis_financial_statement.csv")
    cost = st.select_slider("Assumed unit cost (share of full price)", [0.4, 0.5, 0.6], value=0.5, format_func=lambda v: f"{v:.0%}")
    kk = int(cost * 100)
    a = st_[st_.season == "All seasons"].iloc[0]
    k = st.columns(5)
    k[0].metric("Sell-through", pct(a.sell_through))
    k[1].metric("Markdown cost / full-price sales", pct(a.markdown_rate, 1))
    k[2].metric("Unsold stock / revenue", pct(a.stock_to_sales))
    k[3].metric("Margin on goods sold", pct(a[f"gross_margin_cost{kk}"]))
    k[4].metric("Margin if unsold stock is lost", pct(a[f"margin_after_unsold_cost{kk}"]))
    st.dataframe(pd.DataFrame({
        "Season": st_.season, "Units delivered": st_.units_delivered, "Sell-through": st_.sell_through * 100,
        "Net revenue": st_.net_revenue, "Markdown cost": st_.markdown_cost, "Unsold stock (full price)": st_.closing_stock_value,
        "Markdown rate": st_.markdown_rate * 100, "Unsold stock / revenue": st_.stock_to_sales * 100,
        "Turnover (12 wks)": st_.inventory_turnover_12w, "Days of inventory": st_.days_of_inventory,
        "Margin on goods sold": st_[f"gross_margin_cost{kk}"] * 100, "Margin if unsold lost": st_[f"margin_after_unsold_cost{kk}"] * 100}),
        hide_index=True, width="stretch",
        column_config={c_: NUM(format="localized") for c_ in ["Units delivered", "Net revenue", "Markdown cost", "Unsold stock (full price)"]}
        | {c_: PCT(c_) for c_ in ["Sell-through", "Unsold stock / revenue", "Margin on goods sold", "Margin if unsold lost"]}
        | {"Markdown rate": NUM("Markdown rate", format="%.1f%%"), "Turnover (12 wks)": NUM(format="%.2f"), "Days of inventory": NUM(format="%.0f")})
    st.image(str(FIG / "vis_financial_statement.png"))
    st.markdown("**The week-2 rule in the statement (2019 seasons)**")
    pol = load_csv("vis_financial_policy.csv")
    gp = pol[f"gross_profit_after_unsold_cost{kk}"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Revenue", pct(pol.net_revenue.iloc[1] / pol.net_revenue.iloc[0] - 1, 1))
    m2.metric("Stock bought (full price)", pct(pol.full_price_value_delivered.iloc[1] / pol.full_price_value_delivered.iloc[0] - 1, 1))
    m3.metric("Gross profit after unsold stock", pct(gp.iloc[1] / gp.iloc[0] - 1, 1))
    st.caption("Gross profit after unsold stock = net revenue − assumed cost × all stock bought. The later clearance value of "
               "leftover stock is not in the data, so this is a cautious view.")
