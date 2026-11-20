"""Review 2 dashboard: the project at its mid-point (data, cleaning, EDA findings, relationships, plan).

Run:  streamlit run dashboard/review2_app.py
Reads outputs/review2/review2_data.json (python -m src.review2_data) and the data-quality tables (python -m src.data_quality).
All retail data: VISUELLE 2.0 (Nuna Lie, an Italian fast-fashion chain).
"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
TAB, PROC = ROOT / "outputs" / "tables", ROOT / "data" / "processed"
ACC, TEAL, SAND, INK = "#C4663A", "#2F6F73", "#D9C7A7", "#3A3532"
CLUSTER_COLORS = ["#C4663A", "#2F6F73", "#8C6BB1", "#D9A441"]

st.set_page_config(page_title="From Trend to Waste — Review 2", layout="wide")


@st.cache_data
def load():
    d = json.loads((ROOT / "outputs" / "review2" / "review2_data.json").read_text())
    p = PROC / "r2_vis_products.parquet"
    prods = pd.read_parquet(p) if p.exists() else None
    ov = pd.read_csv(TAB / "data_overview.csv") if (TAB / "data_overview.csv").exists() else None
    log = pd.read_csv(TAB / "data_quality_log.csv") if (TAB / "data_quality_log.csv").exists() else None
    return d, prods, ov, log


D, PRODS, OV, LOG = load()
S = D["stats"]
pct = lambda x, d=0: f"{x * 100:.{d}f}%"


def finding(title, text):
    st.markdown(f"<div style='background:#F1DED2;border-left:4px solid {ACC};padding:12px 16px;border-radius:6px'>"
                f"<b>{title}</b><br>{text}</div>", unsafe_allow_html=True)


st.title("From Trend to Waste")
st.caption("Fast-fashion consumer behaviour and excess-inventory risk · Review 2 · "
           "Madhusudhanan G · Pharos Sophy Samuel T J · Sri Charan")

tabs = st.tabs(["Overview", "Data & preprocessing", "Findings: 5 questions", "Methodology & next steps"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    st.progress(0.6, text="Project progress: about 60%")
    a, b, c = st.columns(3)
    a.markdown("**Done**\n- Data: VISUELLE 2.0 (Nuna Lie) sales, deliveries, prices, customers\n- Cleaning and validation\n"
               "- Exploratory analysis (this dashboard)\n- Consumer survey collected")
    b.markdown("**In progress**\n- Choosing and testing models\n- Lifecycle clusters and markdown flags (built)")
    c.markdown("**Next**\n- Pre-launch risk model\n- Week-2 sales forecast\n- What-if simulation of early action\n"
               "- Financial statements from the data\n- Survey analysis (conjoint)\n- Final decision dashboard")
    k = st.columns(5)
    k[0].metric("Products", f"{S['products']:,}", help=f"in {S['shops']} shops, 6 seasons 2017-2019")
    k[1].metric("Customer purchases", f"{S['purchase_lines'] / 1e6:.2f} M", help=f"{S['customers']:,} loyalty-card customers")
    k[2].metric("Stock unsold after 12 weeks", pct(S["vis_leftover"], 1))
    k[3].metric("Products peaking in weeks 1-3", pct(S["peak_first3"]))
    k[4].metric("Survey responses", f"{S['survey_n']}")
    st.subheader("Key insights so far")
    st.markdown(
        f"1. **Demand peaks early.** {pct(S['peak_first3'])} of products have their best-selling week within their first three weeks.\n"
        f"2. **Markdowns lag demand.** The first markdown comes in week {S['first_md_week']:.0f} on median, {S['weeks_peak_to_md']:.0f} weeks "
        f"after the peak, when demand is down to {pct(S['demand_at_md'])} of peak; {pct(S['md_after_half'])} of marked-down products "
        f"were cut only after demand had halved.\n"
        f"3. **A few loyal customers buy most, and frequent customers lean on markdowns.** The top 20% of customers buy "
        f"{pct(S['top20_units'])} of units; the share bought on markdown rises from {pct(S['md_lo'])} to {pct(S['md_hi'])} with buying frequency.\n"
        f"4. **A third of stock is left over.** {pct(S['vis_leftover'], 1)} of delivered units were unsold after 12 weeks, "
        f"and {pct(S['cat_left_lo'])}-{pct(S['cat_left_hi'])} in the larger categories, although shelves rarely ran empty "
        f"({pct(S['stockout'], 1)} of shop-weeks).\n"
        f"5. **More stock per shop, more leftover; and week 2 already shows it.** Products with the most deliveries per shop "
        f"sell {pct(S['dps_d10_st'])} of their stock vs {pct(S['dps_d1_st'])} for the fewest (ρ = {S['dps_rho']:.2f}); "
        f"weak first-two-week sales point to low sell-through (ρ = {S['early_rho']:.2f}).")
    st.subheader("The five questions")
    st.dataframe(pd.DataFrame({
        "Question": ["Q1 What does a product's 12-week life look like?", "Q2 Do markdowns follow demand, or lag behind it?",
                     "Q3 How do frequent and occasional customers differ?",
                     "Q4 How much delivered stock is left unsold?", "Q5 Does pushing more stock per shop leave more unsold?"],
        "Data": [f"{S['lc_products']:,} products", f"{S['lc_products']:,} products", f"{S['customers']:,} customers",
                 f"{S['shops']} shops, 6 seasons", f"{S['q5_n']:,} products"],
        "Answer so far": [f"{S['lc_k']} lifecycles; {pct(S['peak_first3'])} peak in weeks 1-3",
                          f"They lag: first markdown in week {S['first_md_week']:.0f}",
                          f"Top 20% buy {pct(S['top20_units'])}; frequent ones buy more on markdown",
                          f"{pct(S['vis_leftover'], 1)}; {pct(S['cat_left_lo'])}-{pct(S['cat_left_hi'])} by larger category",
                          f"Yes: ρ = {S['dps_rho']:.2f}"]}), hide_index=True, width="stretch")

# ------------------------------------------------------------------ data
with tabs[1]:
    st.subheader("Datasets")
    st.caption("All retail files come from one retailer (Nuna Lie) and join on product code, shop and date.")
    if OV is not None:
        st.dataframe(OV.rename(columns=str.capitalize), hide_index=True, width="stretch")
    st.subheader("Cleaning: missing values, duplicates, outliers, inconsistencies")
    if LOG is not None:
        piv = LOG.pivot_table(index="check", columns="dataset", values="what_we_found", aggfunc="count", fill_value=0)
        c1, c2 = st.columns([1, 2])
        c1.markdown("**Checks run, by dataset**")
        c1.dataframe(piv, width="stretch")
        ds = c2.multiselect("Show dataset", list(LOG.dataset.unique()), default=list(LOG.dataset.unique()))
        c2.dataframe(LOG[LOG.dataset.isin(ds)].rename(columns={"dataset": "Dataset", "check": "Check",
                     "what_we_found": "What we found", "how_handled": "How we handled it"}), hide_index=True, width="stretch")
    st.subheader("Variables we built for the analysis")
    st.markdown("- **Unit:** a product across all its shops, over its first 12 weeks after release\n"
                "- **Short-lived product:** weekly sales fall below 25% of the peak within 4 weeks of it, and weeks 7-12 hold under 20% of 12-week sales (rule fixed before modelling)\n"
                "- **Markdown week:** average discount across shops of at least 15%\n"
                "- **Sell-through:** units sold ÷ units delivered in the first 12 weeks; **leftover** = delivered − sold\n"
                "- **Deliveries per shop** and **weeks 1-2 sales as a share of the season's deliveries**\n"
                "- **Customer buying frequency:** purchase days per year, in the same bands as our survey")

# ------------------------------------------------------------------ findings
with tabs[2]:
    lcc = pd.DataFrame(D["lifecycle_curves"])
    mdc = pd.DataFrame(D["markdown_curves"])
    groups = list(dict.fromkeys(lcc.group))

    st.subheader("Q1 · What does a product's 12-week life look like?")
    l, r = st.columns([2, 1])
    g1 = r.selectbox("Category", groups, key="g1")
    q = lcc[lcc.group == g1].copy()
    q["label"] = q.cluster + " (" + (q.share_of_products * 100).round(0).astype(int).astype(str) + "%)"
    l.altair_chart(alt.Chart(q).mark_line(strokeWidth=3).encode(
        x=alt.X("week:Q", title="Week since release", scale=alt.Scale(domain=[1, 12])),
        y=alt.Y("pct_of_12w_sales:Q", title="% of the product's 12-week sales"),
        color=alt.Color("label:N", title="Lifecycle (share of products)", scale=alt.Scale(range=CLUSTER_COLORS)),
        tooltip=["cluster", "week", alt.Tooltip("pct_of_12w_sales:Q", format=".1f")]).properties(height=320), width="stretch")
    gi = next(x for x in D["groups"] if x["group"] == g1)
    r.metric("Products in this view", f"{gi['products']:,}", help=f"short-lived: {pct(gi['short_lived_share'], 1)}")
    with r:
        finding("Finding", f"K-means groups products into {S['lc_k']} lifecycles. Most demand comes early: {pct(S['peak_first3'])} "
                f"of products have their best-selling week within the first three weeks; only late bloomers build up through the season.")

    st.divider()
    st.subheader("Q2 · Do markdowns follow demand, or lag behind it?")
    l, r = st.columns([2, 1])
    g2 = r.selectbox("Category", groups, key="g2")
    t2 = r.radio("Products", ["All products", "Short-lived", "Other"], horizontal=True)
    q = mdc[(mdc.group == g2) & (mdc.type == t2)].melt(id_vars="week", value_vars=["demand_pct_of_peak", "pct_on_markdown"]).dropna()
    q["variable"] = q.variable.map({"demand_pct_of_peak": "Demand (% of own peak)", "pct_on_markdown": "% of products on markdown"})
    l.altair_chart(alt.Chart(q).mark_line(strokeWidth=3).encode(
        x=alt.X("week:Q", title="Week since release", scale=alt.Scale(domain=[1, 12])), y=alt.Y("value:Q", title="%"),
        color=alt.Color("variable:N", title=None, scale=alt.Scale(range=[ACC, TEAL]), legend=alt.Legend(orient="top")),
        strokeDash=alt.StrokeDash("variable:N", title=None, legend=alt.Legend(orient="top")),
        tooltip=["variable", "week", alt.Tooltip("value:Q", format=".1f")]).properties(height=320), width="stretch")
    with r:
        finding("Finding", f"Discounts rise as demand falls, not before: the first markdown comes in week {S['first_md_week']:.0f} on median, "
                f"{S['weeks_peak_to_md']:.0f} weeks after the peak, when demand is down to {pct(S['demand_at_md'])} of peak. "
                f"{pct(S['md_after_half'])} of marked-down products ({pct(S['sl_md_after_half'])} of short-lived ones) were cut only after demand halved.")

    st.divider()
    st.subheader("Q3 · How do frequent and occasional customers differ?")
    l, r = st.columns([2, 1])
    metric = r.radio("Show", ["Share bought on markdown", "Short-lived products bought", "Share of customers", "Share of units bought"])
    col = {"Share bought on markdown": "markdown_share", "Short-lived products bought": "short_lived_share",
           "Share of customers": "share_of_customers", "Share of units bought": "share_of_units"}[metric]
    fq = pd.DataFrame(D["by_freq"])
    fq["value"] = fq[col] * 100
    order = list(fq.freq_band)
    bars = alt.Chart(fq).mark_bar(color=TEAL).encode(
        x=alt.X("freq_band:N", sort=order, title="How often the customer buys", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("value:Q", title=f"{metric} (%)"), tooltip=["freq_band", "customers", alt.Tooltip("value:Q", format=".1f")])
    l.altair_chart((bars + bars.mark_text(dy=-8, color=INK).encode(text=alt.Text("value:Q", format=".1f"))).properties(height=320), width="stretch")
    with r:
        finding("Relationship", f"Frequent customers buy a larger share on markdown ({pct(S['md_lo'])} → {pct(S['md_hi'])}; Spearman ρ = "
                f"{S['rho_md']:.2f}), but short-lived products are a similar small share for everyone ({pct(S['sl_lo'], 1)}-{pct(S['sl_hi'], 1)}). "
                f"Purchases are concentrated: the top 20% of customers buy {pct(S['top20_units'])} of units, and {pct(S['one_day'])} bought on a single day only.")

    st.divider()
    st.subheader("Q4 · How much delivered stock is left unsold?")
    l, r = st.columns([2, 1])
    cs_ = pd.DataFrame(D["vis_category_season"])
    seasons = r.multiselect("Seasons", sorted(cs_.season.unique()), default=sorted(cs_.season.unique()))
    n = r.slider("Categories shown (largest first)", 5, 20, 12)
    q = cs_[cs_.season.isin(seasons)].groupby("category")[["delivered", "leftover"]].sum().nlargest(n, "delivered")
    q = q.assign(leftover_pct=q.leftover / q.delivered * 100).reset_index()
    overall = cs_[cs_.season.isin(seasons)][["delivered", "leftover"]].sum()
    avg = overall.leftover / max(overall.delivered, 1) * 100
    bars = alt.Chart(q).mark_bar(color=ACC).encode(
        x=alt.X("leftover_pct:Q", title="% of delivered units still unsold after 12 weeks"),
        y=alt.Y("category:N", sort="-x", title=None), tooltip=["category", "delivered", alt.Tooltip("leftover_pct:Q", format=".1f")])
    rule = alt.Chart(pd.DataFrame({"x": [avg]})).mark_rule(color=INK, strokeDash=[4, 3]).encode(x="x:Q")
    l.altair_chart((bars + rule).properties(height=340), width="stretch")
    r.metric("Unsold after 12 weeks (selected seasons)", f"{avg:.1f}%")
    with r:
        finding("Finding", f"About a third of stock is left over in every season and in the larger categories ({pct(S['vis_leftover'], 1)} overall; "
                f"{pct(S['cat_left_lo'])} for {S['cat_left_lo_name']} to {pct(S['cat_left_hi'])} for {S['cat_left_hi_name']}, among the 75% of categories with the most deliveries). "
                "Over-supply is general, not limited to a few products.")

    st.divider()
    st.subheader("Q5 · Does pushing more stock per shop leave more unsold?")
    l, r = st.columns([2, 1])
    xv = r.radio("Compare 12-week sell-through with", ["Deliveries per shop", "Weeks 1-2 sales ÷ season deliveries"])
    key = "deliveries_per_shop" if xv.startswith("Deliveries") else "early_share"
    if PRODS is not None:
        sea = r.multiselect("Seasons ", sorted(PRODS.season.unique()), default=sorted(PRODS.season.unique()))
        q = PRODS[PRODS.season.isin(sea)].copy()
        rho = q[key].rank().corr(q.sell_through_12w.rank())
        q["x"] = q[key]
        pts = alt.Chart(q).mark_circle(size=14, opacity=0.25, color=TEAL).encode(
            x=alt.X("x:Q", title=xv, scale=alt.Scale(type="log") if key == "deliveries_per_shop" else alt.Scale()),
            y=alt.Y("sell_through_12w:Q", title="12-week sell-through", axis=alt.Axis(format="%")),
            tooltip=["category", "season", alt.Tooltip("x:Q", format=".2f"), alt.Tooltip("sell_through_12w:Q", format=".0%")])
        dec = q.assign(d=pd.qcut(q.x.rank(method="first"), 10, labels=False)).groupby("d").agg(x=("x", "median"), y=("sell_through_12w", "mean"))
        line = alt.Chart(dec).mark_line(color=ACC, strokeWidth=4, point=True).encode(x="x:Q", y="y:Q")
        l.altair_chart((pts + line).properties(height=340), width="stretch")
        r.metric("Spearman ρ (selected seasons)", f"{rho:.2f}", help=f"{len(q):,} products")
    else:
        dec = pd.DataFrame(D["relationships"][key]["deciles"])
        l.altair_chart(alt.Chart(dec).mark_line(color=ACC, point=True).encode(x="x:Q", y="sell_through:Q"), width="stretch")
    with r:
        finding("Relationship", f"Products sent in the largest quantities per shop sell {pct(S['dps_d10_st'])} of their stock, vs "
                f"{pct(S['dps_d1_st'])} for the smallest (ρ = {S['dps_rho']:.2f}). And by week 2 the outcome is visible: the slowest "
                f"early sellers end at {pct(S['early_d1_st'])}, the fastest at {pct(S['early_d10_st'])} (ρ = {S['early_rho']:.2f}). "
                "Orange line = average per tenth of products. Association, not causation.")

# ------------------------------------------------------------------ methodology
with tabs[3]:
    st.subheader("Methodology for the remaining 40%")
    st.dataframe(pd.DataFrame([
        ("Can excess-stock products be spotted before launch?", "Logistic regression (+ gradient boosting check), time-based split",
         "Explainable odds ratios for buyers; tested on later seasons", "Q4, Q5"),
        ("How much will sell after week 2?", "Gradient boosting vs naive and lifecycle baselines; WAPE",
         "Weekly counts with many zeros; Q5 shows week 2 carries a signal", "Q1, Q5"),
        ("Which customers to reach?", "RFM features + K-means; CLTV", "Standard marketing segmentation; purchases are concentrated (Q3)", "Q3"),
        ("Would acting at week 2 reduce leftover?", "Week-by-week stock replay with a markdown-uplift regression",
         "Uses real deliveries and sales; we cannot experiment on the past", "Q2, Q4, Q5"),
        ("What does excess stock cost?", "Financial statements built from the data (stock-to-sales, markdown rate, margins)",
         "Nuna Lie publishes no accounts; the data has prices, discounts and stock", "Q4"),
        ("What do consumers value?", "Conjoint analysis on the survey (OLS with respondent fixed effects)",
         "Measures trade-offs between price, style, lifespan, label", "Survey"),
    ], columns=["Question", "Method", "Why it fits", "Builds on"]), hide_index=True, width="stretch")
    st.subheader("Plan")
    st.dataframe(pd.DataFrame([
        ("1", "Pre-launch risk model and odds ratios", "Next"), ("2", "Week-2 forecast and early-warning flag", "Next"),
        ("3", "Customer segments and CLTV", "Next"), ("4", "What-if simulation and financial statements", "Next"),
        ("5", "Survey conjoint and comparison with customers", "Next"), ("6", "Decision dashboard, report and final review", "Final review"),
    ], columns=["Step", "Work", "When"]), hide_index=True, width="stretch")
