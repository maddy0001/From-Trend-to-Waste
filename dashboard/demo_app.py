"""Guided project demo: the whole analysis as one story, step by step (VISUELLE 2.0 / Nuna Lie + our survey).

Run:  streamlit run dashboard/demo_app.py
Every number is read from the project's outputs (rerun `python -m src.run_all` to refresh). Product photos come from
visuelle2/images (licensed data, local only). Switch on "Presenter notes" in the sidebar for what to say and click.
"""
import json
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import (CUSTOMER_MAX_SHOPS, HIGH_LEFTOVER_SELL_THROUGH, K_WEEKS, MARKDOWN_SENSITIVITY,  # noqa: E402
                        MARKDOWN_THRESHOLD, SEASON_ORDER)
from src.env_factors import CO2E_PER_KG, GARMENT_KG, units_to_impact  # noqa: E402

TAB, PROC, IMG = ROOT / "outputs" / "tables", ROOT / "data" / "processed", ROOT / "visuelle2" / "images"
MODEL = ROOT / "outputs" / "models" / "vis_risk_logit.joblib"
WEEKS = list(range(1, K_WEEKS + 1))
U_COLS, D_COLS, DSC_COLS = ([f"{c}{k}" for k in range(K_WEEKS)] for c in ("u", "dl", "d"))
HIGH = HIGH_LEFTOVER_SELL_THROUGH
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Project terracotta + a categorical set that passes the colour-blind checks (dataviz palette validator, white surface)
INK, TEXT, MUTED, GRID, SAND, GREY = "#2B2B2B", "#3A3532", "#6F675E", "#ECE7E0", "#D8D0C4", "#9A9187"
TERRA, TEAL, PURPLE, GOLD = "#C4663A", "#00909A", "#7B5FB8", "#B7881F"
CLUSTER = {"Fast fade": TERRA, "Steady decline": TEAL, "Mid-season peak": PURPLE, "Late bloomer": GOLD}
SHAPE_TEXT = {  # from the naming rules in src/vis_lifecycle.py (name_clusters)
    "Fast fade": "Over 40% of its 12-week sales come in the first three weeks, then demand drops away.",
    "Steady decline": "Starts strong and loses demand gradually, week after week.",
    "Mid-season peak": "After the launch week, demand builds again to a second high in mid-season.",
    "Late bloomer": "Slow after launch: about half of its sales come in weeks 7-12.",
}
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
MD_ON = f"On markdown ({MARKDOWN_THRESHOLD:.0%}+ off)"
MD_OFF = "Full price or small discount"

st.set_page_config(page_title="From Trend to Waste · Demo", page_icon="🧵", layout="wide", initial_sidebar_state="expanded")

CSS = """<style>
.stApp {background:#FAF9F7;}
.block-container {padding-top:2.4rem; padding-bottom:3rem; max-width:1320px;}
[data-testid="stSidebar"] {background:#F2EDE6;}
.kicker {font-size:.78rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:#B5532C; margin:0 0 .2rem;}
.question {font-family:Georgia, Cambria, 'Times New Roman', serif; font-size:2.1rem; line-height:1.2; font-weight:700; color:#2B2B2B; margin:0 0 .35rem;}
.sub {font-size:1.06rem; color:#5E574F; margin:0 0 1rem; max-width:980px;}
.hero {background:#2B2B2B; border-radius:22px; padding:30px 36px 28px; margin:0 0 1.2rem;}
.hero .kicker {color:#E8A07C;}
.hero .htitle {font-family:Georgia, Cambria, serif; color:#FFFFFF; font-size:3.1rem; font-weight:700; line-height:1.1; margin:.3rem 0 .6rem;}
.hero p {color:#E9E4DC; font-size:1.15rem; line-height:1.5; margin:0; max-width:920px;}
.flow {display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin-top:24px;}
.flow .chip {padding:8px 18px; border-radius:999px; border:1px solid #6B635A; color:#FAF9F7; font-weight:700; font-size:.82rem; letter-spacing:.1em;}
.flow .chip.last {background:#C4663A; border-color:#C4663A;}
.flow .arrow {color:#9A9187; font-size:1.1rem;}
.kpi {background:#FFFFFF; border:1px solid #ECE6DD; border-radius:16px; padding:15px 18px 14px; box-shadow:0 1px 3px rgba(43,43,43,.05); margin-bottom:.6rem;}
.kpi .v {font-family:-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; font-size:1.85rem; font-weight:700; color:#B5532C; line-height:1.15;}
.kpi .l {font-size:.9rem; color:#4A443E; margin-top:6px; line-height:1.35;}
.take {background:#F6E7DD; border-left:6px solid #C4663A; border-radius:12px; padding:14px 18px; color:#2B2B2B; font-size:1.03rem; line-height:1.5; margin:.8rem 0 .5rem;}
.read {color:#6F675E; font-size:.92rem; line-height:1.45; margin:.3rem 0 .6rem;}
.say {background:#FFFDF9; border:2px dashed #C4663A; border-radius:14px; padding:12px 18px; margin:.2rem 0 1rem; font-size:.98rem; line-height:1.5;}
.say .t {font-weight:800; color:#B5532C; letter-spacing:.12em; font-size:.72rem; margin-bottom:2px;}
.card {background:#FFFFFF; border:1px solid #ECE6DD; border-radius:16px; padding:14px 18px; box-shadow:0 1px 3px rgba(43,43,43,.05); margin-bottom:.6rem;}
.big {font-family:-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; font-size:3.3rem; font-weight:800; color:#2B2B2B; line-height:1.05;}
.meter {height:14px; background:#EFE9E1; border-radius:999px; position:relative; margin:14px 0 6px;}
.meter .fill {height:14px; border-radius:999px;}
.meter .avg {position:absolute; top:-6px; width:3px; height:26px; background:#2B2B2B; border-radius:2px;}
.pill {display:inline-block; padding:4px 12px; border-radius:999px; font-size:.84rem; font-weight:700; margin:6px 0 4px;}
.pill.bad {background:#F6E0D4; color:#8E3F1B;}
.pill.good {background:#D6EEEF; color:#0B5F65;}
.pill.mid {background:#F3EAD3; color:#6E5212;}
.kv {display:flex; justify-content:space-between; gap:12px; padding:6px 0; border-bottom:1px solid #F0EBE4; font-size:.93rem;}
.kv:last-child {border-bottom:none;}
.kv .k {color:#6F675E;}
.kv .v {font-weight:700; color:#2B2B2B; text-align:right;}
.dot {display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:7px; vertical-align:middle;}
.num {display:inline-flex; align-items:center; justify-content:center; width:30px; height:30px; border-radius:50%; background:#C4663A; color:#FFFFFF; font-weight:800; margin-right:10px;}
[data-testid="stVegaLiteChart"] {background:#FFFFFF; border:1px solid #ECE6DD; border-radius:16px; padding:6px 4px 2px;}
[data-testid="stImage"] img {border-radius:14px; border:1px solid #ECE6DD; background:#FFFFFF;}
.stTabs [data-baseweb="tab-list"] {gap:8px;}
.stTabs [data-baseweb="tab"] {font-weight:600; font-size:1rem;}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data
@st.cache_data
def js(name):
    p = TAB / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def csv(name):
    p = TAB / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def products():
    p = pd.read_parquet(PROC / "vis_products.parquet").sort_values("external_code").reset_index(drop=True)
    U, D = p[U_COLS].to_numpy(float), p[D_COLS].to_numpy(float)
    p["stock_never_negative"] = ((D.cumsum(1) - U.cumsum(1)) >= 0).all(1)  # clean examples for the product view
    return p


@st.cache_resource
def risk_model():
    return joblib.load(MODEL)


@st.cache_data
def risk_cutoffs():
    """Top-quartile cut-offs used by the odds-ratio model (price position, first-order depth), as in src/vis_risk_model.py."""
    p = products()
    p = p[p.delivered_12w > 0]
    tier = (p.groupby("category").price.rank(method="average") - 0.5) / p.groupby("category").price.transform("size")
    return float(tier.quantile(0.75)), float((p.dl0 / p.n_shops).quantile(0.75))


def photo(code, season):
    f = IMG / f"{'PE' if season.startswith('SS') else 'AI'}{season[2:]}" / f"{int(code):05d}.png"
    return f if f.exists() else None


# ----------------------------------------------------------------------------- small helpers
def init(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def go(i):
    st.session_state.step = i


def next_item(key):
    st.session_state[key] = st.session_state.get(key, 0) + 1


def preset(values):
    st.session_state.update(values)


def demo_order(score):
    """Most typical item first (lowest score), then the rest in a fixed random order, so rehearsals repeat."""
    first = score.idxmin()
    return [first, *np.random.default_rng(42).permutation(score.index.drop(first).to_numpy())]


def picker(key, filters, order):
    if st.session_state.get(key + "_filters") != filters:
        st.session_state[key + "_filters"] = filters
        st.session_state[key] = 0
    return order[st.session_state.get(key, 0) % len(order)]


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


def signed(x, d=1):
    return f"{x * 100:+.{d}f}%".replace("-", "−")


def n0(x):
    return f"{x:,.0f}"


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def header(kicker, question, sub=None):
    html(f'<div class="kicker">{kicker}</div><div class="question">{question}</div>'
         + (f'<div class="sub">{sub}</div>' if sub else ""))


def kpis(items):
    for col, (value, label) in zip(st.columns(len(items)), items):
        col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>',
                     unsafe_allow_html=True)


def take(text):
    html(f'<div class="take">💡 {text}</div>')


def read(text):
    html(f'<div class="read">👀 <b>How to read it:</b> {text}</div>')


def kv(pairs):
    return "".join(f'<div class="kv"><span class="k">{k}</span><span class="v">{v}</span></div>' for k, v in pairs)


def method(text, syllabus):
    with st.expander("🔧 How we did it"):
        st.markdown(text)
        st.caption(syllabus)


def numbers(df):
    with st.expander("📋 Show the numbers"):
        st.dataframe(df, hide_index=True, width="stretch")


def notes(say, click=None):
    if st.session_state.get("notes"):
        s = f'<div class="say"><div class="t">🎤 SAY</div>{say}'
        if click:
            s += f'<div class="t" style="margin-top:10px">🖱️ CLICK</div>{click}'
        html(s + "</div>")


def show(chart, height=300):
    chart = (chart.properties(height=height)
             .configure(font=FONT, background="#FFFFFF", padding={"left": 10, "right": 16, "top": 12, "bottom": 8})
             .configure_axis(labelColor=MUTED, titleColor=MUTED, labelFontSize=12, titleFontSize=12,
                             titleFontWeight="normal", gridColor=GRID, domainColor="#CFC8BE", tickColor="#CFC8BE", labelLimit=280)
             .configure_legend(orient="top", labelColor=TEXT, labelFontSize=12.5, symbolSize=150, labelLimit=320)
             .configure_view(stroke=None)
             .configure_title(fontSize=15, color=INK, anchor="start", fontWeight=600, offset=12))
    st.altair_chart(chart, width="stretch", theme=None)


def week_axis(title="Week since release"):
    return alt.X("Week:Q", scale=alt.Scale(domain=[0.6, K_WEEKS + 0.4], nice=False),
                 axis=alt.Axis(values=WEEKS, format="d", title=title, grid=False))


# ----------------------------------------------------------------------------- step 0: start
def step_start():
    vis, lc, cs, sv, fs, rs = (js("vis_summary"), js("vis_lifecycle_summary"), js("vis_customers_summary"),
                               js("survey_descriptives"), js("vis_financials_summary"), js("vis_risk_model_summary"))
    stock, am, sim, gp = vis["stock"], lc["all_products_markdown"], vis["simulation_main_scenario"], fs["policy_gross_profit_change"]
    chips = '<span class="arrow">→</span>'.join(f'<span class="chip{" last" if w == "WASTE" else ""}">{w}</span>'
                                                for w in ["TREND", "PURCHASE", "DEMAND", "INVENTORY", "WASTE"])
    html('<div class="hero"><div class="kicker">Business Data Analytics · project demo</div>'
         '<div class="htitle">From Trend to Waste</div>'
         "<p>Fast fashion's environmental cost is linked to overproduction. We follow one real retailer's own data to the "
         "shop end of it: where excess stock builds up, how early it can be spotted, and what acting early is worth.</p>"
         f'<div class="flow">{chips}</div></div>')
    auc_2wk = max(rs["auc_after_2_weeks_lr"], rs["auc_after_2_weeks_gb"])
    notes(f"We studied one real fast-fashion retailer, Nuna Lie, using its own sales, deliveries, prices and loyalty-card "
          f"data, plus our own survey of {sv['respondents'] if sv else '—'} people. Our question: where does excess stock come "
          "from, can it be spotted early, and is acting early worth it? These five numbers are the answer. The demo shows "
          "where each one comes from.",
          "Next → (bottom of the page), or any stage card below.")
    st.markdown("##### The data: one retailer, everything linked")
    kpis([(n0(stock["products"]), "products over 6 seasons (2017-2019)"),
          (n0(stock["shops"]), "shops of Nuna Lie, an Italian fast-fashion chain"),
          (n0(cs["customers"]), "loyalty-card customers and their purchases"),
          (n0(sv["respondents"]) if sv else "—", "people answered our own survey")])
    st.caption("VISUELLE 2.0 links sales, deliveries, prices, discounts and loyalty-card purchases by product code, shop "
               "and date, so every retail result comes from the same retailer.")
    st.markdown("##### The answer in five numbers")
    kpis([(pct(stock["leftover_share_of_delivered"], 1), "of delivered stock is still unsold after 12 weeks"),
          (f"week {am['median_first_markdown_week']:.0f}",
           f"first markdown (median), when demand is already down to {pct(am['median_demand_at_markdown_vs_peak'])} of its peak"),
          (f"AUC {auc_2wk:.2f}", f"excess stock is predictable after 2 weeks ({rs['auc_with_plan_gb']:.2f} before launch; "
                                 "0.5 = a coin toss)"),
          (signed(sim["leftover_change"]), f"unsold stock if the retailer acts at week 2 (revenue {signed(sim['revenue_change'])})"),
          (signed(gp["cost50"]), f"gross profit from acting early at a 50% unit cost ({signed(gp['cost40'])} to "
                                 f"{signed(gp['cost60'])} for 40-60%)")])
    st.markdown("##### The demo, step by step")
    cols = st.columns(4)
    for i, (title, desc, icon) in enumerate(STEPS[1:], start=1):
        with cols[(i - 1) % 4].container(border=True):
            st.markdown(f"**{icon} {title}**")
            st.caption(desc)
            st.button("Open →", key=f"open_{i}", on_click=go, args=(i,), width="stretch")


# ----------------------------------------------------------------------------- step 1: explore
def step_explore():
    p, lc, vis = products(), js("vis_lifecycle_summary"), js("vis_summary")
    am, stock = lc["all_products_markdown"], vis["stock"]
    header("Stage 1 · Exploratory and diagnostic analytics", "What happens to a product after it reaches the shops?",
           "Four views: one real product, the four demand shapes, markdown timing, and the leftover stock across seasons.")
    notes("① One real product: the bars are weekly sales, coloured when the product was on markdown. Sales peak early, the "
          "discount comes late, and the shaded gap is stock still sitting in the shops. ② Products follow four demand "
          f"shapes, but {pct(am['share_peak_in_first_3_weeks'])} sell best in their first three weeks. ③ Across all products, "
          f"markdowns build up only as demand fades: the first one comes in week {am['median_first_markdown_week']:.0f}. "
          f"④ It happens every season: {pct(stock['leftover_share_of_delivered'], 1)} of stock is left after 12 weeks.",
          "Tab ①: 🎲 Show another product, then pick the shape 'Late bloomer'. Tab ③: move the markdown slider to 10% and 20%: "
          "the pattern holds.")
    t1, t2, t3, t4 = st.tabs(["① Follow one product", "② Four demand shapes", "③ Markdowns come late", "④ Seasons and leftover"])
    with t1:
        follow_one_product(p, am, stock)
    with t2:
        four_shapes(p, am, lc)
    with t3:
        markdowns_late(p)
    with t4:
        seasons_leftover(p, stock)
    method(f"- **Data:** VISUELLE 2.0, every product's first {K_WEEKS} weeks in every shop: weekly sales, deliveries and "
           "discounts. Negative sales (returns) are set to 0; stock = deliveries so far − sales so far.\n"
           f"- **Demand shapes:** k-means clustering on each product's {K_WEEKS}-week sales curve, turned into shares so big and "
           f"small products compare ({lc['products_in_sample']:,} products with at least {lc['min_units']} units). "
           f"k = {lc['k_chosen']} was chosen with the silhouette score.\n"
           f"- **Markdown:** a week counts as a markdown week when the product's average discount across shops is at least "
           f"{MARKDOWN_THRESHOLD:.0%} (checked at 10% and 20%).",
           "Syllabus: descriptive and diagnostic analytics · Module 1 cluster analysis · Module 3 inventory management")


def follow_one_product(p, am, stock):
    pool = p[p.in_lifecycle_sample & p.stock_never_negative & (p.delivered_12w >= 100)]
    init("p1_shape", "Any")
    init("p1_cat", "All categories")
    c1, c2, c3 = st.columns([2.4, 1, 0.9], vertical_alignment="bottom")
    shape = c1.segmented_control("Demand shape", ["Any", *CLUSTER], key="p1_shape", required=True)
    cat = c2.selectbox("Category", ["All categories", *pool.category.value_counts().index[:10]], key="p1_cat")
    c3.button("🎲 Show another product", key="p1_next", on_click=next_item, args=("p1_n",), width="stretch")
    sub = pool if shape == "Any" else pool[pool.cluster_name == shape]
    sub = sub if cat == "All categories" else sub[sub.category == cat]
    if sub.empty:
        st.info("No product matches this combination. Pick another shape or category.")
        return
    marked = sub.first_markdown_week.notna()
    fmd_med = (sub.first_markdown_week[marked] + 1).median() if marked.any() else 0.0
    Us = sub[U_COLS].to_numpy(float)
    at_md = pd.Series(Us[np.arange(len(sub)), sub.first_markdown_week.fillna(0).astype(int)] / Us.max(1), index=sub.index)
    at_med = at_md[marked].median() if marked.any() else 0.0
    score = ((sub.sell_through - sub.sell_through.median()).abs() / 0.1
             + (sub.peak_week - sub.peak_week.median()).abs() / 2
             + pd.Series(np.where(marked, (sub.first_markdown_week + 1 - fmd_med).abs() / 3 + (at_md - at_med).abs() / 0.15, 3.0),
                         index=sub.index)
             + 1.0 * (sub.delivered_12w < 300) + 0.7 * ((sub[D_COLS] > 0).sum(axis=1) < 3))  # first pick: stock that builds up
    r = sub.loc[picker("p1_n", (shape, cat), demo_order(score))]

    U, D, dsc = (r[c].to_numpy(float) for c in (U_COLS, D_COLS, DSC_COLS))
    deliv, sold = D.sum(), U.sum()
    left, peak = deliv - sold, int(U.argmax()) + 1
    fmd = None if pd.isna(r.first_markdown_week) else int(r.first_markdown_week) + 1
    kpis([(n0(deliv), f"units delivered to {n0(r.n_shops)} shops"), (n0(sold), "units sold in 12 weeks"),
          (n0(left), f"units still unsold ({pct(left / deliv)} of the stock)"), (f"week {peak}", "best-selling week"),
          (f"week {fmd}" if fmd else "none", f"first markdown ({MARKDOWN_THRESHOLD:.0%}+ off)")])
    img_col, rest = st.columns([0.8, 2.7], gap="medium")
    with img_col:
        f = photo(r.external_code, r.season)
        if f:
            st.image(str(f), width="stretch", caption="Photo and labels as provided in VISUELLE 2.0")
        html('<div class="card">' + kv([("Product", f"#{int(r.external_code)}"), ("Category", r.category), ("Colour", r.color),
                                         ("Fabric", r.fabric), ("Season", r.season), ("Demand shape", r.cluster_name)]) + "</div>")
    a_col, b_col = rest.columns(2, gap="medium")
    wk = pd.DataFrame({"Week": WEEKS, "Units sold": U, "Average discount": dsc,
                       "Price": np.where(dsc >= MARKDOWN_THRESHOLD, MD_ON, MD_OFF)})
    bars = alt.Chart(wk).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=22).encode(
        x=week_axis(), y=alt.Y("Units sold:Q", title="Units sold that week"),
        color=alt.Color("Price:N", scale=alt.Scale(domain=[MD_OFF, MD_ON], range=[TEAL, TERRA]), legend=alt.Legend(title=None)),
        tooltip=["Week:Q", alt.Tooltip("Units sold:Q", format=",.0f"), alt.Tooltip("Average discount:Q", format=".0%")])
    best = alt.Chart(wk[wk.Week == peak]).mark_text(dy=-9, fontSize=12, fontWeight="bold", color=INK).encode(
        x="Week:Q", y="Units sold:Q", text=alt.value("best week"))
    with a_col:
        show(alt.layer(bars, best).properties(title="Units sold each week"), 285)
    cum = pd.DataFrame({"Week": WEEKS, "Delivered so far": D.cumsum(), "Sold so far": U.cumsum()})
    cum["Still in the shops"] = cum["Delivered so far"] - cum["Sold so far"]
    long = cum.melt("Week", ["Delivered so far", "Sold so far"], var_name="Line", value_name="Units")
    gap = alt.Chart(cum).mark_area(color=TERRA, opacity=0.16).encode(
        x=week_axis(), y="Sold so far:Q", y2="Delivered so far:Q",
        tooltip=["Week:Q", alt.Tooltip("Delivered so far:Q", format=",.0f"), alt.Tooltip("Sold so far:Q", format=",.0f"),
                 alt.Tooltip("Still in the shops:Q", format=",.0f")])
    lines = alt.Chart(long).mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=36, filled=True)).encode(
        x=week_axis(), y=alt.Y("Units:Q", title="Units, cumulative"),
        color=alt.Color("Line:N", scale=alt.Scale(domain=["Delivered so far", "Sold so far"], range=[INK, TEAL]),
                        legend=alt.Legend(title=None)),
        tooltip=["Line:N", "Week:Q", alt.Tooltip("Units:Q", format=",.0f")])
    end = pd.DataFrame({"Week": [K_WEEKS], "y": [(cum["Delivered so far"].iloc[-1] + cum["Sold so far"].iloc[-1]) / 2],
                        "t": [f"{n0(left)} unsold"]})
    lab = alt.Chart(end).mark_text(align="right", dx=-10, fontSize=12.5, fontWeight="bold", color="#8E3F1B").encode(
        x="Week:Q", y="y:Q", text="t:N")
    with b_col:
        show(alt.layer(gap, lines, lab).properties(title="Stock in the shops (the shaded gap)"), 285)
    story = f"This <b>{r.category}</b> sold best in <b>week {peak}</b>. "
    story += (f"Its first markdown came in <b>week {fmd}</b>, when weekly sales were at <b>{pct(U[fmd - 1] / U.max())}</b> "
              "of the best week. " if fmd else f"It was never marked down by {MARKDOWN_THRESHOLD:.0%} or more in its first "
              f"{K_WEEKS} weeks. ")
    story += f"After {K_WEEKS} weeks, <b>{n0(left)} of {n0(deliv)}</b> delivered units were still in the shops."
    rest.markdown(f'<div class="card">📖 {story}</div>', unsafe_allow_html=True)
    rest.caption("The first product shown is a typical one for your filters; 🎲 cycles through the rest in a fixed "
                 "random order (products with at least 100 units delivered).")
    take(f"Across all products: <b>{pct(am['share_peak_in_first_3_weeks'])}</b> sell best in their first three weeks, the "
         f"first markdown comes in <b>week {am['median_first_markdown_week']:.0f}</b> (median), and "
         f"<b>{pct(stock['leftover_share_of_delivered'], 1)}</b> of delivered stock is still unsold after {K_WEEKS} weeks.")


def four_shapes(p, am, lc):
    pool = p[p.in_lifecycle_sample]
    init("p2_cat", "All categories")
    left, right = st.columns([2.1, 1], gap="large")
    with right:
        cat = st.selectbox("Category", ["All categories", *pool.category.value_counts().index[:10]], key="p2_cat")
    sub = pool if cat == "All categories" else pool[pool.category == cat]
    S = sub[U_COLS].to_numpy(float)
    S = S / S.sum(1, keepdims=True)
    share = sub.cluster_name.value_counts(normalize=True)
    rows = []
    for name in CLUSTER:
        m = (sub.cluster_name == name).to_numpy()
        if m.any():
            rows += [{"Shape": name, "Week": w, "Share of 12-week sales": v} for w, v in zip(WEEKS, S[m].mean(0))]
    df = pd.DataFrame(rows)
    line = alt.Chart(df).mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=36, filled=True)).encode(
        x=week_axis(), y=alt.Y("Share of 12-week sales:Q", axis=alt.Axis(format=".0%"), title="Share of the product's 12-week sales"),
        color=alt.Color("Shape:N", scale=alt.Scale(domain=list(CLUSTER), range=list(CLUSTER.values())), legend=alt.Legend(title=None)),
        tooltip=["Shape:N", "Week:Q", alt.Tooltip("Share of 12-week sales:Q", format=".1%")])
    with left:
        show(line.properties(title=f"Average sales curve of each demand shape · {cat} ({len(sub):,} products)"), 360)
        read("each line is the average shape of one group of products: the share of a product's 12-week sales that "
             "came in each week. K-means found the groups by itself, without being told what to look for.")
    with right:
        for name, colour in CLUSTER.items():
            html(f'<div class="card" style="padding:10px 14px"><span class="dot" style="background:{colour}"></span><b>{name}</b>'
                 f' <span style="color:{MUTED}">· {pct(share.get(name, 0))} of products</span>'
                 f'<div style="font-size:.88rem;color:#4A443E;margin-top:3px">{SHAPE_TEXT[name]}</div></div>')
        st.caption(f"In this selection, {pct((sub.peak_week <= 3).mean())} of products sell best in weeks 1-3.")
    take(f"Whatever the shape, most demand comes early: <b>{pct(am['share_peak_in_first_3_weeks'])}</b> of products sell best "
         f"in their first three weeks. Only the late bloomers ({pct(lc['cluster_shares']['Late bloomer'])} of products) "
         "keep selling into the second half of the season.")


def markdowns_late(p):
    pool = p[p.in_lifecycle_sample]
    init("p3_thr", MARKDOWN_THRESHOLD)
    thr = st.select_slider("Count a week as 'on markdown' when the product's average discount is at least",
                           sorted(set(MARKDOWN_SENSITIVITY) | {MARKDOWN_THRESHOLD}), key="p3_thr", format_func=lambda v: f"{v:.0%}")
    U, dsc = pool[U_COLS].to_numpy(float), pool[DSC_COLS].to_numpy(float)
    on = dsc >= thr
    marked = on.any(1)
    fm = on.argmax(1)
    at_md = U[marked, fm[marked]] / np.maximum(U[marked].max(1), 1)  # demand in the first markdown week vs own peak
    med_week, med_demand = float(np.median(fm[marked])) + 1, float(np.median(at_md))
    kpis([(f"week {med_week:g}", "first markdown (median product)"),
          (pct(med_demand), "of the best week's demand is left by then"),
          (pct((at_md < 0.5).mean()), "of marked-down products were cut only after demand had halved"),
          (pct(marked.mean()), f"of products were marked down within {K_WEEKS} weeks")])
    df = pd.DataFrame({"Week": WEEKS * 2, "Share": np.r_[(U / U.max(1, keepdims=True)).mean(0), on.mean(0)],
                       "Line": ["Demand: % of the product's best week"] * K_WEEKS + ["Products on markdown"] * K_WEEKS})
    lines = alt.Chart(df).mark_line(strokeWidth=3, point=alt.OverlayMarkDef(size=44, filled=True)).encode(
        x=week_axis(), y=alt.Y("Share:Q", axis=alt.Axis(format=".0%"), title=None, scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Line:N", scale=alt.Scale(domain=["Demand: % of the product's best week", "Products on markdown"],
                                                  range=[TEAL, TERRA]), legend=alt.Legend(title=None)),
        tooltip=["Line:N", "Week:Q", alt.Tooltip("Share:Q", format=".0%")])
    mk = pd.DataFrame({"Week": [med_week], "y": [0.97], "t": [f"first markdown (median): week {med_week:g}"]})
    rule = alt.Chart(mk).mark_rule(color=INK, strokeWidth=1.5).encode(x="Week:Q")
    lab = alt.Chart(mk).mark_text(align="left", dx=6, fontSize=12.5, fontWeight="bold", color=INK).encode(x="Week:Q", y="y:Q", text="t:N")
    show(alt.layer(lines, rule, lab).properties(title=f"All {len(pool):,} products, week by week"), 340)
    read("the teal line is how much demand is left compared with each product's best week; the orange line is the share "
         "of products on markdown that week. Discounts climb only after demand has already fallen.")
    take(f"Markdowns arrive after demand has gone: by the time a typical product is first discounted (week {med_week:g}), it "
         f"has lost <b>{pct(1 - med_demand)}</b> of its best week's demand. Change the threshold: the pattern holds.")


def seasons_leftover(p, stock):
    ls = csv("vis_leftover_by_season").set_index("season").loc[SEASON_ORDER].reset_index()
    long = ls.melt("season", ["sold", "leftover"], var_name="k", value_name="Units")
    long["Stock"] = long.k.map({"sold": "Sold within 12 weeks", "leftover": "Unsold after 12 weeks"})
    long["o"] = (long.k == "leftover").astype(int)
    bars = alt.Chart(long).mark_bar(size=34, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("season:N", sort=SEASON_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Units:Q", title="Units delivered", axis=alt.Axis(format="~s")),
        color=alt.Color("Stock:N", scale=alt.Scale(domain=["Sold within 12 weeks", "Unsold after 12 weeks"], range=[TEAL, TERRA]),
                        legend=alt.Legend(title=None)),
        order=alt.Order("o:Q"), tooltip=["season:N", "Stock:N", alt.Tooltip("Units:Q", format=",.0f")])
    ls["top"], ls["t"] = ls.sold + ls.leftover, [f"{v:.0%} unsold" for v in ls.leftover_share]
    lab = alt.Chart(ls).mark_text(dy=-9, fontSize=12, color=INK, fontWeight="bold").encode(
        x=alt.X("season:N", sort=SEASON_ORDER), y="top:Q", text="t:N")
    si = csv("vis_seasonal_index")
    si["Month"], si["Index"] = si.month.map(lambda m: MONTHS[m - 1]), si.seasonal_index * 100
    si["Kind"] = np.where(si.Index == si.Index.max(), "Busiest month", np.where(si.Index == si.Index.min(), "Quietest month", "Other months"))
    mb = alt.Chart(si).mark_bar(size=20, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Month:N", sort=MONTHS, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Index:Q", title="Weekly purchases (100 = average week)"),
        color=alt.Color("Kind:N", scale=alt.Scale(domain=["Busiest month", "Quietest month", "Other months"], range=[TERRA, TEAL, SAND]),
                        legend=alt.Legend(title=None)),
        tooltip=["Month:N", alt.Tooltip("Index:Q", format=".0f")])
    hund = alt.Chart(pd.DataFrame({"y": [100]})).mark_rule(color=INK, strokeWidth=1).encode(y="y:Q")
    ext = alt.Chart(si[si.Kind != "Other months"]).mark_text(dy=-9, fontSize=12, fontWeight="bold", color=INK).encode(
        x=alt.X("Month:N", sort=MONTHS), y="Index:Q", text=alt.Text("Index:Q", format=".0f"))
    a, b = st.columns(2, gap="medium")
    with a:
        show(alt.layer(bars, lab).properties(title="Stock delivered each season: sold vs unsold"), 300)
    with b:
        show(alt.layer(mb, hund, ext).properties(title="Seasonality of customer purchases (2018-2019)"), 300)
    st_ = p.sell_through.dropna().clip(0.30, 1.0)  # a few products sold more than the window's deliveries: shown at 100%
    edges = np.round(np.arange(0.30, 1.0001, 0.05), 2)
    cnt, _ = np.histogram(st_, bins=edges)
    hist = pd.DataFrame({"lo": edges[:-1] + 0.002, "hi": edges[1:] - 0.002, "Products": cnt})  # small gap between bars
    hist["Range"] = [f"{x:.0%}-{y:.0%}" for x, y in zip(edges[:-1], edges[1:])]
    hist["Group"] = np.where(edges[1:] <= HIGH + 1e-9, f"Under {HIGH:.0%} sold: high leftover", f"{HIGH:.0%} or more sold")
    hb = alt.Chart(hist).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("lo:Q", title="Share of its delivered stock that the product sold in 12 weeks",
                scale=alt.Scale(domain=[0.3, 1.0], nice=False), axis=alt.Axis(format=".0%", values=[float(v) for v in edges[::2]])),
        x2="hi:Q",
        y=alt.Y("Products:Q", title="Products"), y2=alt.datum(0),
        color=alt.Color("Group:N", scale=alt.Scale(domain=[f"Under {HIGH:.0%} sold: high leftover", f"{HIGH:.0%} or more sold"],
                                                   range=[TERRA, TEAL]), legend=alt.Legend(title=None)),
        tooltip=["Range:N", alt.Tooltip("Products:Q", format=",.0f")])
    show(hb.properties(title="How much of its stock does each product sell? (sell-through)"), 270)
    read("each bar counts products by sell-through (sold ÷ delivered). The orange bars are the products we call "
         "<b>high leftover</b>: the ones Stage 2 tries to predict before launch.")
    take(f"About a third of stock is left every season ({pct(ls.leftover_share.min())}-{pct(ls.leftover_share.max())}), "
         f"although shelves rarely ran empty ({pct(stock['share_shop_weeks_ending_at_zero_stock'], 1)} of shop-weeks). "
         f"<b>{pct(stock['share_products_sell_through_below_60pct'], 1)}</b> of products sold under {HIGH:.0%} of their stock.")


# ----------------------------------------------------------------------------- step 2: predict
def nice_term(t):
    k, _, v = t.partition(" = ")
    if k == "season_type":
        return "Season: autumn-winter" if v == "AW" else "Season: spring-summer"
    if k == "price_q":
        return "Price: priciest quarter" if "priciest" in v else f"Price: {v}"
    if k == "depth_q":
        return "First order: deepest quarter" if "deepest" in v else f"First order: {v}"
    return {"category": "Category", "color": "Colour", "fabric": "Fabric"}.get(k, k) + f": {v}"


def step_predict():
    rs, m, orr, mt = js("vis_risk_model_summary"), risk_model(), csv("vis_risk_odds_ratios"), csv("vis_risk_model_metrics")
    base, d, lv = m["base_rate"], m["defaults"], m["levels"]
    auc_2wk = max(rs["auc_after_2_weeks_lr"], rs["auc_after_2_weeks_gb"])
    header("Stage 2 · Pre-launch risk prediction", "Before a product launches, can we tell if it will be left over?",
           f"A model learns from the 2017-2018 seasons which products ended with <b>high leftover</b> (under {HIGH:.0%} of "
           "their stock sold in 12 weeks), and is tested on the 2019 seasons it never saw.")
    safe = {"r_cat": "long sleeve", "r_col": "black", "r_fab": "georgette", "r_sea": "AW", "r_tier": 0.5, "r_month": 9,
            "r_shops": int(d["n_shops"]), "r_init": round(d["initial_per_shop"] * 2) / 2}
    risky = {"r_cat": "culottes", "r_col": "yellow", "r_fab": "scuba crepe", "r_sea": "SS", "r_tier": 0.9, "r_month": 3,
             "r_shops": int(d["n_shops"]), "r_init": 12.0}
    for k, v in safe.items():
        init(k, v)
    notes(f"Before launch a buyer only knows the design and the buying plan. The model turns that into a risk score. A basic "
          f"black top scores below the average of {pct(base)}; a yellow scuba-crepe trend item at the top of its price range, "
          "bought deep, scores far above. The bars show why. The honest part: before launch the signal is modest "
          f"(AUC {rs['auc_with_plan_gb']:.2f}), so it should set how deep the first order is, and week-2 sales decide the rest.",
          "Click '⚠️ Risky trend item', then lower 'Units per shop in the first delivery' and watch the score fall.")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("**① Describe the planned product**, or start from an example")
        b1, b2 = st.columns(2)
        b1.button("👕 Basic black top", on_click=preset, args=(safe,), width="stretch")
        b2.button("⚠️ Risky trend item", on_click=preset, args=(risky,), width="stretch")
        a, b = st.columns(2)
        cat = a.selectbox("Category", lv["category"], key="r_cat")
        col = b.selectbox("Colour", lv["color"], key="r_col")
        fab = a.selectbox("Fabric", lv["fabric"], key="r_fab")
        sea = b.selectbox("Season", lv["season_type"], key="r_sea",
                          format_func=lambda v: {"SS": "Spring-summer", "AW": "Autumn-winter"}.get(v, v))
        tier = a.slider("Price position in its category", 0.0, 1.0, step=0.05, key="r_tier",
                        help="0 = the cheapest product in its category, 1 = the priciest")
        month = b.selectbox("Release month", list(range(1, 13)), key="r_month", format_func=lambda v: MONTHS[v - 1])
        st.markdown("**② The buying plan**")
        c, e = st.columns(2)
        shops = c.slider("Number of shops", 1, 110, key="r_shops")
        first = e.slider("Units per shop in the first delivery", 1.0, 20.0, step=0.5, key="r_init")
    X = pd.DataFrame([{"category": cat, "color": col, "fabric": fab, "season_type": sea, "price_tier": tier,
                       "release_month": month, "log_shops": np.log(shops), "log_initial_per_shop": np.log1p(first)}])
    prob = float(m["pipe"].predict_proba(X[m["cats"] + m["num"]])[0, 1])
    level = "bad" if prob > base * 1.25 else "good" if prob < base * 0.8 else "mid"
    colour = {"bad": TERRA, "mid": GOLD, "good": TEAL}[level]
    advice = {"bad": "⚠️ Higher risk: buy a shallow first order and replenish only if week-1-2 sales are strong.",
              "mid": "About average: normal buying depth, then check the week-2 forecast.",
              "good": "✅ Lower risk than average: normal buying depth, still check the week-2 forecast."}[level]
    with right:
        html(f'<div class="card"><div class="kicker">Risk score</div><div class="big">{pct(prob)}</div>'
             f'<div style="color:#4A443E;margin-top:4px">chance this product ends with <b>high leftover</b> '
             f'(under {HIGH:.0%} of its stock sold in 12 weeks)</div>'
             f'<div class="meter"><div class="fill" style="width:{min(prob, 1) * 100:.1f}%;background:{colour}"></div>'
             f'<div class="avg" style="left:{base * 100:.1f}%"></div></div>'
             f'<div class="read">The black mark is the average product ({pct(base)}). This one: <b>{prob / base:.1f}×</b> the average.</div>'
             f'<span class="pill {level}">{advice}</span></div>')
    p75, d75 = risk_cutoffs()
    mine = {f"category = {cat}", f"color = {col}", f"fabric = {fab}", f"season_type = {sea}"}
    mine |= {"price_q = Q4 priciest"} if tier >= p75 else set()
    mine |= {"depth_q = Q4 deepest"} if first >= d75 else set()
    sig = orr[(orr.p_value < 0.05) & ~orr.term.str.contains("rare")].copy()
    sig["Feature"] = sig.term.map(nice_term)
    sig["Change in the odds"] = sig.odds_ratio - 1
    sig["Effect"] = np.where(sig.odds_ratio > 1, "Raises the risk", "Lowers the risk")
    sig["Your product"] = np.where(sig.term.isin(mine), "◀ your product", "")
    sig["tx"] = sig["Change in the odds"].clip(lower=0)  # tag sits right of the bar (or of zero, for bars going left)
    order = sig.sort_values("Change in the odds", ascending=False).Feature.tolist()
    ob = alt.Chart(sig).mark_bar(size=18, cornerRadiusEnd=4).encode(
        y=alt.Y("Feature:N", sort=order, title=None), x=alt.X("Change in the odds:Q", axis=alt.Axis(format="+.0%"), title="Change in the odds of high leftover"),
        color=alt.Color("Effect:N", scale=alt.Scale(domain=["Raises the risk", "Lowers the risk"], range=[TERRA, TEAL]), legend=alt.Legend(title=None)),
        opacity=alt.condition(alt.datum["Your product"] != "", alt.value(1.0), alt.value(0.45)),
        tooltip=["Feature:N", alt.Tooltip("odds_ratio:Q", title="Odds ratio", format=".2f"),
                 alt.Tooltip("ci_low:Q", title="95% CI low", format=".2f"), alt.Tooltip("ci_high:Q", title="95% CI high", format=".2f")])
    tag = alt.Chart(sig[sig["Your product"] != ""]).mark_text(align="left", dx=6, fontWeight="bold", color=INK, fontSize=12).encode(
        y=alt.Y("Feature:N", sort=order), x="tx:Q", text="Your product:N")
    MODELS = ["Logistic regression", "Gradient boosting"]
    keep = mt[mt.model.isin(MODELS)].copy()
    keep["Stage"] = keep.stage.map({"Before launch: product only": "Product only", "Before launch: product + buying plan": "+ buying plan",
                                    "After 2 weeks of sales": "After 2 weeks"})
    stages = ["Product only", "+ buying plan", "After 2 weeks"]
    ab = alt.Chart(keep).mark_bar(size=22, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Stage:N", sort=stages, title=None, axis=alt.Axis(labelAngle=0)), xOffset=alt.XOffset("model:N", sort=MODELS),
        y=alt.Y("roc_auc:Q", title="AUC on the unseen 2019 products", scale=alt.Scale(domain=[0.5, 0.75])), y2=alt.datum(0.5),
        color=alt.Color("model:N", scale=alt.Scale(domain=MODELS, range=[TEAL, TERRA]),
                        legend=alt.Legend(title=None)),
        tooltip=["model:N", "Stage:N", alt.Tooltip("roc_auc:Q", title="AUC", format=".2f")])
    at = alt.Chart(keep).mark_text(dy=-8, fontSize=11.5, color=INK).encode(
        x=alt.X("Stage:N", sort=stages), xOffset=alt.XOffset("model:N", sort=MODELS), y="roc_auc:Q",
        text=alt.Text("roc_auc:Q", format=".2f"))
    a, b = st.columns([1.25, 1], gap="large")
    with a:
        show(alt.layer(ob, tag).properties(title="③ What pushes the risk up or down (significant features only)"), 330)
        ref = rs["reference_levels"]
        read(f"each bar compares one feature with a reference product (a {ref['color']} {ref['category']} in {ref['fabric']}, "
             "spring-summer, middle price and order depth). Bars matching your product are solid.")
    with b:
        show(alt.layer(ab, at).properties(title="④ How good is the model?"), 330)
        read("pick one product that ended with high leftover and one that did not: AUC is how often the model gives the "
             "first one the higher risk. 0.5 = a coin toss, 1.0 = always right.")
    take(f"Before launch the signal is modest (AUC <b>{rs['auc_with_plan_gb']:.2f}</b> with the buying plan): good for "
         f"deciding how deep the first order is, not for yes/no decisions. Two weeks of sales lift it to <b>{auc_2wk:.2f}</b>, "
         "so: buy shallow, then decide at week 2.")
    method(f"- **Models:** logistic regression, a decision tree and gradient boosting, trained on SS17-AW18 and tested on "
           f"SS19 + AW19 ({rs['n_test']:,} products). The decision tree was the weakest (AUC {rs['auc_tree']:.2f}).\n"
           "- **The score above** comes from the logistic regression with the buying plan (unweighted, so its "
           "probabilities are calibrated).\n"
           "- **Odds ratios** come from a logistic regression on all products; only features significant at 5% are shown.",
           "Syllabus: Module 1 logistic regression · predictive analytics")


# ----------------------------------------------------------------------------- step 3: forecast
def step_forecast():
    vis, pr, wkf, fm = js("vis_summary"), csv("vis_test_leftover_predictions"), csv("vis_test_weekly_forecast"), csv("vis_forecast_metrics")
    lp, flag_at = vis["leftover_prediction"], vis["simulation_main_scenario"]["flag_threshold_pred_sell_through"]
    header("Stage 3 · Week-2 demand forecasting", "After two weeks on sale, how much will a product sell by week 12?",
           "The model sees only weeks 1-2, then forecasts weeks 3-12. It is tested on the 2019 seasons, which it never saw.")
    notes(f"At week 2 we have real sales. The model forecasts the rest of the season; the grey line is the naive guess that "
          f"week 2 simply repeats. Weekly error falls from {pct(vis['forecast_naive_weekly_wape'])} to "
          f"{pct(vis['forecast_best_weekly_wape'])}. A product predicted to sell under {flag_at:.0%} of its stock is flagged: "
          f"the flag catches {pct(lp['high_leftover_recall'])} of the products that really end badly.",
          "🎲 Show another product. Then move the watchlist slider up to 70%: more products are caught, but more are false alarms.")
    if wkf is None or pr is None:
        st.warning("Weekly forecasts not found: run `python -m src.vis_forecast` (or `src.run_all`).")
        return
    pr = pr.dropna(subset=["sell_through", "pred_sell_through"]).set_index("external_code")
    pr["flagged"] = pr.pred_sell_through < flag_at
    init("f_set", "Flagged at week 2")
    c1, c2 = st.columns([2.2, 1], vertical_alignment="bottom")
    which = c1.segmented_control("Products", ["Flagged at week 2", "All 2019 products"], key="f_set", required=True)
    c2.button("🎲 Show another product", key="f_next", on_click=next_item, args=("f_n",), width="stretch")
    sub = pr[pr.flagged] if which.startswith("Flagged") else pr
    sub = sub[sub.delivered_12w >= 100]
    score = ((sub.pred_sell_through - sub.sell_through).abs() + 0.5 * (sub.delivered_12w < 300)
             + (0.5 * (sub.sell_through >= HIGH) if which.startswith("Flagged") else 0))
    code = picker("f_n", which, demo_order(score))
    r = pr.loc[code]
    U = products().set_index("external_code").loc[code, U_COLS].to_numpy(float)
    f = wkf[wkf.external_code == code].sort_values("week")
    actual = pd.DataFrame({"Week": WEEKS, "Units": U, "Line": "Actual sales"})
    fc = pd.DataFrame({"Week": [2, *f.week], "Units": [U[1], *f.forecast], "Line": "Forecast made at week 2"})
    nv = pd.DataFrame({"Week": [2, *f.week], "Units": [U[1], *f.naive], "Line": "Naive: week 2 repeats"})
    df = pd.concat([actual, fc, nv], ignore_index=True)
    lines_ = ["Actual sales", "Forecast made at week 2", "Naive: week 2 repeats"]
    band = alt.Chart(pd.DataFrame({"a": [0.6], "b": [2.4]})).mark_rect(color=SAND, opacity=0.45).encode(x="a:Q", x2="b:Q")
    seen = alt.Chart(pd.DataFrame({"Week": [1.5], "t": ["the model|sees these|2 weeks"]})).mark_text(
        baseline="bottom", fontSize=11, color=MUTED, dy=-6, lineBreak="|").encode(x="Week:Q", y=alt.datum(0), text="t:N")
    ln = alt.Chart(df).mark_line(strokeWidth=2.8, point=alt.OverlayMarkDef(size=40, filled=True)).encode(
        x=week_axis(), y=alt.Y("Units:Q", title="Units sold per week"),
        color=alt.Color("Line:N", scale=alt.Scale(domain=lines_, range=[INK, TERRA, GREY]), legend=alt.Legend(title=None)),
        strokeDash=alt.StrokeDash("Line:N", scale=alt.Scale(domain=lines_, range=[[1, 0], [1, 0], [5, 4]]), legend=None),
        tooltip=["Line:N", "Week:Q", alt.Tooltip("Units:Q", format=",.0f")])
    a, b = st.columns([2.1, 1], gap="large")
    with a:
        show(alt.layer(band, seen, ln).properties(title=f"Product #{int(code)} · {r.category} · {r.season}"), 330)
    with b:
        img = photo(code, r.season)
        if img:
            st.image(str(img), width=150)
        pill = (f'<span class="pill bad">⚑ Flagged at week 2: act now</span>' if r.flagged
                else '<span class="pill good">Not flagged</span>')
        rmd = "—" if pd.isna(r.recommended_markdown_week) else f"week {r.recommended_markdown_week:.0f}"
        amd = "none" if pd.isna(r.actual_first_markdown_week) else f"week {r.actual_first_markdown_week:.0f}"
        html(f'<div class="card">{pill}' + kv([("Predicted to sell", f"{pct(r.pred_sell_through)} of its stock"),
                                               ("Actually sold", f"{pct(r.sell_through)} of its stock"),
                                               ("Predicted unsold", f"{n0(r.pred_leftover)} units"),
                                               ("Actually unsold", f"{n0(r.leftover_12w)} units"),
                                               ("Suggested markdown from", rmd), ("Actual first markdown", amd)]) + "</div>")
    st.markdown("##### How good is it across all 2019 products?")
    a, b = st.columns([1.1, 1], gap="large")
    allp = fm[fm.group == "All products"].sort_values("weekly_wape")
    allp["Kind"] = np.where(allp.model == vis["forecast_best_model"], "Best model", "Other models")
    wb = alt.Chart(allp).mark_bar(size=20, cornerRadiusEnd=4).encode(
        y=alt.Y("model:N", sort=allp.model.tolist(), title=None), x=alt.X("weekly_wape:Q", axis=alt.Axis(format=".0%"), title="Weekly forecast error (WAPE), lower is better"),
        color=alt.Color("Kind:N", scale=alt.Scale(domain=["Best model", "Other models"], range=[TERRA, SAND]), legend=None),
        tooltip=["model:N", alt.Tooltip("weekly_wape:Q", title="WAPE", format=".1%")])
    wt = alt.Chart(allp).mark_text(align="left", dx=5, fontSize=12, color=INK).encode(
        y=alt.Y("model:N", sort=allp.model.tolist()), x="weekly_wape:Q", text=alt.Text("weekly_wape:Q", format=".0%"))
    with a:
        show(alt.layer(wb, wt).properties(title="Forecast error by model, weeks 3-12"), 250)
        read("WAPE = total forecast error ÷ total sales. The naive method assumes week 2 simply repeats.")
    with b:
        init("f_thr", flag_at)
        t = st.select_slider("Watchlist: flag a product when its predicted sell-through is below",
                             [round(x, 2) for x in np.arange(0.40, 0.801, 0.05)], key="f_thr", format_func=lambda v: f"{v:.0%}")
        flag, bad = pr.pred_sell_through < t, pr.sell_through < HIGH
        caught = (flag & bad).sum() / bad.sum()
        right_ = (flag & bad).sum() / flag.sum() if flag.any() else float("nan")
        html('<div class="card">' + kv([("Products flagged", f"{flag.sum():,} of {lp['test_products']:,}"),
                                        (f"Caught: products that really sold under {HIGH:.0%}", pct(caught)),
                                        ("Right: flagged products that really did", pct(right_) if flag.any() else "—")])
             + f'<div class="read" style="margin-top:8px">The naive method at {flag_at:.0%}: caught '
             f'{pct(lp["naive_flag_recall"])}, right {pct(lp["naive_flag_precision"])}.</div></div>')
    take(f"Two weeks of sales cut the weekly forecast error from <b>{pct(vis['forecast_naive_weekly_wape'])}</b> to "
         f"<b>{pct(vis['forecast_best_weekly_wape'])}</b>. At a {flag_at:.0%} threshold the flag catches "
         f"<b>{pct(lp['high_leftover_recall'])}</b> of the products that end badly, and is right {pct(lp['high_leftover_precision'])} "
         "of the time: week 2 is the moment to act.")
    method(f"- **Models:** naive, a category lifecycle profile, Poisson regression and gradient boosting, each fed weeks 1-2. "
           f"Google Trends was tested as an extra input and did not help, so the best model is "
           f"'{vis['forecast_best_model']}'.\n"
           "- **Training and test:** trained on SS17-AW18, tested on SS19 + AW19.\n"
           "- **The flag:** predicted sell-through = (weeks 1-2 sales + forecast) ÷ the product's deliveries. "
           "'Suggested markdown from' is the first week the forecast falls below half of the best week so far.",
           "Syllabus: Module 3 forecasting and demand planning · predictive analytics")


# ----------------------------------------------------------------------------- step 4: act
def step_act():
    vis, grid = js("vis_summary"), csv("vis_simulation_grid")
    main = vis["simulation_main_scenario"]
    base_left = vis["simulation_baseline_test_leftover_units"]
    pols = {"Cut later deliveries": "A: cut later deliveries", "Mark down earlier": "B: earlier markdown", "Both": "A+B"}
    header("Stage 4 · Inventory intervention simulation", "What if the retailer had acted at week 2?",
           "We replay the real 2019 seasons shop by shop, week by week. Only the products flagged at week 2 are changed.")
    a30 = grid[(grid.policy == pols["Cut later deliveries"]) & (grid.delivery_cut == main["delivery_cut"])
               & (grid.flag_threshold_pred_sell_through == main["flag_threshold_pred_sell_through"])
               & (grid.uplift_estimate == main["uplift_estimate"])].iloc[0]
    notes(f"Now we act on the flagged products: cut their later deliveries, mark them down earlier, or both. Replaying 2019 "
          f"with both (a {main['delivery_cut']:.0%} cut and a {main['markdown_level']:.0%} markdown from week 4), unsold stock "
          f"falls {pct(-main['leftover_change'], 1)} while revenue moves only {signed(main['revenue_change'])}. Cutting "
          f"deliveries alone is the cheapest lever: {signed(a30.leftover_change)} unsold stock for {signed(a30.revenue_change, 2)} revenue.",
          "Pick 'Cut later deliveries', then move the cut from 10% to 40%.")
    init("s_pol", "Both")
    init("s_cut", main["delivery_cut"])
    init("s_md", main["markdown_level"])
    c1, c2, c3 = st.columns([1.5, 1, 1], gap="medium")
    lab = c1.segmented_control("What should the retailer do with flagged products?", list(pols), key="s_pol", required=True)
    pol = pols[lab]
    cut = c2.select_slider("Cut their later deliveries by", sorted(grid[grid.policy == "A+B"].delivery_cut.unique()),
                           key="s_cut", format_func=lambda v: f"{v:.0%}", disabled=pol.startswith("B"))
    md = c3.select_slider("Markdown from week 4", sorted(grid[grid.policy == "A+B"].markdown_level.unique()),
                          key="s_md", format_func=lambda v: f"{v:.0%}", disabled=pol.startswith("A:"))
    with st.expander("⚙️ More settings"):
        init("s_thr", main["flag_threshold_pred_sell_through"])
        init("s_upl", main["uplift_estimate"])
        e1, e2 = st.columns(2)
        thr = e1.radio("Flag products whose predicted sell-through is below", sorted(grid.flag_threshold_pred_sell_through.unique()),
                       key="s_thr", format_func=lambda v: f"{v:.0%}", horizontal=True)
        upl = e2.radio("How much a discount lifts sales", sorted(grid.uplift_estimate.unique()), key="s_upl", horizontal=True,
                       help="Both are estimated from the data; 'FE (conservative)' is the main one.")
    q = grid[(grid.flag_threshold_pred_sell_through == thr) & (grid.uplift_estimate == upl)]
    row = q[q.policy == pol]
    if not pol.startswith("B"):
        row = row[row.delivery_cut == cut]
    if not pol.startswith("A:"):
        row = row[row.markdown_level == md]
    row = row.iloc[0]
    kpis([(signed(row.leftover_change), "unsold stock"), (signed(row.sold_change), "units sold"),
          (signed(row.revenue_change, 2), "revenue"), (n0(row.units_leftover_avoided), "units kept out of the unsold pile"),
          (pct(row.share_products_flagged), "of 2019 products flagged")])
    ba = pd.DataFrame({"Scenario": ["2019 as it happened", "With the week-2 rule"],
                       "Units unsold": [base_left, row.leftover_units], "Kind": ["Without", "With"]})
    bb = alt.Chart(ba).mark_bar(size=34, cornerRadiusEnd=4).encode(
        y=alt.Y("Scenario:N", sort=ba.Scenario.tolist(), title=None), x=alt.X("Units unsold:Q", axis=alt.Axis(format="~s"), title="Units unsold after 12 weeks"),
        color=alt.Color("Kind:N", scale=alt.Scale(domain=["Without", "With"], range=[GREY, TEAL]), legend=None),
        tooltip=["Scenario:N", alt.Tooltip("Units unsold:Q", format=",.0f")])
    bt = alt.Chart(ba).mark_text(align="right", dx=-8, color="#FFFFFF", fontWeight="bold", fontSize=13).encode(
        y=alt.Y("Scenario:N", sort=ba.Scenario.tolist()), x="Units unsold:Q", text=alt.Text("Units unsold:Q", format=",.0f"))
    curve = q[q.policy == pols["Cut later deliveries"]].assign(Policy="Cut later deliveries")
    both = q[(q.policy == "A+B") & (q.markdown_level == md)].assign(Policy=f"Both ({md:.0%} markdown)")
    only = q[(q.policy == pols["Mark down earlier"]) & (q.markdown_level == md)].assign(Policy=f"Markdown only ({md:.0%})")
    tr = pd.concat([curve, both, only], ignore_index=True)
    tr["Cut"], tr["Change"] = tr.delivery_cut, tr.leftover_change
    dom = ["Cut later deliveries", f"Both ({md:.0%} markdown)", f"Markdown only ({md:.0%})"]
    tl = alt.Chart(tr).mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=60, filled=True)).encode(
        x=alt.X("Cut:Q", axis=alt.Axis(format=".0%", values=[0, 0.1, 0.2, 0.3, 0.4]), title="Cut to later deliveries",
                scale=alt.Scale(domain=[-0.02, 0.42], nice=False)),
        y=alt.Y("Change:Q", axis=alt.Axis(format="+.0%"), title="Change in unsold stock"),
        color=alt.Color("Policy:N", scale=alt.Scale(domain=dom, range=[TEAL, TERRA, PURPLE]), legend=alt.Legend(title=None)),
        tooltip=["Policy:N", alt.Tooltip("Cut:Q", format=".0%"), alt.Tooltip("Change:Q", format="+.1%"),
                 alt.Tooltip("revenue_change:Q", title="Revenue", format="+.2%")])
    cur = alt.Chart(pd.DataFrame({"Cut": [0.0 if pol.startswith("B") else cut], "Change": [row.leftover_change]})).mark_point(
        size=420, shape="circle", filled=False, strokeWidth=2.5, color=INK).encode(x="Cut:Q", y="Change:Q")
    a, b = st.columns(2, gap="medium")
    with a:
        show(alt.layer(bb, bt).properties(title="Unsold stock in the 2019 seasons"), 220)
        imp = units_to_impact(row.units_leftover_avoided)
        html(f'<div class="card">🌍 About <b>{imp["tonnes_textile_low"]:.0f}-{imp["tonnes_textile_high"]:.0f} tonnes</b> of '
             f'textiles kept out of the unsold pile, or <b>{imp["tonnes_co2e_low"]:.0f}-{imp["tonnes_co2e_high"]:.0f} t CO₂e</b> '
             f'<span style="color:{MUTED}">(an estimate: {GARMENT_KG["low"]}-{GARMENT_KG["high"]} kg per garment × '
             f'{CO2E_PER_KG["low"]}-{CO2E_PER_KG["high"]:.0f} kg CO₂e per kg; it assumes the cut deliveries would not have '
             'been made).</span></div>')
    with b:
        show(alt.layer(tl, cur).properties(title="Every option we simulated (circle = your choice)"), 300)
    if pd.notna(row.breakeven_cost_to_price):
        st.caption(f"Gross profit rises whenever unit cost is above {pct(row.breakeven_cost_to_price, 1)} of full price, "
                   "which holds for any real retailer. Stage 5 puts it in money.")
    take(f"Acting early on about one product in ten cuts unsold stock by <b>{pct(-main['leftover_change'], 1)}</b> "
         f"(both levers) with revenue {signed(main['revenue_change'])}. The cheapest lever is cutting later deliveries: "
         f"<b>{signed(a30.leftover_change)}</b> unsold stock for {signed(a30.revenue_change, 2)} revenue.")
    method("- **Replay:** the 2019 seasons, shop by shop and week by week: stock = deliveries − sales, and sales can never "
           "exceed the stock on the shelf.\n"
           "- **Policies:** the products flagged at week 2 get their later deliveries cut and/or a discount from week 4.\n"
           f"- **Discount effect:** estimated from the data with a fixed-effects regression (+10 points of discount → "
           f"+{pct(np.expm1(vis['uplift']['beta_fe'] * 0.1), 1)} sales). Stock-outs are rare "
           f"({pct(vis['stock']['share_shop_weeks_ending_at_zero_stock'], 1)} of shop-weeks), so sales are a fair proxy for demand.\n"
           "- **Not modelled:** the later clearance value of leftover stock.",
           "Syllabus: Module 3 inventory management · prescriptive analytics (what-if simulation)")


# ----------------------------------------------------------------------------- step 5: money
def step_money():
    fs, pol, fsum = csv("vis_financial_statement").set_index("season"), csv("vis_financial_policy"), js("vis_financials_summary")
    a = fs.loc["All seasons"]
    header("Stage 5 · Financial evaluation (Module 4)", "What does excess stock cost, and what is acting early worth?",
           "Nuna Lie publishes no accounts, so we built season statements from the data itself. Prices are normalised, so "
           "money is in <b>index units</b> (ratios are unaffected). Unit cost is not in the data: we show a range.")
    init("m_cost", 0.5)
    cost = st.select_slider("Assumed unit cost, as a share of full price", fsum["cost_ratios_assumed"], key="m_cost",
                            format_func=lambda v: f"{v:.0%}")
    k = int(round(cost * 100))
    unsold_cost, cogs = cost * (a.full_price_value_delivered - a.full_price_value_sold), cost * a.full_price_value_sold
    notes(f"In money: the waterfall turns the full-price value of what was sold into gross profit. Markdowns take "
          f"{pct(a.markdown_rate, 1)}, but the stock left unsold costs more: {unsold_cost / a.markdown_cost:.1f}× the "
          f"markdowns at a {cost:.0%} unit cost. We do not know the real cost, so we show 40, 50 and 60% of price: at every "
          "level the week-2 rule raises gross profit.",
          "Move the unit cost between 40% and 60%.")
    kpis([(pct(a.sell_through), "of delivered stock sold (sell-through)"), (pct(a.markdown_rate, 1), "markdown cost ÷ full-price sales"),
          (pct(a.stock_to_sales), "unsold stock (at full price) ÷ revenue"), (f"{a.inventory_turnover_12w:.1f}×", "inventory turnover in 12 weeks"),
          (f"{a.days_of_inventory:.0f} days", "of inventory on hand")])
    gp, gp_after = a.net_revenue - cogs, a.net_revenue - cogs - unsold_cost
    steps = [("Full-price value|of goods sold", 0, a.full_price_value_sold, "Total"),
             ("Markdown|discounts", a.net_revenue, a.full_price_value_sold, "Cost"),
             ("Net|revenue", 0, a.net_revenue, "Total"),
             ("Cost of|goods sold", gp, a.net_revenue, "Cost"),
             ("Gross profit|on goods sold", 0, gp, "Total"),
             ("Cost of stock|left unsold", gp_after, gp, "Cost"),
             ("Gross profit after|unsold stock", 0, gp_after, "Result")]
    wf = pd.DataFrame(steps, columns=["Step", "start", "end", "Kind"])
    wf["Amount"] = wf.end - wf.start
    wf["t"] = [f"{v / 1000:.1f}k" if kd != "Cost" else f"−{v / 1000:.1f}k" for v, kd in zip(wf.Amount, wf.Kind)]
    wf["Share of net revenue"] = wf.Amount / a.net_revenue
    order = wf.Step.tolist()
    wb = alt.Chart(wf).mark_bar(size=46, cornerRadius=3).encode(
        x=alt.X("Step:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelExpr="split(datum.label, '|')", labelFontSize=11.5)),
        y=alt.Y("start:Q", title="Index units (prices are normalised)", axis=alt.Axis(format="~s")), y2="end:Q",
        color=alt.Color("Kind:N", scale=alt.Scale(domain=["Total", "Cost", "Result"], range=[GREY, TERRA, TEAL]), legend=None),
        tooltip=[alt.Tooltip("Step:N"), alt.Tooltip("Amount:Q", format=",.0f"), alt.Tooltip("Share of net revenue:Q", format=".1%")])
    wt = alt.Chart(wf).mark_text(dy=-8, fontSize=12, fontWeight="bold", color=INK).encode(
        x=alt.X("Step:N", sort=order), y="end:Q", text="t:N")
    rows = []
    for c in fsum["cost_ratios_assumed"]:
        col = f"gross_profit_after_unsold_cost{int(round(c * 100))}"
        before, after = pol[col].iloc[0], pol[col].iloc[1]
        rows += [{"Unit cost": f"{c:.0%} of price", "Scenario": "2019 as it happened", "Gross profit": before, "c": c, "t": ""},
                 {"Unit cost": f"{c:.0%} of price", "Scenario": "With the week-2 rule", "Gross profit": after, "c": c,
                  "t": f"{after - before:+,.0f} ({signed(after / before - 1)})".replace("-", "−")}]
    sens = pd.DataFrame(rows)
    sens["Chosen"] = np.isclose(sens.c, cost)
    sb = alt.Chart(sens).mark_bar(size=26, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Unit cost:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="Scenario:N",
        y=alt.Y("Gross profit:Q", title="Gross profit after unsold stock (index units)", axis=alt.Axis(format="~s")),
        color=alt.Color("Scenario:N", scale=alt.Scale(domain=["2019 as it happened", "With the week-2 rule"], range=[GREY, TEAL]),
                        legend=alt.Legend(title=None)),
        opacity=alt.condition(alt.datum.Chosen, alt.value(1.0), alt.value(0.35)),
        tooltip=["Unit cost:N", "Scenario:N", alt.Tooltip("Gross profit:Q", format=",.0f")])
    st_ = alt.Chart(sens[sens.t != ""]).mark_text(dy=-8, fontSize=11.5, fontWeight="bold", color=INK).encode(
        x="Unit cost:N", xOffset="Scenario:N", y="Gross profit:Q", text="t:N")
    c1, c2 = st.columns([1.35, 1], gap="large")
    with c1:
        show(alt.layer(wb, wt).properties(title=f"Where the money goes, all six seasons (unit cost {cost:.0%} of price)"), 360)
        read(f"from left to right: discounts turn full-price sales into revenue; the cost of the goods sold leaves the gross "
             f"profit ({pct(a[f'gross_margin_cost{k}'])} of revenue); paying for stock that was never sold cuts it to "
             f"{pct(a[f'margin_after_unsold_cost{k}'])}.")
    with c2:
        show(alt.layer(sb, st_).properties(title="The week-2 rule in money (2019 seasons)"), 360)
        read("each pair compares 2019 as it happened with the week-2 rule, at one assumed unit cost. The label is the gain in "
             "index units and in %.")
    take(f"Unsold stock is the biggest hidden cost: at a {cost:.0%} unit cost it costs <b>{unsold_cost / a.markdown_cost:.1f}×</b> "
         f"as much as all the markdown discounts. The week-2 rule raises gross profit at every assumed cost "
         f"(<b>{signed(fsum['policy_gross_profit_change']['cost40'])} to {signed(fsum['policy_gross_profit_change']['cost60'])}</b>), "
         f"and does so whenever unit cost is above {pct(fsum['breakeven_cost_to_price'], 1)} of price.")
    numbers(fs.reset_index()[["season", "units_delivered", "sell_through", "net_revenue", "markdown_rate", "stock_to_sales",
                              "inventory_turnover_12w", "days_of_inventory", f"gross_margin_cost{k}", f"margin_after_unsold_cost{k}"]]
            .round(3))
    method("- **Revenue** = price × (1 − discount) × units sold; **markdown cost** = full-price value sold − revenue.\n"
           "- **Gross profit on goods sold** = revenue − unit cost × full-price value sold.\n"
           "- **After unsold stock** = revenue − unit cost × everything delivered (the retailer paid for it all). The true "
           "margin lies between the two, since leftover stock is usually cleared later at some price, which the data does "
           "not show.",
           "Syllabus: Module 4 analysis of financial statements")


# ----------------------------------------------------------------------------- step 6: customers
def step_customers():
    seg, fq, cs = csv("vis_customer_segments"), csv("vis_customers_by_frequency"), js("vis_customers_summary")
    header("Customers · Who buys, and how?", "Which customers matter most, and do they buy trends or discounts?",
           f"{cs['customers']:,} loyalty-card customers, December 2016 to February 2020, linked to the same products and shops.")
    top = seg.sort_values("cltv_index", ascending=False).iloc[0]
    notes(f"Who buys? One segment, '{top.segment}', is {pct(top.share_of_customers)} of customers but buys "
          f"{pct(top.share_of_units)} of units. Customers who buy more often buy a bigger share on markdown "
          f"({pct(cs['markdown_share_lowest_band'])} → {pct(cs['markdown_share_highest_band'])}), but not more short-lived "
          "trend products: our hypothesis was not supported, and we report that.",
          "Pick each segment in the box to see its profile.")
    kpis([(n0(cs["customers"]), f"customers ({cs['card_ids_excluded']} shop cards excluded)"), (pct(cs["share_units_top20pct_customers"]), "of units bought by the top 20%"),
          (pct(cs["share_one_purchase_day"]), "bought on a single day only"), (pct(cs["markdown_share_all_known_lines"]), "of purchases were on markdown")])
    names = seg.sort_values("share_of_units", ascending=False).segment.tolist()
    long = seg.melt("segment", ["share_of_customers", "share_of_units"], var_name="m", value_name="Share")
    long["Measure"] = long.m.map({"share_of_customers": "Share of customers", "share_of_units": "Share of units bought"})
    sb = alt.Chart(long).mark_bar(size=16, cornerRadiusEnd=4).encode(
        y=alt.Y("segment:N", sort=names, title=None), yOffset="Measure:N", x=alt.X("Share:Q", axis=alt.Axis(format=".0%"), title=None),
        color=alt.Color("Measure:N", scale=alt.Scale(domain=["Share of customers", "Share of units bought"], range=[TEAL, TERRA]),
                        legend=alt.Legend(title=None)),
        tooltip=["segment:N", "Measure:N", alt.Tooltip("Share:Q", format=".0%")])
    stx = alt.Chart(long).mark_text(align="left", dx=4, fontSize=11.5, color=INK).encode(
        y=alt.Y("segment:N", sort=names), yOffset="Measure:N", x="Share:Q", text=alt.Text("Share:Q", format=".0%"))
    a, b = st.columns([1.4, 1], gap="large")
    with a:
        show(alt.layer(sb, stx).properties(title="Customer segments (k-means on recency, frequency and spend)"), 330)
    with b:
        init("c_seg", names[0])
        pick = st.selectbox("Segment profile", names, key="c_seg")
        s = seg.set_index("segment").loc[pick]
        rank = int(seg.cltv_index.rank(ascending=False)[seg.segment == pick].iloc[0])
        html('<div class="card">' + kv([("Customers", f"{s.customers:,.0f} ({pct(s.share_of_customers)})"),
                                        ("Share of units · of spend", f"{pct(s.share_of_units)} · {pct(s.share_of_spend)}"),
                                        ("Purchase days (median)", f"{s.purchase_days_median:.0f}"),
                                        ("Days since last purchase (median)", f"{s.recency_median:.0f}"),
                                        ("Items per basket", f"{s.items_per_basket_mean:.1f}"),
                                        ("Bought on markdown", pct(s.markdown_share_mean)),
                                        ("Bought again the next year", pct(s.retention_12m)),
                                        ("Lifetime value (CLTV, index)", f"{s.cltv_index:.2f} · #{rank} of {len(seg)}")]) + "</div>")
    fl = fq.melt("freq_band", ["markdown_share", "short_lived_share"], var_name="m", value_name="Share")
    fl["What they bought"] = fl.m.map({"markdown_share": "Bought on markdown", "short_lived_share": "Short-lived (trend) products"})
    bands = fq.freq_band.tolist()
    fb = alt.Chart(fl).mark_bar(size=22, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("freq_band:N", sort=bands, title="How often the customer buys", axis=alt.Axis(labelAngle=0),
                scale=alt.Scale(paddingInner=0.6)), xOffset="What they bought:N",
        y=alt.Y("Share:Q", axis=alt.Axis(format=".0%"), title="Share of their purchases"),
        color=alt.Color("What they bought:N", scale=alt.Scale(domain=["Bought on markdown", "Short-lived (trend) products"], range=[TERRA, TEAL]),
                        legend=alt.Legend(title=None)),
        tooltip=["freq_band:N", "What they bought:N", alt.Tooltip("Share:Q", format=".1%")])
    ft = alt.Chart(fl).mark_text(dy=-8, fontSize=11.5, color=INK).encode(
        x=alt.X("freq_band:N", sort=bands), xOffset="What they bought:N", y="Share:Q", text=alt.Text("Share:Q", format=".0%"))
    show(alt.layer(fb, ft).properties(title="Do frequent buyers buy trends, or discounts?"), 300)
    take(f"A loyal core buys most: <b>{top.segment}</b> customers are {pct(top.share_of_customers)} of customers but "
         f"<b>{pct(top.share_of_units)}</b> of units. Frequent buyers lean on markdowns ({pct(cs['markdown_share_lowest_band'])} → "
         f"{pct(cs['markdown_share_highest_band'])}), but short-lived products stay at {pct(cs['short_lived_share_lowest_band'], 1)}-"
         f"{pct(cs['short_lived_share_highest_band'], 1)} in every band: the hypothesis 'frequent shoppers buy more trend items' "
         "is <b>not supported</b>.")
    method(f"- **Customers:** IDs that buy in more than {CUSTOMER_MAX_SHOPS} shops are shop or generic cards, not people, so "
           "they are excluded.\n"
           "- **Segments:** k-means on recency, purchase days, spend per basket and items per basket (RFM), with the number of "
           "segments chosen by silhouette score.\n"
           "- **CLTV** = last-12-month spend × expected lifetime from the 12-month retention rate (index units: compare "
           "segments only). This is association, not cause and effect.",
           "Syllabus: Module 1 cluster analysis, RFM and customer lifetime value")


# ----------------------------------------------------------------------------- step 7: consumers
def step_consumers():
    sv, pw = js("survey_descriptives"), csv("survey_conjoint_partworths")
    header("Consumers · What happens after purchase?", "How long do people keep clothes, and what would they pay for?",
           "Sales data stops at the till, so we ran our own survey (Google Form).")
    if sv is None or pw is None:
        st.info("Survey results not found: run `python -m src.survey_analysis` on the real responses.")
        return
    design = pd.read_csv(ROOT / "survey" / "conjoint_design.csv")
    ratings = sv["mean_rating_by_profile"]
    p1, p6 = design.set_index("profile").loc["P1"], design.set_index("profile").loc["P6"]
    notes(f"{sv['respondents']} people answered. They keep clothes about {sv['mean_keep_years']:.1f} years, and "
          f"{pct(sv['share_wears_under_10'])} wear a fast-fashion item 10 times or fewer. In the conjoint, each person rated 8 "
          f"tops; lifespan mattered most ({pct(sv['conjoint_importance']['Lifespan'])} of the choice). A {p6.price} top that "
          f"lasts {p6.lifespan.split(' (')[0]} was rated {ratings['P6']:.1f}, higher than a {p1.price} one-season top "
          f"({ratings['P1']:.1f}).",
          f"Set Price to '{p6.price}' and Lifespan to '{p6.lifespan}': this exact top was in the survey.")
    kpis([(f"{sv['mean_keep_years']:.1f} years", "a garment is kept on average (WRAP UK: 2.2-3.3 years)"),
          (pct(sv["share_wears_under_10"]), "wear a fast-fashion item 10 times or fewer"),
          (pct(sv["mean_unworn_share"]), "of the wardrobe went unworn last year"),
          (pct(sv["disposal_share_never"]["Sell / resell"]), "never resell clothes")])
    st.markdown("##### 🎨 Design a top: how would our respondents rate it?")
    attrs = [("price_high", "Price", "price"), ("trendy", "Style", "style"), ("long_life", "Lifespan", "lifespan"),
             ("certified", "Label", "label")]
    cols, x = st.columns([0.8, 0.9, 1.1, 1.2]), {}
    short = lambda v: v.split(" (")[0].replace("Rs 1499", "Rs 1,499").replace("Certified sustainable", "Certified")
    for col, (attr, title, field) in zip(cols, attrs):
        opts = [design.loc[design[attr] == v, field].iloc[0] for v in (0, 1)]
        init(f"cj_{attr}", opts[0])
        x[attr] = int(col.segmented_control(title, opts, key=f"cj_{attr}", required=True, format_func=short) == opts[1])
    part = pw.set_index("attribute").part_worth
    gm = float(np.mean(list(ratings.values())))
    pred = gm + sum(part[a] * (x[a] - 0.5) for a, *_ in attrs)  # balanced design: each feature is "on" in half the tops
    hit = design[(design[[a for a, *_ in attrs]] == pd.Series(x)).all(axis=1)]
    a, b = st.columns([1, 1.35], gap="large")
    with a:
        shown = (f'In the survey, this exact top got <b>{ratings[hit.profile.iloc[0]]:.1f} / 10</b> on average.' if len(hit)
                 else "This exact top was not one of the 8 shown, so the rating is the model's estimate.")
        html(f'<div class="card"><div class="kicker">Predicted average rating</div><div class="big">{pred:.1f}'
             f'<span style="font-size:1.4rem;color:{MUTED}"> / 10</span></div>'
             f'<div class="meter"><div class="fill" style="width:{min(max(pred, 0), 10) * 10:.0f}%;background:{TEAL}"></div>'
             f'<div class="avg" style="left:{gm * 10:.0f}%"></div></div>'
             f'<div class="read">The black mark is the average of all 8 tops ({gm:.1f}). {shown}</div></div>')
    pdf = pw.assign(On=[bool(x[a_]) for a_ in pw.attribute],
                    Feature=pw.attribute.map({"price_high": "Rs 1,499 instead of Rs 799", "trendy": "Trendy instead of timeless",
                                              "long_life": "Lasts 3+ years instead of 1 season", "certified": "Sustainability label"}))
    pdf["Effect"] = np.where(pdf.part_worth > 0, "Raises the rating", "Lowers the rating")
    pb = alt.Chart(pdf).mark_bar(size=20, cornerRadiusEnd=4).encode(
        y=alt.Y("Feature:N", sort=pdf.sort_values("part_worth", ascending=False).Feature.tolist(), title=None),
        x=alt.X("part_worth:Q", title="Change in the 1-10 rating",
                scale=alt.Scale(domain=[min(pw.part_worth.min(), 0) - 0.3, pw.part_worth.max() + 0.5])),
        color=alt.Color("Effect:N", scale=alt.Scale(domain=["Raises the rating", "Lowers the rating"], range=[TEAL, TERRA]), legend=alt.Legend(title=None)),
        opacity=alt.condition(alt.datum.On, alt.value(1.0), alt.value(0.35)),
        tooltip=["Feature:N", alt.Tooltip("part_worth:Q", title="Effect", format="+.2f"), alt.Tooltip("ci_low:Q", title="95% CI low", format="+.2f"),
                 alt.Tooltip("ci_high:Q", title="95% CI high", format="+.2f")])
    pt = alt.Chart(pdf).mark_text(align="left", dx=4, fontSize=12, color=INK).encode(
        y=alt.Y("Feature:N", sort=pdf.sort_values("part_worth", ascending=False).Feature.tolist()),
        x=alt.X("part_worth:Q"), text=alt.Text("part_worth:Q", format="+.2f"))
    with b:
        show(alt.layer(pb, pt).properties(title=f"What each feature does to the rating ({sv['conjoint_respondents']} people × 8 tops)"), 230)
        read("solid bars are the features your top has. Lifespan has the biggest effect, then the label; a higher price "
             "lowers the rating.")
    dp = pd.DataFrame([{"Route": r, "How often": h, "Share": s[r]} for h, s in
                       (("Often or always", sv["disposal_share_often_or_always"]), ("Never", sv["disposal_share_never"])) for r in s])
    routes = sorted(sv["disposal_share_often_or_always"], key=sv["disposal_share_often_or_always"].get, reverse=True)
    HOW = ["Often or always", "Never"]
    db = alt.Chart(dp).mark_bar(size=14, cornerRadiusEnd=4).encode(
        y=alt.Y("Route:N", sort=routes, title=None), yOffset=alt.YOffset("How often:N", sort=HOW), x=alt.X("Share:Q", axis=alt.Axis(format=".0%"), title="Share of respondents"),
        color=alt.Color("How often:N", scale=alt.Scale(domain=HOW, range=[TEAL, TERRA]), legend=alt.Legend(title=None)),
        tooltip=["Route:N", "How often:N", alt.Tooltip("Share:Q", format=".0%")])
    dt = alt.Chart(dp).mark_text(align="left", dx=4, fontSize=11.5, color=INK).encode(
        y=alt.Y("Route:N", sort=routes), yOffset=alt.YOffset("How often:N", sort=HOW), x="Share:Q",
        text=alt.Text("Share:Q", format=".0%"))
    show(alt.layer(db, dt).properties(title="What happens to clothes people stop using"), 320)
    take(f"Lifespan matters most (<b>{pct(sv['conjoint_importance']['Lifespan'])}</b> of the choice), yet clothes are dropped "
         f"early and the second life is barely used: <b>{pct(sv['disposal_share_never']['Sell / resell'])}</b> never resell and "
         f"{pct(sv['disposal_share_never']['Recycle / take-back'])} never use recycling or take-back.")
    st.caption(f"Stated, not observed, behaviour of a convenience sample in India ({pct(sv['share_age_18_24'])} aged 18-24): it "
               "describes these respondents, not all consumers. Free-text comments are never shown.")
    method("- **Conjoint:** 8 tops from a balanced design (price × style × lifespan × label), each rated 1-10. A regression on "
           "each person's ratings (their own average removed) gives each feature's effect.\n"
           "- **The rating above** = the average of all ratings + each chosen feature's effect, measured from the middle of "
           "the design. People who gave all 8 tops the same rating are left out of the conjoint.",
           "Syllabus: Module 1 conjoint analysis · Module 5 ethics (consent, anonymity)")


# ----------------------------------------------------------------------------- step 8: wrap-up
def step_wrap():
    ps, pdz, vis, fs, cs, sv = (csv("pilot_sample_size"), js("pilot_design"), js("vis_summary"), js("vis_financials_summary"),
                                js("vis_customers_summary"), js("survey_descriptives"))
    st_all = csv("vis_financial_statement").set_index("season").loc["All seasons"]
    header("Wrap-up · How to prove it, and what to do", "A simulation is not proof. How would a retailer test this for real?",
           "A one-season randomised pilot (Module 1: marketing experiments), then five actions, cheapest first.")
    notes(f"A simulation is not proof, so we designed a pilot. Randomise the flagged products, not whole shops: "
          f"{pdz['products_per_arm_full_effect']} products per group are enough to detect the effect, while shops would need "
          f"{pdz['shops_per_arm_full_effect_simple']} per group. Then five actions, cheapest first. What is bad for the planet "
          "is also bad for the margin.",
          "Switch the effect size to 50%: the product pilot still needs few products.")
    eff = {"100% of the simulated effect": 1.0, "75%": 0.75, "50% (if the real effect is half)": 0.5}
    init("w_eff", next(iter(eff)))
    pick = st.segmented_control("How big is the real effect?", list(eff), key="w_eff", required=True)
    ps = ps.assign(Effect=ps.effect_as_share_of_simulated.map({v: k for k, v in eff.items()}),
                   Chosen=np.isclose(ps.effect_as_share_of_simulated, eff[pick]))
    dom = ["Randomise shops", "Randomise flagged products"]
    pb = alt.Chart(ps).mark_bar(size=30, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Effect:N", sort=list(eff), title=None, axis=alt.Axis(labelAngle=0)), xOffset=alt.XOffset("design:N", sort=dom),
        y=alt.Y("units_per_arm:Q", title="Needed in each group (80% power)"),
        color=alt.Color("design:N", scale=alt.Scale(domain=dom, range=[GREY, TEAL]), legend=alt.Legend(title=None)),
        opacity=alt.condition(alt.datum.Chosen, alt.value(1.0), alt.value(0.35)),
        tooltip=["design:N", "Effect:N", alt.Tooltip("units_per_arm:Q", title="Needed per group", format=",.0f")])
    pt = alt.Chart(ps).mark_text(dy=-8, fontSize=12.5, fontWeight="bold", color=INK).encode(
        x=alt.X("Effect:N", sort=list(eff)), xOffset=alt.XOffset("design:N", sort=dom), y="units_per_arm:Q",
        text=alt.Text("units_per_arm:Q", format=",.0f"))
    chosen = ps[ps.Chosen].set_index("design").units_per_arm
    a, b = st.columns([1.5, 1], gap="large")
    with a:
        show(alt.layer(pb, pt).properties(title="Pilot size: shops vs flagged products"), 300)
    with b:
        html('<div class="card">' + kv([("Flagged products per group", f"{chosen['Randomise flagged products']:,.0f}"),
                                        ("Shops per group", f"{chosen['Randomise shops']:,.0f}"),
                                        ("Flagged products in a season", f"about {pdz['flagged_products_per_season']}"),
                                        ("Shops in a season", f"about {pdz['shops_available_per_season']}")]) + "</div>")
        read("half the flagged products follow the week-2 rule, half do not, chosen at random within each category. "
             "Whole shops dilute the effect, because most products in a shop are never flagged.")
    st.markdown("##### What a fast-fashion retailer should do")
    top = cs["top_cltv_segment"]
    recs = [("Buy shallow, then react", "Use the pre-launch risk score to set a smaller first order for riskier products."),
            ("Hold a week-2 review", f"Forecast every new product; below {HIGH:.0%} predicted sell-through, cut its later deliveries first."),
            ("Mark down early, only where needed", "Discount flagged products from week 4 instead of waiting until demand has collapsed."),
            ("Use the loyal core", f"'{top}' customers buy {pct(cs['top_cltv_segment_share_units'])} of units: reach them first with new products."),
            ("Report excess stock as a KPI", f"Track unsold stock ÷ revenue ({pct(st_all.stock_to_sales)}) and the markdown rate "
                                             f"({pct(st_all.markdown_rate, 1)}) next to gross margin.")]
    rc = st.columns(5)
    for i, (col, (t, d)) in enumerate(zip(rc, recs), start=1):
        col.markdown(f'<div class="card" style="min-height:190px"><span class="num">{i}</span><b>{t}</b>'
                     f'<div style="font-size:.9rem;color:#4A443E;margin-top:8px">{d}</div></div>', unsafe_allow_html=True)
    with st.expander("⚖️ Limits we are honest about", expanded=False):
        st.markdown(
            "- **Association, not causation.** The environmental harm is cited (UNEP, Ellen MacArthur Foundation), not measured by us.\n"
            "- **We see deliveries and sales, not production.** Overproduction is the likely upstream cause, not a measured fact.\n"
            "- **One retailer** (a mid-sized Italian chain, 2017-2019). Prices are normalised (index units) and unit cost is assumed (40-60%).\n"
            f"- **The survey** is a young convenience sample ({sv['respondents'] if sv else '—'} people): it describes them, not all consumers.")
    sim = vis["simulation_main_scenario"]
    html('<div class="hero" style="margin-top:1.2rem"><div class="kicker">Conclusion</div>'
         '<div class="htitle" style="font-size:2.3rem">What is bad for the planet is also bad for the margin.</div>'
         f"<p>{pct(vis['stock']['leftover_share_of_delivered'], 1)} of delivered stock is left unsold. It is partly predictable "
         f"before launch and clearer at week 2, and acting then cuts unsold stock by {pct(-sim['leftover_change'], 1)} while "
         f"raising gross profit ({signed(fs['policy_gross_profit_change']['cost40'])} to "
         f"{signed(fs['policy_gross_profit_change']['cost60'])}, for unit costs of 40-60%).</p></div>")


# ----------------------------------------------------------------------------- navigation
STEPS = [
    ("Start", "The story in one minute", "🧵"),
    ("Stage 1 · Explore", "What happens to stock in the shops", "🔍"),
    ("Stage 2 · Predict", "Spot risky products before launch", "🎯"),
    ("Stage 3 · Forecast", "Read the signal after 2 weeks", "📈"),
    ("Stage 4 · Act", "What if the retailer acts at week 2", "🛠️"),
    ("Stage 5 · Money", "What excess stock costs", "💰"),
    ("Customers", "Who buys, and how", "👥"),
    ("Consumers", "What our survey says", "🗣️"),
    ("Wrap-up", "Prove it, and what to do", "✅"),
]
PAGES = [step_start, step_explore, step_predict, step_forecast, step_act, step_money, step_customers, step_consumers, step_wrap]
GLOSSARY = [
    ("Sell-through", "the share of delivered stock that sold (sold ÷ delivered)."),
    ("High leftover", f"a product that sold under {HIGH:.0%} of its stock in {K_WEEKS} weeks."),
    ("Markdown", f"a discount; a week counts as a markdown week when the average discount is {MARKDOWN_THRESHOLD:.0%} or more."),
    ("K-means", "a method that groups similar items (sales curves, customers) without being told the groups."),
    ("AUC", "how well a score ranks risky products above safe ones: 0.5 = a coin toss, 1 = perfect."),
    ("Odds ratio", "how much a feature multiplies the odds of high leftover (1 = no effect)."),
    ("WAPE", "total forecast error ÷ total sales (lower is better)."),
    ("Naive forecast", "assumes week 2's sales simply repeat."),
    ("RFM · CLTV", "recency, frequency, monetary value · customer lifetime value."),
    ("Conjoint", "people rate options that differ in features; the ratings show what each feature is worth to them."),
    ("Index units", "VISUELLE prices are normalised, so money is in index units; ratios are unaffected."),
    ("Power analysis", "how many test units a pilot needs to detect the effect reliably."),
]

NEEDED = [PROC / "vis_products.parquet", TAB / "vis_summary.json", TAB / "vis_lifecycle_summary.json", MODEL]
missing = [str(p.relative_to(ROOT)) for p in NEEDED if not p.exists()]
if missing:
    st.error("Project outputs are missing (" + ", ".join(missing) + "). Run `python -X utf8 -m src.run_all` first (see README).")
    st.stop()

init("step", 0)
init("notes", False)
with st.sidebar:
    html('<div style="font-family:Georgia,serif;font-size:1.35rem;font-weight:700;color:#2B2B2B">🧵 From Trend to Waste</div>')
    st.caption("Guided demo · VISUELLE 2.0 (Nuna Lie) + our survey")
    st.radio("Demo steps", list(range(len(STEPS))), key="step", format_func=lambda k: f"{STEPS[k][2]}  {STEPS[k][0]}")
    st.toggle("🎤 Presenter notes", key="notes", help="Shows what to say and what to click on each step")
    with st.expander("📖 Glossary"):
        for term, text in GLOSSARY:
            st.markdown(f"**{term}**: {text}")
    st.caption("Every number is read from the project's outputs (listed in reports/claims_register.csv).")

step = st.session_state.step
if st.session_state.get("_shown_step") != step:  # a new step starts at the top of the page
    st.session_state["_shown_step"] = step
    st.html(f"<script>/* step {step} */ var m = document.querySelector('[data-testid=\"stMain\"]');"
            "if (m) {m.scrollTo(0, 0);} window.scrollTo(0, 0);</script>", unsafe_allow_javascript=True)
st.progress((step + 1) / len(STEPS), text=f"Step {step + 1} of {len(STEPS)} · {STEPS[step][0]}")
PAGES[step]()
st.divider()
b1, mid, b2 = st.columns([1, 2.4, 1], vertical_alignment="center")
if step > 0:
    b1.button(f"← {STEPS[step - 1][0]}", key="nav_back", on_click=go, args=(step - 1,), width="stretch")
mid.markdown(f"<div style='text-align:center;color:{MUTED}'>{STEPS[step][2]} {STEPS[step][1]}</div>", unsafe_allow_html=True)
if step < len(STEPS) - 1:
    b2.button(f"{STEPS[step + 1][0]} →", key="nav_next", on_click=go, args=(step + 1,), type="primary", width="stretch")
