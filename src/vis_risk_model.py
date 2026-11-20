"""Pre-launch excess-stock risk (before launch -> after 2 weeks) — VISUELLE 2.0, Nuna Lie.

Target (fixed before modelling): high leftover = the product sells less than 60% of its delivered stock in
its first 12 weeks (all shops). The same threshold is used by the week-2 flag in src/vis_forecast.py.
Time-based split: products of 2017-2018 seasons train, 2019 seasons test (never seen in training).
Stages:
  A. Before launch, product only: category, colour, fabric, season type, release month, price position
  B. Before launch, + the buying plan: number of shops and initial units per shop (decided before release)
  C. After 2 weeks: B + first two weeks of sales
Explainable output: odds ratios (statsmodels logit, all products). The dashboard gets an unweighted logistic
regression (stage B) so its probabilities are calibrated to the base rate.
"""
import joblib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import MODELS, PROCESSED, RANDOM_STATE, TEST_SEASONS, TRAIN_SEASONS
from .utils import PALETTE, plt, register_claim, save_json, save_table, savefig

CATS = ["category", "color", "fabric", "season_type"]
NUM_A = ["price_tier", "release_month"]
NUM_B = NUM_A + ["log_shops", "log_initial_per_shop"]
NUM_C = NUM_B + ["log_w1", "log_w2", "early_sell_through"]


def load():
    p = pd.read_parquet(PROCESSED / "vis_products.parquet")
    p = p[p.delivered_12w > 0].sort_values("external_code").reset_index(drop=True)  # fixed order -> reproducible
    p["season_type"] = p.season.str[:2]
    # Price position within category (0 = cheapest, 1 = priciest), known before launch
    p["price_tier"] = (p.groupby("category").price.rank(method="average") - 0.5) / p.groupby("category").price.transform("size")
    p["log_shops"] = np.log(p.n_shops)
    p["initial_per_shop"] = p.dl0 / p.n_shops
    p["log_initial_per_shop"] = np.log1p(p.initial_per_shop)
    p["log_w1"], p["log_w2"] = np.log1p(p.u0), np.log1p(p.u1)
    p["early_sell_through"] = (p.u0 + p.u1) / np.maximum(p.dl0 + p.dl1, 1)
    p["y"] = p.high_leftover.astype(int)
    return p


def pre(num):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False), CATS),
        ("num", StandardScaler(), num)])


def precision_at(y, s, frac=0.10):
    top = np.argsort(-s)[:max(1, int(len(s) * frac))]
    return float(y[top].mean())


def evaluate(name, stage, y, s):
    return {"model": name, "stage": stage, "roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
            "precision_top10pct": precision_at(y, s), "base_rate": float(y.mean())}


def main():
    df = load()
    tr, te = df[df.season.isin(TRAIN_SEASONS)], df[df.season.isin(TEST_SEASONS)]
    ytr, yte = tr.y.values, te.y.values
    print(f"train {len(tr)} (high-leftover rate {ytr.mean():.3f})  test {len(te)} ({yte.mean():.3f})")
    models = {
        "Logistic regression": lambda: LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced"),
        "Decision tree (depth 4)": lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, class_weight="balanced",
                                                                  random_state=RANDOM_STATE),
        "Gradient boosting": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                                                    random_state=RANDOM_STATE),
    }
    stages = {"Before launch: product only": NUM_A, "Before launch: product + buying plan": NUM_B,
              "After 2 weeks of sales": NUM_C}
    rows, scores = [], {}
    for stage, num in stages.items():
        for name, make in models.items():
            if stage == "After 2 weeks of sales" and name.startswith("Decision"):
                continue
            pipe = Pipeline([("pre", pre(num)), ("m", make())]).fit(tr[CATS + num], ytr)
            s = pipe.predict_proba(te[CATS + num])[:, 1]
            rows.append(evaluate(name, stage, yte, s))
            scores[(stage, name)] = s
    rate = tr.groupby("category").y.mean()
    rows.append(evaluate("Baseline: category rate", "Baseline", yte, te.category.map(rate).fillna(ytr.mean()).values))
    res = pd.DataFrame(rows)
    save_table(res, "vis_risk_model_metrics")
    print(res.round(3).to_string())

    # Odds ratios on all products, compact feature set with readable reference levels
    o = df.copy()
    ref = {}
    for c in ("category", "color", "fabric"):  # levels with fewer than 100 products are pooled as "rare"
        top = o[c].value_counts()
        ref[c] = top.index[0]  # reference = the most common real level
        o[c] = np.where(o[c].isin(top[top >= 100].index), o[c], "rare " + c)
    o["price_q"] = pd.qcut(o.price_tier, 4, labels=["Q1 cheapest", "Q2", "Q3", "Q4 priciest"])
    o["depth_q"] = pd.qcut(o.initial_per_shop.rank(method="first"), 4, labels=["Q1 shallowest", "Q2", "Q3", "Q4 deepest"])
    formula = (f"y ~ C(category, Treatment('{ref['category']}')) + C(color, Treatment('{ref['color']}')) "
               f"+ C(fabric, Treatment('{ref['fabric']}')) + C(season_type, Treatment('SS')) "
               "+ C(price_q, Treatment('Q2')) + C(depth_q, Treatment('Q2'))")
    lm = smf.logit(formula, data=o).fit(disp=False, maxiter=300)
    ci = lm.conf_int()
    orr = pd.DataFrame({"term": lm.params.index, "odds_ratio": np.exp(lm.params.values), "ci_low": np.exp(ci[0].values),
                        "ci_high": np.exp(ci[1].values), "p_value": lm.pvalues.values})
    orr = orr[orr.term != "Intercept"]
    orr["term"] = (orr.term.str.replace(r"C\((\w+), Treatment\('[^']+'\)\)\[T\.", r"\1 = ", regex=True).str.replace("]", "", regex=False))
    orr = orr.sort_values("odds_ratio", ascending=False)
    save_table(orr, "vis_risk_odds_ratios")

    get = lambda st, m: res[(res.stage == st) & (res.model == m)].iloc[0]
    a, b = get("Before launch: product only", "Logistic regression"), get("Before launch: product + buying plan", "Logistic regression")
    bg = get("Before launch: product + buying plan", "Gradient boosting")
    c_lr, c_gb = get("After 2 weeks of sales", "Logistic regression"), get("After 2 weeks of sales", "Gradient boosting")
    tree = get("Before launch: product + buying plan", "Decision tree (depth 4)")
    depth = orr[orr.term == "depth_q = Q4 deepest"].iloc[0]
    summary = {"target": "sell-through below 60% after 12 weeks", "n_train": int(len(tr)), "n_test": int(len(te)),
               "test_base_rate": float(yte.mean()), "train_base_rate": float(ytr.mean()),
               "auc_product_only": a.roc_auc, "auc_with_plan": b.roc_auc, "auc_with_plan_gb": bg.roc_auc,
               "auc_after_2_weeks_lr": c_lr.roc_auc, "auc_after_2_weeks_gb": c_gb.roc_auc, "auc_tree": tree.roc_auc,
               "p10_product_only": a.precision_top10pct, "p10_with_plan": b.precision_top10pct,
               "p10_after_2_weeks": max(c_lr.precision_top10pct, c_gb.precision_top10pct),
               "or_deepest_vs_q2": float(depth.odds_ratio), "or_deepest_ci": [float(depth.ci_low), float(depth.ci_high)],
               "reference_levels": ref}
    save_json(summary, "vis_risk_model_summary")
    src = "outputs/tables/vis_risk_model_metrics.csv"
    rc = register_claim
    rc("risk_base_rate", "High-leftover rate among 2019 test products (sell-through < 60%)", summary["test_base_rate"], src)
    rc("risk_auc_product", "Before launch, product only: logistic regression test ROC-AUC", a.roc_auc, src)
    rc("risk_auc_plan", "Before launch, product + buying plan: logistic regression test ROC-AUC", b.roc_auc, src)
    rc("risk_auc_plan_gb", "Before launch, product + buying plan: gradient boosting test ROC-AUC", bg.roc_auc, src)
    rc("risk_auc_2wk", "After 2 weeks: best test ROC-AUC", max(c_lr.roc_auc, c_gb.roc_auc), src)
    rc("risk_auc_tree", "Decision tree (depth 4) test ROC-AUC, before launch", tree.roc_auc, src)
    rc("risk_p10_plan", "Before launch (with plan): share high-leftover among top-10% risk scores", b.precision_top10pct, src)
    rc("risk_or_deepest", "Odds ratio of high leftover, deepest vs second-quartile initial units per shop",
       f"{depth.odds_ratio:.2f} (95% CI {depth.ci_low:.2f}-{depth.ci_high:.2f})", "outputs/tables/vis_risk_odds_ratios.csv")

    # Dashboard model: unweighted logistic regression (calibrated probabilities), stage B
    calib = Pipeline([("pre", pre(NUM_B)), ("m", LogisticRegression(C=0.5, max_iter=5000))]).fit(tr[CATS + NUM_B], ytr)
    pc = calib.predict_proba(te[CATS + NUM_B])[:, 1]
    save_json({"auc": float(roc_auc_score(yte, pc)), "mean_predicted": float(pc.mean()), "observed_rate": float(yte.mean())},
              "vis_risk_dashboard_model_check")
    joblib.dump({"pipe": calib, "cats": CATS, "num": NUM_B, "levels": {c: sorted(df[c].dropna().unique().tolist()) for c in CATS},
                 "price_by_category": df.groupby("category").price.apply(lambda s: sorted(s.tolist())).to_dict(),
                 "defaults": {"n_shops": int(df.n_shops.median()), "initial_per_shop": float(df.initial_per_shop.median())},
                 "base_rate": float(ytr.mean())}, MODELS / "vis_risk_logit.joblib")
    plots(orr, yte, scores[("Before launch: product + buying plan", "Logistic regression")], ref)


PRETTY = {"category": "Category", "color": "Colour", "fabric": "Fabric", "season_type": "Season", "price_q": "Price",
          "depth_q": "Initial units per shop"}


def pretty(t):
    k, _, v = t.partition(" = ")
    return f"{PRETTY.get(k, k)}: {v}"


def plots(orr, yte, s, ref):
    sig = orr[orr.p_value < 0.05]
    show = pd.concat([sig.head(8), sig.tail(8)]).drop_duplicates().sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(7.5, 0.36 * len(show) + 1.3))
    y = np.arange(len(show))
    col = [PALETTE["accent"] if v > 1 else PALETTE["teal"] for v in show.odds_ratio]
    ax.hlines(y, show.ci_low, show.ci_high, color=col, lw=2)
    ax.scatter(show.odds_ratio, y, color=col, zorder=3)
    ax.axvline(1, color=PALETTE["muted"], lw=1)
    ax.set_xscale("log")
    ax.minorticks_off()
    ticks = [t for t in (0.25, 0.4, 0.6, 1, 1.5, 2.5, 4, 6) if show.ci_low.min() * 0.8 <= t <= show.ci_high.max() * 1.25]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}x" for t in ticks])
    ax.set_yticks(y)
    ax.set_yticklabels([pretty(t) for t in show.term], fontsize=9)
    ax.set_xlabel("Odds of high leftover vs the reference product (log scale, 95% CI)")
    ax.set_title("What raises the risk of excess stock?", pad=18)
    ax.text(0, 1.012, f"Reference: {ref['category']}, {ref['color']}, {ref['fabric']}, spring-summer, price and depth quartile 2",
            transform=ax.transAxes, fontsize=8.5, color=PALETTE["muted"])
    savefig(fig, "vis_risk_odds_ratios")

    dec = pd.qcut(pd.Series(s).rank(method="first"), 10, labels=False)
    rate = pd.Series(yte).groupby(dec.values).mean() * 100
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(np.arange(1, 11), rate.values, color=[PALETTE["accent"] if i >= 8 else PALETTE["sand"] for i in range(10)])
    ax.axhline(yte.mean() * 100, color=PALETTE["ink"], ls="--", lw=1, label=f"Average ({yte.mean():.0%})")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Risk-score decile (1 = lowest predicted risk, 10 = highest)")
    ax.set_ylabel("% with high leftover")
    ax.set_title("Pre-launch risk score on unseen 2019 products")
    ax.legend(frameon=False)
    savefig(fig, "vis_risk_deciles")


if __name__ == "__main__":
    main()
