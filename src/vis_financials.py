"""Financial statements built from the data (Module 4) — VISUELLE 2.0, Nuna Lie.

Nuna Lie is a private company and publishes no accounts, so this module builds a season-level inventory statement
from the dataset's own deliveries, sales, prices and discounts (first 12 weeks of every product in every shop):
  revenue (after discounts), markdown cost, stock delivered and closing (unsold) stock at full price,
  sell-through, stock-to-sales, inventory turnover, days of inventory.
Two limits, stated wherever the numbers appear:
  - VISUELLE prices are normalised, so money amounts are INDEX UNITS (not euros); ratios are unaffected.
  - Cost of goods is not in the data: gross margins use an ASSUMED cost-to-full-price ratio, shown as a range
    (40%, 50%, 60%). Ratios that need no cost (sell-through, stock-to-sales, turnover, markdown rate) are data only.
The same statement is rebuilt for the 2019 test seasons with and without the week-2 rule (src/vis_forecast.simulate).
"""
import json

import numpy as np
import pandas as pd

from . import vis_forecast as vf
from .config import SEASON_ORDER, TAB
from .utils import PALETTE, plt, register_claim, save_json, save_table, savefig

COST_RATIOS = (0.4, 0.5, 0.6)   # assumed unit cost as a share of full price (not in the data)
DAYS = 7 * vf.K


def statement(price, S, D, Dsc):
    """Inventory statement for a set of product-shops (rows). Money in index units."""
    avail, end = vf.stock_path(D, S)
    left = np.maximum(end[:, -1], 0)
    fp_sold = (price[:, None] * S).sum()
    net = (price[:, None] * (1 - Dsc) * S).sum()
    fp_deliv = (price[:, None] * D).sum()
    avg_stock = np.maximum(end, 0).sum(0).mean()          # average end-of-week stock, units
    out = {"units_delivered": D.sum(), "units_sold": S.sum(), "units_unsold": left.sum(),
           "full_price_value_delivered": fp_deliv, "full_price_value_sold": fp_sold, "net_revenue": net,
           "markdown_cost": fp_sold - net, "closing_stock_value": (price * left).sum()}
    out.update({"sell_through": out["units_sold"] / out["units_delivered"],
                "markdown_rate": out["markdown_cost"] / fp_sold,
                "stock_to_sales": out["closing_stock_value"] / net,
                "inventory_turnover_12w": out["units_sold"] / avg_stock,
                "days_of_inventory": avg_stock / (out["units_sold"] / DAYS)})
    for c in COST_RATIOS:
        k = int(c * 100)
        out[f"gross_margin_cost{k}"] = (net - c * fp_sold) / net                     # on goods sold
        out[f"margin_after_unsold_cost{k}"] = (net - c * fp_deliv) / net            # if unsold stock is not recovered
    return out


def main():
    meta, S, D, Dsc, _ = vf.load()
    price = meta.price.values
    rows = [{"season": s, **statement(price[m], S[m], D[m], Dsc[m])}
            for s in SEASON_ORDER for m in [(meta.season == s).values]]
    rows.append({"season": "All seasons", **statement(price, S, D, Dsc)})
    st = pd.DataFrame(rows)
    save_table(st, "vis_financial_statement")
    print(st.round(3).to_string())

    # Week-2 rule on the 2019 test seasons: same statement, baseline vs policy (as in src/vis_forecast.py)
    vs = json.loads((TAB / "vis_summary.json").read_text())
    main_s = vs["simulation_main_scenario"]
    pred = pd.read_csv(TAB / "vis_test_leftover_predictions.csv")
    test = meta.season.isin(vf.TEST).values
    flagged = set(pred.loc[pred.pred_sell_through < main_s["flag_threshold_pred_sell_through"], "external_code"])
    flag = np.isin(meta.external_code.values[test], list(flagged))
    beta = vs["uplift"]["beta_fe"]
    pt, St, Dt, Dsct = price[test], S[test], D[test], Dsc[test]
    base = vf.simulate(Dt, St, Dsct, pt, np.zeros(len(St), bool), beta=beta)
    pol = vf.simulate(Dt, St, Dsct, pt, flag, r_cut=main_s["delivery_cut"], md_level=main_s["markdown_level"], beta=beta)
    check = pol["leftover"] / base["leftover"] - 1
    assert abs(check - main_s["leftover_change"]) < 1e-9, "policy replay must match src/vis_forecast.py"
    pr = []
    for name, r in (("Without the rule (actual 2019)", base), ("With the week-2 rule", pol)):
        row = {"scenario": name, "units_delivered": r["delivered"], "units_sold": r["sold"], "units_unsold": r["leftover"],
               "net_revenue": r["revenue"], "full_price_value_delivered": r["purchase_value_full_price"]}
        for c in COST_RATIOS:
            row[f"gross_profit_after_unsold_cost{int(c * 100)}"] = r["revenue"] - c * r["purchase_value_full_price"]
        pr.append(row)
    pol_tab = pd.DataFrame(pr)
    save_table(pol_tab, "vis_financial_policy")
    gp_change = {f"cost{int(c * 100)}": float(pol_tab.iloc[1][f"gross_profit_after_unsold_cost{int(c * 100)}"]
                                              / pol_tab.iloc[0][f"gross_profit_after_unsold_cost{int(c * 100)}"] - 1)
                 for c in COST_RATIOS}

    a = st.set_index("season").loc["All seasons"]
    by = st[st.season != "All seasons"].set_index("season")
    summary = {"money_units": "index units (VISUELLE prices are normalised)", "cost_ratios_assumed": list(COST_RATIOS),
               "all": {k: float(v) for k, v in a.items()},
               "stock_to_sales_range": [float(by.stock_to_sales.min()), float(by.stock_to_sales.max())],
               "markdown_rate_range": [float(by.markdown_rate.min()), float(by.markdown_rate.max())],
               "policy_gross_profit_change": gp_change, "policy_revenue_change": float(pol["revenue"] / base["revenue"] - 1),
               "policy_full_price_value_delivered_change": float(pol["purchase_value_full_price"] / base["purchase_value_full_price"] - 1),
               "breakeven_cost_to_price": main_s["breakeven_cost_to_price"]}
    save_json(summary, "vis_financials_summary")
    print(pol_tab.round(1).to_string())
    print(summary)

    src = "outputs/tables/vis_financial_statement.csv"
    rc = register_claim
    rc("fin_markdown_rate", "Markdown cost as share of full-price value of goods sold, all seasons", float(a.markdown_rate), src)
    rc("fin_stock_to_sales", "Closing (unsold) stock at full price / net revenue, all seasons", float(a.stock_to_sales), src)
    rc("fin_turnover", "Inventory turnover over 12 weeks (units sold / average stock)", float(a.inventory_turnover_12w), src)
    rc("fin_days_inventory", "Days of inventory (average stock / daily units sold)", float(a.days_of_inventory), src)
    rc("fin_gm_cost50", "Gross margin on goods sold, assuming cost = 50% of full price", float(a.gross_margin_cost50), src)
    rc("fin_margin_unsold_cost50", "Margin if unsold stock is not recovered, assuming cost = 50% of full price",
       float(a.margin_after_unsold_cost50), src)
    rc("fin_policy_gp_cost50", "Week-2 rule: change in gross profit after unsold stock (2019, cost = 50% of full price)",
       gp_change["cost50"], "outputs/tables/vis_financial_policy.csv")
    rc("fin_policy_gp_range", "Week-2 rule: change in gross profit after unsold stock, cost 40% / 60% of full price",
       f"{gp_change['cost40']:.4f} / {gp_change['cost60']:.4f}", "outputs/tables/vis_financial_policy.csv")
    plots(by)


def plots(by):
    by = by.loc[SEASON_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(by))
    axes[0].bar(x - 0.2, by.stock_to_sales * 100, width=0.4, color=PALETTE["accent"], label="Unsold stock / net revenue")
    axes[0].bar(x + 0.2, by.markdown_rate * 100, width=0.4, color=PALETTE["teal"], label="Markdown cost / full-price sales")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(by.index)
    axes[0].set_ylabel("%")
    axes[0].set_title("Excess stock and markdowns, by season")
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.3))
    for c, col, ls in ((0.4, PALETTE["teal"], "-"), (0.5, PALETTE["ink"], "-"), (0.6, PALETTE["accent"], "-")):
        k = int(c * 100)
        axes[1].plot(x, by[f"gross_margin_cost{k}"] * 100, color=col, lw=2, marker="o", label=f"On goods sold (cost {k}%)")
        axes[1].plot(x, by[f"margin_after_unsold_cost{k}"] * 100, color=col, lw=1.6, ls="--", marker="o",
                     label=f"If unsold stock is lost (cost {k}%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(by.index)
    axes[1].set_ylabel("Margin (% of net revenue)")
    axes[1].set_title("Margins under assumed unit costs")
    axes[1].legend(frameon=False, fontsize=7.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.45))
    savefig(fig, "vis_financial_statement")


if __name__ == "__main__":
    main()
