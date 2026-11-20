"""Fill reports/final_report_template.md with numbers read from outputs/ -> reports/final_report.md.

No number in the report is typed by hand: every {{token}} is computed here from a script output
(or from src/literature.py / src/env_factors.py for cited external facts). Unknown tokens fail loudly.
Also writes outputs/tables/deck_data.json for the slide deck.
"""
import json
import re

import pandas as pd

from .config import REPORTS, ROOT, TAB
from .env_factors import CO2E_PER_KG, GARMENT_KG
from .literature import LITERATURE


TEMPLATES = ["final_report_template.md", "viva_notes_template.md", "project_summary_template.md", "review2_guide_template.md"]


def j(name):
    return json.loads((TAB / name).read_text())


def c(name):
    return pd.read_csv(TAB / name)


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


def signed(x, d=1):
    s = f"{x * 100:+.{d}f}%"
    return "0.0%" if s in ("-0.0%", "+0.0%") else s


def md_table(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.get(k, str)(r[k]) for k in cols) + " |")
    return "\n".join(out)


def values():
    V = {}
    vis = j("vis_summary.json")
    st = vis["stock"]
    lc = j("vis_lifecycle_summary.json")
    clus = c("vis_lifecycle_by_cluster.csv")
    timing = c("vis_markdown_timing.csv").set_index("short_lived")
    rs = j("vis_risk_model_summary.json")
    risk = c("vis_risk_model_metrics.csv")
    orr = c("vis_risk_odds_ratios.csv")
    cs = j("vis_customers_summary.json")
    seg = c("vis_customer_segments.csv")
    freq = c("vis_customers_by_frequency.csv")
    fs = j("vis_financials_summary.json")
    fst = c("vis_financial_statement.csv")
    fpol = c("vis_financial_policy.csv")
    fc = c("vis_forecast_metrics.csv")
    grid = c("vis_simulation_grid.csv")
    survey = (TAB / "survey_descriptives.json")
    f2 = lambda x: f"{x:.2f}"

    # ---- data
    V["vis_products"] = f"{st['products']:,}"
    V["vis_shops"] = f"{st['shops']:,}"
    V["vis_ps"] = f"{st['product_shops']:,}"
    V["vis_delivered"] = f"{st['units_delivered_12w']:,.0f}"
    V["vis_sold"] = f"{st['units_sold_12w']:,.0f}"
    V["cust_lines"] = f"{cs['purchase_lines_raw']:,}"
    V["cust_lines_used"] = f"{cs['purchase_lines_used']:,}"
    V["cust_n"] = f"{cs['customers']:,}"
    V["cust_cards"] = str(cs["card_ids_excluded"])
    V["cust_known_disc"] = pct(cs["share_lines_with_known_discount"])
    V["cust_matched_shop"] = pct(cs["share_lines_matched_to_shop_sales"])

    # ---- lifecycles and markdowns
    V["lc_n"] = f"{lc['products_in_sample']:,}"
    V["lc_min"] = str(lc["min_units"])
    V["lc_k"] = str(lc["k_chosen"])
    V["lc_sil"] = f2(lc["silhouette_by_k"][str(lc["k_chosen"])])
    shares = lc["cluster_shares"]
    V["lc_clusters_text"] = ", ".join(f"{k.lower()} ({pct(v)})" for k, v in sorted(shares.items(), key=lambda kv: -kv[1]))
    am = lc["all_products_markdown"]
    V["lc_peak_first3"] = pct(am["share_peak_in_first_3_weeks"])
    V["lc_md_share"] = pct(lc["share_marked_down_12w"])
    V["lc_units_md"] = pct(lc["share_units_on_markdown"])
    V["lc_units_md_10"] = pct(lc["markdown_sensitivity"]["10pct"]["share_units_sold_on_markdown"])
    V["lc_units_md_20"] = pct(lc["markdown_sensitivity"]["20pct"]["share_units_sold_on_markdown"])
    V["lc_first_md_week"] = f"{am['median_first_markdown_week']:.0f}"
    V["lc_weeks_peak_md"] = f"{am['median_weeks_peak_to_markdown']:.0f}"
    V["lc_demand_at_md"] = pct(am["median_demand_at_markdown_vs_peak"])
    V["lc_md_after_half"] = pct(am["marked_down_after_demand_halved"])
    V["sl_share"] = pct(lc["share_short_lived"], 1)
    V["sl_n"] = f"{lc['short_lived_products']:,}"
    V["sl_weeks_half"] = f"{lc['short_lived_median_weeks_peak_to_half']:.0f}"
    V["sl_md_after_half"] = pct(lc["short_lived_marked_down_after_demand_halved"])
    V["other_md_after_half"] = pct(lc["other_marked_down_after_demand_halved"])
    V["sl_weeks_peak_md"] = f"{lc['short_lived_median_weeks_peak_to_markdown']:.0f}"
    V["sl_st"] = pct(lc["short_lived_sell_through_median"])
    V["other_st"] = pct(lc["other_sell_through_median"])
    V["table_clusters"] = md_table(
        clus[["cluster_name", "share_products", "mean_share_last6", "median_weeks_peak_to_half", "share_marked_down",
              "share_short_lived", "median_sell_through"]].rename(columns={
            "cluster_name": "Lifecycle cluster", "share_products": "% of products", "mean_share_last6": "% of sales in weeks 7-12",
            "median_weeks_peak_to_half": "Weeks from peak to half (median)", "share_marked_down": "% marked down",
            "share_short_lived": "% short-lived", "median_sell_through": "Sell-through (median)"}),
        {"% of products": pct, "% of sales in weeks 7-12": pct, "Weeks from peak to half (median)": lambda x: f"{x:.0f}",
         "% marked down": pct, "% short-lived": lambda x: pct(x, 1), "Sell-through (median)": pct})

    # ---- pre-launch risk
    V["risk_target"] = "sell-through below 60% after 12 weeks"
    V["risk_base"] = pct(rs["test_base_rate"])
    V["risk_base_train"] = pct(rs["train_base_rate"])
    V["risk_auc_product"] = f2(rs["auc_product_only"])
    V["risk_auc_plan"] = f2(rs["auc_with_plan"])
    V["risk_auc_plan_gb"] = f2(rs["auc_with_plan_gb"])
    V["risk_auc_2wk"] = f2(max(rs["auc_after_2_weeks_lr"], rs["auc_after_2_weeks_gb"]))
    V["risk_auc_tree"] = f2(rs["auc_tree"])
    V["risk_auc_baseline"] = f2(risk[risk.model == "Baseline: category rate"].roc_auc.iloc[0])
    V["risk_p10_plan"] = pct(rs["p10_with_plan"])
    V["risk_p10_2wk"] = pct(rs["p10_after_2_weeks"])
    V["risk_lift"] = f"{rs['p10_with_plan'] / rs['test_base_rate']:.1f}"
    V["risk_or_deep"] = f2(rs["or_deepest_vs_q2"])
    V["risk_or_deep_ci"] = f"{rs['or_deepest_ci'][0]:.2f}-{rs['or_deepest_ci'][1]:.2f}"
    ref = rs["reference_levels"]
    V["risk_ref"] = f"{ref['category']}, {ref['color']}, {ref['fabric']}, spring-summer, price and depth quartile 2"
    pretty = {"category": "category", "color": "colour", "fabric": "fabric", "season_type": "season", "price_q": "price",
              "depth_q": "initial units per shop"}
    def nice(t):
        k, v = t.split(" = ", 1)
        if k == "season_type":
            return "autumn-winter season" if v == "AW" else "spring-summer season"
        if k in ("price_q", "depth_q"):
            word = v.split(" ", 1)[1] if " " in v else v
            return f"{word} quarter of prices" if k == "price_q" else f"{word} quarter of initial units per shop"
        return f"'{v}' fabric" if k == "fabric" else f"{v} {pretty[k]}"
    sig = orr[(orr.p_value < 0.05) & ~orr.term.str.contains("rare")]
    V["risk_up"] = ", ".join(nice(t) for t in sig[sig.odds_ratio > 1].term)
    V["risk_down"] = ", ".join(nice(t) for t in sig[sig.odds_ratio < 1].term)
    V["table_risk"] = md_table(risk[["stage", "model", "roc_auc", "pr_auc", "precision_top10pct"]].rename(columns={
        "stage": "Stage", "model": "Model", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "precision_top10pct": "Hit rate in top 10%"}),
        {"ROC-AUC": f2, "PR-AUC": f2, "Hit rate in top 10%": pct})
    V["table_or"] = md_table(sig[["term", "odds_ratio", "ci_low", "ci_high", "p_value"]].assign(term=sig.term.map(nice)).rename(
        columns={"term": "Attribute (vs reference)", "odds_ratio": "Odds ratio", "ci_low": "95% CI low", "ci_high": "95% CI high",
                 "p_value": "p-value"}),
        {"Odds ratio": f2, "95% CI low": f2, "95% CI high": f2, "p-value": lambda x: "< 0.001" if x < 0.001 else f"{x:.3f}"})

    # ---- customers
    V["cust_k"] = str(cs["k"])
    V["cust_top10"] = pct(cs["share_units_top10pct_customers"])
    V["cust_top20"] = pct(cs["share_units_top20pct_customers"])
    V["cust_one_day"] = pct(cs["share_one_purchase_day"])
    V["cust_md_share"] = pct(cs["markdown_share_all_known_lines"])
    V["cust_md_lo"] = pct(cs["markdown_share_lowest_band"])
    V["cust_md_hi"] = pct(cs["markdown_share_highest_band"])
    V["cust_sl_all"] = pct(cs["short_lived_share_all_known_units"], 1)
    V["cust_sl_lo"] = pct(cs["short_lived_share_lowest_band"], 1)
    V["cust_sl_hi"] = pct(cs["short_lived_share_highest_band"], 1)
    V["cust_rho_md"] = f2(cs["spearman_frequency_vs_markdown_share"]["rho"])
    V["cust_top_seg"] = cs["top_cltv_segment"]
    V["cust_top_seg_cust"] = pct(cs["top_cltv_segment_share_customers"])
    V["cust_top_seg_units"] = pct(cs["top_cltv_segment_share_units"])
    V["cust_top_seg_spend"] = pct(cs["top_cltv_segment_share_spend"])
    V["table_cust_segments"] = md_table(seg[["segment", "share_of_customers", "purchase_days_median", "recency_median",
                                             "items_per_basket_mean", "markdown_share_mean", "share_of_units", "share_of_spend",
                                             "cltv_index"]].rename(columns={
        "segment": "Segment", "share_of_customers": "% customers", "purchase_days_median": "Purchase days (median)",
        "recency_median": "Days since last purchase (median)", "items_per_basket_mean": "Items per basket",
        "markdown_share_mean": "Bought on markdown", "share_of_units": "% of units", "share_of_spend": "% of spend",
        "cltv_index": "CLTV (index)"}),
        {"% customers": pct, "Purchase days (median)": lambda x: f"{x:.0f}", "Days since last purchase (median)": lambda x: f"{x:.0f}",
         "Items per basket": lambda x: f"{x:.1f}", "Bought on markdown": pct, "% of units": pct, "% of spend": pct,
         "CLTV (index)": f2})
    V["table_cust_freq"] = md_table(freq[["freq_band", "share_of_customers", "share_of_units", "markdown_share", "short_lived_share"]].rename(
        columns={"freq_band": "Buys", "share_of_customers": "% of customers", "share_of_units": "% of units",
                 "markdown_share": "Bought on markdown", "short_lived_share": "Short-lived products"}),
        {"% of customers": pct, "% of units": pct, "Bought on markdown": pct, "Short-lived products": lambda x: pct(x, 1)})

    # ---- stock, forecasting, simulation
    V["vis_leftover_units"] = f"{st['leftover_units_12w_product_shop']:,.0f}"
    V["vis_leftover"] = pct(st["leftover_share_of_delivered"], 1)
    V["vis_st_median"] = pct(st["median_product_sell_through"])
    V["vis_below60"] = pct(st["share_products_sell_through_below_60pct"])
    V["vis_stockout"] = pct(st["share_shop_weeks_ending_at_zero_stock"], 1)
    V["vis_ever_so"] = pct(st["share_product_shops_ever_stocked_out_12w"], 1)
    V["vis_sl_left"] = pct(st["short_lived_leftover_share"])
    V["vis_other_left"] = pct(st["other_leftover_share"])
    V["vis_returns"] = f"{st['negative_sales_cells_clipped']:,}"
    V["vis_late_deliv"] = f"{st['units_delivered_after_week12_excluded']:,}"
    up = vis["uplift"]
    V["upl_fe"] = pct(up["uplift_per_10pp_fe"], 1)
    V["upl_fd"] = pct(up["uplift_per_10pp_fd"], 1)
    V["fc_best"] = vis["forecast_best_model"]
    V["fc_wape"] = pct(vis["forecast_best_weekly_wape"])
    V["fc_naive"] = pct(vis["forecast_naive_weekly_wape"])
    V["fc_total"] = pct(vis["forecast_best_remaining_total_wape"])
    V["fc_bias"] = pct(vis["forecast_best_bias"])
    fa = fc[fc.group == "All products"][["model", "weekly_wape", "weekly_mae", "bias", "remaining_total_wape"]]
    V["table_fc"] = md_table(fa.rename(columns={"model": "Model", "weekly_wape": "Weekly WAPE", "weekly_mae": "Weekly MAE (units)",
                                                "bias": "Bias", "remaining_total_wape": "WAPE, rest-of-season total"}),
                             {"Weekly WAPE": pct, "Weekly MAE (units)": lambda x: f"{x:.1f}", "Bias": lambda x: f"{x * 100:+.0f}%",
                              "WAPE, rest-of-season total": pct})
    lo = vis["leftover_prediction"]
    V["lo_st_corr"] = f2(lo["corr_pred_vs_actual_sell_through"])
    V["lo_st_corr_naive"] = f2(lo["naive_corr_sell_through"])
    V["lo_rec_naive"] = pct(lo["naive_flag_recall"])
    V["lo_prec_naive"] = pct(lo["naive_flag_precision"])
    V["lo_plan_corr"] = f2(lo["plan_only_corr_leftover_units"])
    V["lo_prec"] = pct(lo["high_leftover_precision"])
    V["lo_rec"] = pct(lo["high_leftover_recall"])
    V["lo_test_n"] = f"{lo['test_products']:,}"
    V["ttp_rho"] = f2(st["spearman_time_to_peak_vs_leftover_share"])
    V["ttp_p"] = f2(st["spearman_time_to_peak_p_value"])
    fl = c("vis_leftover_flag_by_model.csv")
    V["table_flag"] = md_table(fl[fl.flag_recall.notna()][["model", "corr_sell_through", "flag_recall", "flag_precision", "share_flagged"]].rename(
        columns={"model": "Model", "corr_sell_through": "Sell-through correlation", "flag_recall": "Caught (recall)",
                 "flag_precision": "Precision", "share_flagged": "% flagged"}),
        {"Sell-through correlation": f2, "Caught (recall)": pct, "Precision": pct, "% flagged": pct})
    tr_ = vis["markdown_trigger"]
    V["trig_rec"] = f"{tr_['median_recommended_week']:.0f}"
    V["trig_act"] = f"{tr_['median_actual_first_markdown_week']:.0f}"
    V["trig_earlier"] = pct(tr_["share_recommended_earlier_than_actual"])
    g = grid[(grid.flag_threshold_pred_sell_through == 0.6) & (grid.uplift_estimate == "FE (conservative)")]
    main = g[(g.policy == "A+B") & (g.delivery_cut == 0.3) & (g.markdown_level == 0.2)].iloc[0]
    a30 = g[(g.policy == "A: cut later deliveries") & (g.delivery_cut == 0.3)].iloc[0]
    b20 = g[(g.policy == "B: earlier markdown") & (g.markdown_level == 0.2)].iloc[0]
    V["sim_flagged"] = pct(main.share_products_flagged)
    V["sim_base_left"] = f"{vis['simulation_baseline_test_leftover_units']:,.0f}"
    for key, r in (("main", main), ("a30", a30), ("b20", b20)):
        V[f"{key}_left"] = signed(r.leftover_change)
        V[f"{key}_left_red"] = pct(-r.leftover_change, 1)
        V[f"{key}_sold"] = signed(r.sold_change)
        V[f"{key}_rev"] = signed(r.revenue_change)
        V[f"{key}_deliv"] = signed(r.delivered_change)
        V[f"{key}_units"] = f"{r.units_leftover_avoided:,.0f}"
        V[f"{key}_be"] = pct(r.breakeven_cost_to_price, 1) if pd.notna(r.breakeven_cost_to_price) else "n/a"
    imp = vis["simulation_main_impact"]
    V["main_t_lo"], V["main_t_hi"] = f"{imp['tonnes_textile_low']:.0f}", f"{imp['tonnes_textile_high']:.0f}"
    V["main_co2_lo"], V["main_co2_hi"] = f"{imp['tonnes_co2e_low']:.0f}", f"{imp['tonnes_co2e_high']:.0f}"
    V["kg_lo"], V["kg_hi"] = f"{GARMENT_KG['low']}", f"{GARMENT_KG['high']}"
    V["co2_lo"], V["co2_hi"] = f"{CO2E_PER_KG['low']}", f"{CO2E_PER_KG['high']:.0f}"
    fd = grid[(grid.flag_threshold_pred_sell_through == 0.6) & (grid.uplift_estimate == "First-difference")
              & (grid.policy == "A+B") & (grid.delivery_cut == 0.3) & (grid.markdown_level == 0.2)].iloc[0]
    V["main_left_fd"] = signed(fd.leftover_change)
    V["main_rev_fd"] = signed(fd.revenue_change)
    sg = g[(g.policy == "A+B") & (g.markdown_level == 0.2)].sort_values("delivery_cut")
    V["table_sim"] = md_table(pd.concat([g[g.policy == "A: cut later deliveries"].sort_values("delivery_cut"),
                                         g[g.policy == "B: earlier markdown"], sg])[
        ["policy", "delivery_cut", "markdown_level", "leftover_change", "sold_change", "revenue_change",
         "units_leftover_avoided", "breakeven_cost_to_price"]].rename(columns={
        "policy": "Policy", "delivery_cut": "Delivery cut", "markdown_level": "Markdown", "leftover_change": "Unsold stock",
        "sold_change": "Units sold", "revenue_change": "Revenue", "units_leftover_avoided": "Units avoided",
        "breakeven_cost_to_price": "Break-even cost/price"}),
        {"Delivery cut": lambda x: pct(x) if x else "-", "Markdown": lambda x: pct(x) if x else "-",
         "Unsold stock": signed, "Units sold": signed, "Revenue": signed, "Units avoided": lambda x: f"{x:,.0f}",
         "Break-even cost/price": lambda x: pct(x, 1) if pd.notna(x) else "n/a"})

    # ---- financial statements built from the data
    a = fs["all"]
    V["fin_markdown_rate"] = pct(a["markdown_rate"], 1)
    V["fin_s2s"] = pct(a["stock_to_sales"])
    V["fin_s2s_lo"], V["fin_s2s_hi"] = (pct(x) for x in fs["stock_to_sales_range"])
    V["fin_md_lo"], V["fin_md_hi"] = (pct(x, 1) for x in fs["markdown_rate_range"])
    V["fin_turnover"] = f"{a['inventory_turnover_12w']:.1f}"
    V["fin_doi"] = f"{a['days_of_inventory']:.0f}"
    V["fin_sell_through"] = pct(a["sell_through"])
    for k in (40, 50, 60):
        V[f"fin_gm{k}"] = pct(a[f"gross_margin_cost{k}"])
        V[f"fin_mu{k}"] = pct(a[f"margin_after_unsold_cost{k}"])
        V[f"fin_gp{k}"] = signed(fs["policy_gross_profit_change"][f"cost{k}"])
    V["fin_rev_change"] = signed(fs["policy_revenue_change"])
    V["fin_buy_change"] = signed(fs["policy_full_price_value_delivered_change"])
    show = fst.set_index("season")
    V["table_fin"] = md_table(show.reset_index()[["season", "units_delivered", "sell_through", "net_revenue", "markdown_cost",
                                                  "closing_stock_value", "markdown_rate", "stock_to_sales", "inventory_turnover_12w",
                                                  "gross_margin_cost50", "margin_after_unsold_cost50"]].rename(columns={
        "season": "Season", "units_delivered": "Units delivered", "sell_through": "Sell-through", "net_revenue": "Net revenue",
        "markdown_cost": "Markdown cost", "closing_stock_value": "Unsold stock (full price)", "markdown_rate": "Markdown rate",
        "stock_to_sales": "Unsold stock / revenue", "inventory_turnover_12w": "Turnover (12 wks)",
        "gross_margin_cost50": "Gross margin*", "margin_after_unsold_cost50": "Margin if unsold lost*"}),
        {"Units delivered": lambda x: f"{x:,.0f}", "Sell-through": pct, "Net revenue": lambda x: f"{x:,.0f}",
         "Markdown cost": lambda x: f"{x:,.0f}", "Unsold stock (full price)": lambda x: f"{x:,.0f}", "Markdown rate": lambda x: pct(x, 1),
         "Unsold stock / revenue": pct, "Turnover (12 wks)": lambda x: f"{x:.2f}", "Gross margin*": pct, "Margin if unsold lost*": pct})
    V["table_fin_policy"] = md_table(fpol[["scenario", "units_delivered", "units_unsold", "net_revenue", "full_price_value_delivered",
                                           "gross_profit_after_unsold_cost40", "gross_profit_after_unsold_cost50",
                                           "gross_profit_after_unsold_cost60"]].rename(columns={
        "scenario": "2019 seasons", "units_delivered": "Units delivered", "units_unsold": "Units unsold", "net_revenue": "Net revenue",
        "full_price_value_delivered": "Stock bought (full price)", "gross_profit_after_unsold_cost40": "Gross profit, cost 40%*",
        "gross_profit_after_unsold_cost50": "Gross profit, cost 50%*", "gross_profit_after_unsold_cost60": "Gross profit, cost 60%*"}),
        {c_: (lambda x: f"{x:,.0f}") for c_ in ["Units delivered", "Units unsold", "Net revenue", "Stock bought (full price)",
                                                 "Gross profit, cost 40%*", "Gross profit, cost 50%*", "Gross profit, cost 60%*"]})

    # ---- market trends
    tr = j("trends_summary.json")
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
              "October", "November", "December"]
    V["tr_window"] = tr["window"]
    V["tr_peak_month"] = months[tr["seasonal_peak_month"] - 1]
    V["tr_peak_pct"] = f"{(tr['seasonal_peak_index'] - 1) * 100:.0f}%"
    V["tr_trough_month"] = months[tr["seasonal_trough_month"] - 1]
    V["tr_trough_pct"] = f"{(1 - tr['seasonal_trough_index']) * 100:.0f}%"
    V["tr_yoy"] = f"{tr['yoy_growth_jun_dec'] * 100:+.0f}%"
    gtv = tr["google_trends"]
    V["gt_pre_with"] = pct(gtv["prelaunch_total_sales"]["with Google Trends"]["wape"], 1)
    V["gt_pre_without"] = pct(gtv["prelaunch_total_sales"]["without Google Trends"]["wape"], 1)
    V["gt_w2_with"] = pct(gtv["week2_weekly_wape"]["with Google Trends"], 1)
    V["gt_w2_without"] = pct(gtv["week2_weekly_wape"]["without Google Trends"], 1)

    # ---- pilot
    pil = j("pilot_design.json")
    V["pil_effect"] = f"{pil['expected_effect_pp']:.1f}"
    V["pil_sd"] = f"{pil['between_shop_sd_pp']:.1f}"
    V["pil_r"] = f2(pil["prior_season_correlation"])
    V["pil_shops"] = str(pil["shops_per_arm_full_effect_simple"])
    V["pil_shops_half"] = str(pil["shops_per_arm_half_effect_ancova"])
    V["pil_shops_avail"] = str(pil["shops_available_per_season"])
    V["pil_prod_effect"] = f"{pil['product_design_effect_pp']:.1f}"
    V["pil_prod_n"] = str(pil["products_per_arm_full_effect"])
    V["pil_prod_n_half"] = str(pil["products_per_arm_half_effect"])
    V["pil_flagged"] = str(pil["flagged_products_per_season"])
    ps = c("pilot_sample_size.csv")
    V["table_pilot"] = md_table(ps[["design", "effect_as_share_of_simulated", "effect_pp", "units_per_arm"]].rename(columns={
        "design": "Design", "effect_as_share_of_simulated": "True effect vs simulated", "effect_pp": "Drop in leftover share (pp)",
        "units_per_arm": "Units per arm (80% power, 5% level)"}),
        {"True effect vs simulated": pct, "Drop in leftover share (pp)": lambda x: f"{x:.1f}",
         "Units per arm (80% power, 5% level)": lambda x: f"{int(x)}"})
    for k, (claim, value, src) in LITERATURE.items():
        V[k] = value
        V[k + "_src"] = src

    # ---- Review 2 (mid-project) package: relationships behind Q5 (src/review2_data.py)
    r2 = json.loads((ROOT / "outputs" / "review2" / "review2_data.json").read_text())["stats"]
    V["r2_dps_rho"] = f2(r2["dps_rho"])
    V["r2_early_rho"] = f2(r2["early_rho"])
    V["r2_dps_lo"], V["r2_dps_hi"] = pct(r2["dps_d10_st"]), pct(r2["dps_d1_st"])
    V["r2_early_lo"], V["r2_early_hi"] = pct(r2["early_d1_st"]), pct(r2["early_d10_st"])
    V["r2_cat_lo"], V["r2_cat_hi"] = pct(r2["cat_left_lo"]), pct(r2["cat_left_hi"])

    # ---- cleaning log
    dq = pd.read_csv(TAB / "data_quality_log.csv")
    V["dq_checks"] = str(len(dq))
    V["dq_vis_neg"] = f"{st['negative_sales_cells_clipped']:,}"
    V["table_dq"] = md_table(dq.rename(columns={"dataset": "Dataset", "check": "Check", "what_we_found": "What we found",
                                                "how_handled": "How we handled it"}))
    if survey.exists():
        V.update(survey_values(json.loads(survey.read_text())))
    else:
        V["survey_status"] = ("**Pending — the questionnaire is ready (survey/questionnaire.md) but no real responses "
                              "have been collected yet.** No survey result appears anywhere in this report.")
        V["survey_conclusion"] = "[TBD: the consumer survey will add how long clothes are kept and what durability is worth to buyers.]"
        for t in TEMPLATES:
            for k in re.findall(r"\{\{((?:table_)?sv_\w+)\}\}", (REPORTS / t).read_text()):
                V[k] = "[TBD]"
    return V


def survey_values(sv):
    """Tokens for the consumer survey, from src/survey_analysis.py outputs (real responses only)."""
    q = j("survey_quality.json")
    V = {}
    disp, never = sv["disposal_share_often_or_always"], sv["disposal_share_never"]
    att, mot = sv["attitude_share_agree_4or5"], sv["motive_means_1to5"]
    imp, wtp, ci = sv["conjoint_importance"], sv["wtp_rs"], sv["wtp_rs_ci95"]
    direct = sv["direct_long_life_vs_cheap"]["P6_vs_P1"]
    rob = sv["robustness_v2_only"]
    rs = lambda x: f"Rs {x:,.0f}"
    V["sv_n"] = str(sv["respondents"])
    V["sv_first"] = pd.Timestamp(q["first_response"]).strftime("%-d %b %Y")
    V["sv_last"] = pd.Timestamp(q["last_response"]).strftime("%-d %b %Y")
    V["sv_moe"] = f"{q['margin_of_error_95_for_a_share'] * 100:.0f}"
    V["sv_v1"] = str(q["form_versions"].get("v1 original", 0))
    V["sv_v2"] = str(q["form_versions"].get("v2 edited", 0))
    n_flat = q["conjoint_all_same_rating_excluded"]
    V["sv_straight"] = f"{n_flat} respondent" + ("" if n_flat == 1 else "s")
    V["sv_q9_missing"] = str(q["q9_unanswered"])
    V["sv_comments"] = str(q["free_text_comments_not_shown"])
    V["sv_age1824"] = pct(sv["share_age_18_24"])
    V["sv_students"] = pct(sv["share_students"])
    V["sv_keep_mean"] = f"{sv['mean_keep_years']:.1f}"
    V["sv_keep_lt1"] = pct(sv["share_keep_under_1_year"])
    V["sv_keep_gt3"] = pct(sv["share_keep_over_3_years"])
    V["sv_wears_lt10"] = pct(sv["share_wears_under_10"])
    V["sv_unworn"] = pct(sv["mean_unworn_share"])
    V["sv_unworn_gt25"] = pct(sv["share_unworn_over_25pct"])
    V["sv_impulse"] = pct(sv["share_unplanned_3plus_of_answered"])
    V["sv_impulse_all"] = pct(sv["share_unplanned_3plus_of_answered"] * (sv["respondents"] - q["q9_unanswered"]) / sv["respondents"])
    V["sv_ff"] = pct(sv["share_buys_fast_fashion"])
    V["sv_donate"] = pct(disp["Donate"])
    V["sv_handdown"] = pct(disp["Hand down"])
    V["sv_keepunused"] = pct(disp["Keep unused"])
    V["sv_throw"] = pct(disp["Throw away"])
    V["sv_resell"] = pct(disp["Sell / resell"])
    V["sv_recycle"] = pct(disp["Recycle / take-back"])
    V["sv_resell_never"] = pct(never["Sell / resell"])
    V["sv_recycle_never"] = pct(never["Recycle / take-back"])
    for k, name in [("discount", "Discount"), ("quality", "Quality"), ("novelty", "Novelty"), ("trend", "Trend"),
                    ("env", "Environment")]:
        V[f"sv_mot_{k}"] = f"{mot[name]:.1f}"
    V["sv_top_motive"] = sv["top_motive"].lower()
    V["sv_att_harm"] = pct(att["Fast fashion harms environment"])
    V["sv_att_fewer"] = pct(att["Would buy fewer if they lasted longer"])
    V["sv_att_pay"] = pct(att["Would pay more for longer-lasting"])
    V["sv_att_brands"] = pct(att["Brands should produce fewer items"])
    V["sv_att_know"] = pct(att["Knows clothes use water/energy"])
    V["sv_conj_n"] = str(sv["conjoint_respondents"])
    V["sv_imp_life"] = pct(imp["Lifespan"])
    V["sv_imp_style"] = pct(imp["Style (trendy)"])
    V["sv_imp_price"] = pct(imp["Price"])
    V["sv_imp_label"] = pct(imp["Sustainability label"])
    for k in ("long_life", "certified", "trendy"):
        short = {"long_life": "life", "certified": "label", "trendy": "trendy"}[k]
        V[f"sv_wtp_{short}"] = rs(wtp[k])
        V[f"sv_wtp_{short}_ci"] = f"{rs(ci[k][0])}-{ci[k][1]:,.0f}"
    V["sv_direct_gain"] = f"{direct['mean_rating_difference']:+.2f}"
    V["sv_direct_higher"] = pct(direct["share_rated_long_life_higher"])
    V["sv_direct_lower"] = pct(direct["share_rated_cheap_higher"])
    fmt_p = lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
    V["sv_rho_freq"] = f"{sv['rank_corr_frequency_vs_trend']['rho']:.2f}"
    V["sv_p_freq"] = fmt_p(sv["rank_corr_frequency_vs_trend"]["p_value"])
    V["sv_rho_age"] = f"{sv['rank_corr_age_vs_trend']['rho']:.2f}"
    V["sv_p_age"] = fmt_p(sv["rank_corr_age_vs_trend"]["p_value"])
    V["sv_rob_rho_freq"] = f"{rob['rank_corr_frequency_vs_trend']['rho']:.2f}"
    V["sv_rob_p_freq"] = fmt_p(rob["rank_corr_frequency_vs_trend"]["p_value"])
    ver = sv["by_form_version"]
    V["sv_v1_trend"] = f"{ver['v1 original']['trend']:.1f}" if "v1 original" in ver else "n/a"
    V["sv_v2_trend"] = f"{ver['v2 edited']['trend']:.1f}" if "v2 edited" in ver else "n/a"

    pw = c("survey_conjoint_partworths.csv")
    V["table_sv_conjoint"] = md_table(pw.assign(
        ci=[f"{a:+.2f} to {b:+.2f}" for a, b in zip(pw.ci_low, pw.ci_high)])[["label", "part_worth", "ci", "p_value"]].rename(
        columns={"label": "Feature (vs baseline)", "part_worth": "Effect on 1-10 rating", "ci": "95% CI",
                 "p_value": "p-value"}),
        {"Effect on 1-10 rating": lambda x: f"{x:+.2f}", "p-value": fmt_p})
    seg = c("survey_segments.csv").sort_values("keep_years").reset_index(drop=True)
    V["table_sv_segments"] = md_table(seg[["name", "respondents", "Trend", "Novelty", "Quality", "Environment",
                                           "keep_years", "throw_away_often", "buys_fast_fashion"]].rename(columns={
        "name": "Shopper group", "respondents": "People", "Trend": "Trend (1-5)", "Novelty": "Novelty (1-5)",
        "Quality": "Quality (1-5)", "Environment": "Environment (1-5)", "keep_years": "Keeps clothes (years)",
        "throw_away_often": "Often throws away", "buys_fast_fashion": "Buys fast fashion"}),
        {"People": lambda x: str(int(x)), "Trend (1-5)": lambda x: f"{x:.1f}", "Novelty (1-5)": lambda x: f"{x:.1f}",
         "Quality (1-5)": lambda x: f"{x:.1f}", "Environment (1-5)": lambda x: f"{x:.1f}",
         "Keeps clothes (years)": lambda x: f"{x:.1f}", "Often throws away": pct, "Buys fast fashion": pct})
    for tag, r in (("segt", seg.iloc[0]), ("segl", seg.iloc[-1])):
        V[f"sv_{tag}_name"] = r["name"]
        V[f"sv_{tag}_share"] = pct(r["share"])
        V[f"sv_{tag}_keep"] = f"{r.keep_years:.1f}"
        V[f"sv_{tag}_throw"] = pct(r.throw_away_often)
        V[f"sv_{tag}_ff"] = pct(r.buys_fast_fashion)
        V[f"sv_{tag}_throw_phrase"] = "nobody" if r.throw_away_often == 0 else f"{pct(r.throw_away_often)} of the group"
    bf = c("survey_vis_bridge_by_frequency.csv")
    V["sv_freq_trend_lo"] = f"{bf.trend_influence.iloc[0]:.1f}"
    V["sv_freq_trend_hi"] = f"{bf.trend_influence.iloc[-1]:.1f}"
    V["sv_freq_disc_lo"] = f"{bf.discount_influence.iloc[0]:.1f}"
    V["sv_freq_disc_hi"] = f"{bf.discount_influence.iloc[-1]:.1f}"
    V["sv_rho_freq_disc"] = f"{sv['rank_corr_frequency_vs_discount']['rho']:.2f}"
    V["sv_p_freq_disc"] = fmt_p(sv["rank_corr_frequency_vs_discount"]["p_value"])
    V["table_sv_bridge_freq"] = md_table(bf[["freq_band", "respondents", "trend_influence", "discount_influence",
                                            "vis_customers", "vis_markdown_share", "vis_short_lived_share"]].rename(columns={
        "freq_band": "Buys clothes", "respondents": "Survey people", "trend_influence": "Survey: trend influence (1-5)",
        "discount_influence": "Survey: discount influence (1-5)", "vis_customers": "Nuna Lie customers",
        "vis_markdown_share": "Nuna Lie: bought on markdown", "vis_short_lived_share": "Nuna Lie: short-lived products"}),
        {"Survey people": lambda x: str(int(x)), "Survey: trend influence (1-5)": lambda x: f"{x:.2f}",
         "Survey: discount influence (1-5)": lambda x: f"{x:.2f}", "Nuna Lie customers": lambda x: f"{int(x):,}",
         "Nuna Lie: bought on markdown": pct, "Nuna Lie: short-lived products": lambda x: pct(x, 1)})

    V["survey_status"] = (f"Collected: {sv['respondents']} real responses ({V['sv_first']} to {V['sv_last']}), all with "
                          f"consent. Results are in section 5.9.")
    V["survey_conclusion"] = (f"Our survey of {sv['respondents']} young consumers adds the other end of the chain: "
                              f"{V['sv_wears_lt10']} wear a fast-fashion item 10 times or fewer, about {V['sv_unworn']} of the "
                              f"wardrobe went unworn last year, and lifespan was the most important feature when choosing a "
                              f"garment ({V['sv_imp_life']} of the decision).")
    return V


def render(V, template, target):
    tpl = (REPORTS / template).read_text()
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", tpl)) - set(V))
    if missing:
        raise KeyError(f"{template}: tokens without a computed value: {missing}")
    out = re.sub(r"\{\{(\w+)\}\}", lambda m: str(V[m.group(1)]), tpl)
    (REPORTS / target).write_text(out)
    return len(out.split())


def build():
    V = values()
    n = render(V, "final_report_template.md", "final_report.md")
    for t in TEMPLATES[1:]:
        render(V, t, t.replace("_template", ""))
    (TAB / "deck_data.json").write_text(json.dumps(V, indent=2, default=str))
    print(f"wrote reports/final_report.md ({n} words), reports/viva_notes.md, reports/project_summary.md, reports/review2_guide.md and outputs/tables/deck_data.json")


if __name__ == "__main__":
    build()
