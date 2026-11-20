"""Module 1 'Marketing Experiments': design of a pilot (A/B test) to prove the week-2 rule in stores.

Unit of randomisation: the SHOP (all flagged items in a treated shop follow the rule), which avoids
stock and customer spill-over between arms. Primary KPI: leftover share = unsold after 12 weeks /
delivered, per shop. Effect size and variance come from VISUELLE:
  - expected effect = shop-level change in leftover share under the main policy (simulation replay),
  - between-shop SD of leftover share within a season,
  - ANCOVA gain from the shop's previous-season leftover share (same season type, one year earlier).
"""
import numpy as np
import pandas as pd
from statsmodels.stats.power import TTestIndPower

from . import vis_forecast as vf
from .config import TAB
from .utils import PALETTE, plt, register_claim, save_json, save_table, savefig

SCRIPT = "src/pilot_design.py"
CUT, MD = 0.3, 0.2  # main policy from the simulation


def shop_outcomes(meta, S, D, Dsc, flag_codes=None, **policy):
    """Shop-level delivered / leftover under a policy, replayed with vf.simulate's stock logic."""
    flag = meta.external_code.isin(flag_codes).values if flag_codes is not None else np.zeros(len(meta), bool)
    D2 = D.copy()
    if policy.get("r_cut"):
        D2[flag, vf.ORIGIN:] *= (1 - policy["r_cut"])
    dsc2 = Dsc.copy()
    if policy.get("md_level") is not None:
        dsc2[flag, 3:] = np.maximum(dsc2[flag, 3:], policy["md_level"])
    beta = policy.get("beta", 0.0)
    demand = S * np.exp(beta * (dsc2 - Dsc))
    stock = np.zeros(len(S))
    for k in range(vf.K):
        stock += D2[:, k]
        sold = np.minimum(demand[:, k], np.maximum(stock, 0))
        stock -= sold
    out = meta[["season", "retail"]].copy()
    out["delivered"] = D2.sum(1)
    out["leftover"] = np.maximum(stock, 0)
    g = out.groupby(["season", "retail"])[["delivered", "leftover"]].sum()
    g["leftover_share"] = g.leftover / g.delivered
    return g.reset_index()


def product_outcomes(meta, S, D, Dsc, flag_codes=None, **policy):
    """Product-level delivered / leftover (summed over shops) under a policy."""
    flag = meta.external_code.isin(flag_codes).values if flag_codes is not None else np.zeros(len(meta), bool)
    D2 = D.copy()
    if policy.get("r_cut"):
        D2[flag, vf.ORIGIN:] *= (1 - policy["r_cut"])
    dsc2 = Dsc.copy()
    if policy.get("md_level") is not None:
        dsc2[flag, 3:] = np.maximum(dsc2[flag, 3:], policy["md_level"])
    demand = S * np.exp(policy.get("beta", 0.0) * (dsc2 - Dsc))
    stock = np.zeros(len(S))
    for k in range(vf.K):
        stock += D2[:, k]
        stock -= np.minimum(demand[:, k], np.maximum(stock, 0))
    out = meta[["season", "external_code"]].copy()
    out["delivered"], out["leftover"] = D2.sum(1), np.maximum(stock, 0)
    g = out[out.season.isin(vf.TEST)].groupby(["season", "external_code"])[["delivered", "leftover"]].sum()
    g["leftover_share"] = g.leftover / g.delivered
    return g.reset_index()


def main():
    meta, S, D, Dsc, _ = vf.load()
    vis = __import__("json").loads((TAB / "vis_summary.json").read_text())
    beta = vis["uplift"]["beta_fe"]
    pred = pd.read_csv(TAB / "vis_test_leftover_predictions.csv")
    flagged = set(pred.loc[pred.pred_sell_through < 0.6, "external_code"])

    base = shop_outcomes(meta, S, D, Dsc)
    test = base[base.season.isin(vf.TEST)]
    pol = shop_outcomes(meta, S, D, Dsc, flagged, r_cut=CUT, md_level=MD, beta=beta)
    pol = pol[pol.season.isin(vf.TEST)]
    eff = test.merge(pol, on=["season", "retail"], suffixes=("_base", "_policy"))
    eff["delta"] = eff.leftover_share_policy - eff.leftover_share_base
    effect = float(-eff.delta.mean())                      # expected drop in leftover share (percentage points)
    sd = float(test.groupby("season").leftover_share.std().mean())  # between-shop SD within a season

    # ANCOVA gain: correlation of a shop's leftover share with its share one year earlier (same season type)
    pairs = [("SS18", "SS19"), ("AW18", "AW19"), ("SS17", "SS18"), ("AW17", "AW18")]
    rs = []
    for a, b in pairs:
        m = base[base.season == a].merge(base[base.season == b], on="retail")
        if len(m) > 10:
            rs.append(np.corrcoef(m.leftover_share_x, m.leftover_share_y)[0, 1])
    r = float(np.mean(rs))
    sd_adj = sd * np.sqrt(1 - r ** 2)

    power = TTestIndPower()
    rows = []
    for frac in (0.5, 0.75, 1.0):
        d_raw, d_adj = effect * frac / sd, effect * frac / sd_adj
        rows.append({"effect_as_share_of_simulated": frac, "effect_pp": effect * frac * 100,
                     "shops_per_arm_simple": int(np.ceil(power.solve_power(d_raw, alpha=0.05, power=0.8))),
                     "shops_per_arm_ancova": int(np.ceil(power.solve_power(d_adj, alpha=0.05, power=0.8)))})
    tab = pd.DataFrame(rows)
    tab.insert(0, "design", "Randomise shops")

    # Alternative: randomise the FLAGGED PRODUCTS (half follow the rule chain-wide, half do not).
    # The effect is concentrated on these items, so far fewer units are needed; the risk is
    # substitution between treated and control items, which the analysis must check.
    def product_share(g):
        g = g.groupby(["season", "external_code"])[["delivered", "leftover"]].sum()
        return g.leftover / g.delivered
    base_p = product_outcomes(meta, S, D, Dsc)
    pol_p = product_outcomes(meta, S, D, Dsc, flagged, r_cut=CUT, md_level=MD, beta=beta)
    fp = base_p[base_p.external_code.isin(flagged)].merge(pol_p, on=["season", "external_code"], suffixes=("_base", "_policy"))
    p_effect = float((fp.leftover_share_base - fp.leftover_share_policy).mean())
    p_sd = float(fp.leftover_share_base.std())
    flagged_per_season = int(fp.groupby("season").size().min())
    prow = []
    for frac in (0.5, 0.75, 1.0):
        prow.append({"design": "Randomise flagged products", "effect_as_share_of_simulated": frac,
                     "effect_pp": p_effect * frac * 100,
                     "shops_per_arm_simple": int(np.ceil(power.solve_power(p_effect * frac / p_sd, alpha=0.05, power=0.8))),
                     "shops_per_arm_ancova": np.nan})
    tab = pd.concat([tab, pd.DataFrame(prow)], ignore_index=True).rename(
        columns={"shops_per_arm_simple": "units_per_arm", "shops_per_arm_ancova": "units_per_arm_ancova"})
    save_table(tab, "pilot_sample_size")
    shops_per_season = int(test.groupby("season").retail.nunique().min())
    res = {"base_leftover_share_test_shops_mean": float(test.leftover_share.mean()),
           "expected_effect_pp": effect * 100, "between_shop_sd_pp": sd * 100,
           "prior_season_correlation": r, "sd_after_ancova_pp": sd_adj * 100,
           "shops_per_arm_full_effect_simple": int(tab.iloc[2].units_per_arm),
           "shops_per_arm_full_effect_ancova": int(tab.iloc[2].units_per_arm_ancova),
           "shops_per_arm_half_effect_ancova": int(tab.iloc[0].units_per_arm_ancova),
           "product_design_effect_pp": p_effect * 100, "product_design_sd_pp": p_sd * 100,
           "products_per_arm_full_effect": int(tab.iloc[5].units_per_arm),
           "products_per_arm_half_effect": int(tab.iloc[3].units_per_arm),
           "flagged_products_per_season": flagged_per_season,
           "shops_available_per_season": shops_per_season,
           "share_shops_improving": float((eff.delta < 0).mean())}
    save_json(res, "pilot_design")
    print(res)
    print(tab)

    src = "outputs/tables/pilot_design.json"
    register_claim("pilot_effect_pp", "Expected drop in shop leftover share under the main policy (pp)", res["expected_effect_pp"], src)
    register_claim("pilot_sd_pp", "Between-shop SD of leftover share within a season (pp)", res["between_shop_sd_pp"], src)
    register_claim("pilot_r", "Correlation of shop leftover share with previous year's same season", r, src)
    register_claim("pilot_n_ancova", "Shops per arm for 80% power (full effect, ANCOVA)", res["shops_per_arm_full_effect_ancova"], src)
    register_claim("pilot_n_simple", "Shops per arm for 80% power (full effect, simple t-test)", res["shops_per_arm_full_effect_simple"], src)
    register_claim("pilot_n_half", "Shops per arm for 80% power (half the effect, ANCOVA)", res["shops_per_arm_half_effect_ancova"], src)
    register_claim("pilot_prod_effect", "Flagged products: expected drop in leftover share (pp)", res["product_design_effect_pp"], src)
    register_claim("pilot_prod_n", "Flagged products per arm for 80% power (full effect)", res["products_per_arm_full_effect"], src)
    register_claim("pilot_prod_n_half", "Flagged products per arm for 80% power (half effect)", res["products_per_arm_half_effect"], src)
    register_claim("pilot_flagged_per_season", "Flagged products per season (2019 test seasons, min)", res["flagged_products_per_season"], src)

    ns = np.arange(5, 81)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for frac, col in ((1.0, PALETTE["accent"]), (0.5, PALETTE["teal"])):
        for sdv, ls, lab in ((sd_adj, "-", "ANCOVA (prior-season control)"), (sd, "--", "simple comparison")):
            pw = [power.power(effect * frac / sdv, nobs1=n, alpha=0.05) for n in ns]
            ax.plot(ns, np.array(pw) * 100, color=col, ls=ls, lw=2,
                    label=f"{int(frac * 100)}% of simulated effect, {lab}")
    ax.axhline(80, color=PALETTE["ink"], lw=1)
    ax.set_xlabel("Shops per arm")
    ax.set_ylabel("Power (%)")
    ax.set_title("Pilot size needed to detect the week-2 rule")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    savefig(fig, "pilot_power")


if __name__ == "__main__":
    main()
