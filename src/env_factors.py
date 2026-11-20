"""Environmental conversion factors — every value cited, always used as a low-high range (estimate).

Units -> kg uses an assumed garment weight range; kg -> CO2e uses published per-kg footprints.
These are ESTIMATES for scale only: they are not measurements of the retailer's products.
"""

GARMENT_KG = {
    "low": 0.25,
    "high": 0.50,
    "source": ("Assumption for women's fashion items (tops, dresses, skirts, knitwear, light coats), "
               "consistent with practical garment-weight guides, e.g. ~5 T-shirts or 2-4 shirts per kg "
               "(https://www.indetexx.com/how-much-is-a-kg-of-clothes-garments-weight-guide/)."),
}

CO2E_PER_KG = {
    "low": 3.169,
    "high": 25.0,
    "source": ("Low: EU estimate of CO2 saved per kg of clothing reused (3.169 kg), as cited by UPC-INTEXTER; "
               "High: UPC-INTEXTER literature review, 25 kg CO2 saved per kg of clothing reused (2022), "
               "https://www.upc.edu/en/press-room/news/reusing-1-kg-of-clothing-saves-25-kg-of-co2-according-to-a-study-by-intexter . "
               "Context: WRAP (2017) puts UK clothing's whole-life footprint at 23.2 t CO2e per tonne (2016), "
               "https://www.wrap.ngo/sites/default/files/2020-10/WRAP-valuing-our-clothes-the-cost-of-uk-fashion_WRAP.pdf"),
}


def units_to_impact(units: float) -> dict:
    """Convert garment units to (low, high) tonnes of textile and tonnes CO2e."""
    t_low, t_high = units * GARMENT_KG["low"] / 1000, units * GARMENT_KG["high"] / 1000
    return {
        "units": units,
        "tonnes_textile_low": t_low,
        "tonnes_textile_high": t_high,
        "tonnes_co2e_low": t_low * CO2E_PER_KG["low"],
        "tonnes_co2e_high": t_high * CO2E_PER_KG["high"],
    }
