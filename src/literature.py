"""External facts used in the report — each with its source, registered in the claims register."""
from .utils import register_claim

LITERATURE = {
    "lit_unep_ghg": ("Textile industry share of global greenhouse-gas emissions (UNEP)", "2-8%",
                     "https://www.unep.org/news-and-stories/press-release/un-alliance-sustainable-fashion-addresses-damage-fast-fashion"),
    "lit_emf_truck": ("Textiles landfilled or burned (Ellen MacArthur Foundation, A New Textiles Economy, 2017)",
                      "the equivalent of one garbage truck every second",
                      "https://www.ellenmacarthurfoundation.org/news/one-garbage-truck-of-textiles-wasted-every-second-report-creates-vision-for-change"),
    "lit_emf_recycle": ("Clothing remade into new clothing (Ellen MacArthur Foundation, 2017)", "less than 1%",
                        "https://www.ellenmacarthurfoundation.org/news/one-garbage-truck-of-textiles-wasted-every-second-report-creates-vision-for-change"),
    "lit_wrap_life": ("Average UK garment lifetime (WRAP)", "2.2 years (2012); 3.3 years (later update)",
                      "https://www.wrap.ngo/resources/report/valuing-our-clothes-true-cost-how-we-design-use-and-dispose-clothing-uk-2012"),
    "lit_wrap_co2": ("UK clothing whole-life carbon footprint per tonne (WRAP 2017)", "23.2 t CO2e per tonne (2016)",
                     "https://www.wrap.ngo/sites/default/files/2020-10/WRAP-valuing-our-clothes-the-cost-of-uk-fashion_WRAP.pdf"),
}


def main():
    for k, (claim, value, src) in LITERATURE.items():
        register_claim(k, claim, value, src)


if __name__ == "__main__":
    main()
