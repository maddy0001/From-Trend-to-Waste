"""Data pack for running the analysis in Google Colab -> deliverables/colab/From_Trend_to_Waste_colab_pack.zip

Holds the project code (src/) plus the VISUELLE 2.0 files and the survey export each analysis step needs, so the
notebook notebooks/From_Trend_to_Waste_Colab.ipynb can rerun every algorithm from the raw data.
The pack contains licensed data (VISUELLE 2.0) and the team's survey export: keep it private (own Google Drive /
Colab only) and never commit or share it publicly (git-ignored).
"""
import zipfile

from .config import ROOT, SURVEY_RAW, VIS_RAW

OUT = ROOT / "deliverables" / "colab"
PACK = OUT / "From_Trend_to_Waste_colab_pack.zip"
VIS_FILES = ["sales.csv", "restocks.csv", "price_discount_series.csv", "customer_data.csv", "vis2_gtrends_data.csv"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = (sorted((ROOT / "src").glob("*.py")) + [VIS_RAW / f for f in VIS_FILES]
             + [ROOT / "survey" / "conjoint_design.csv"] + sorted(SURVEY_RAW.glob("*.csv")))
    with zipfile.ZipFile(PACK, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix())
    print(f"wrote {PACK} ({PACK.stat().st_size / 1e6:.0f} MB, {len(files)} files)")


if __name__ == "__main__":
    main()
