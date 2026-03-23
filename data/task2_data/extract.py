"""
Download a subset of NGA painting images + metadata from local opendata CSVs.

Expects in the same directory as this script:
  - objects.csv
  - objects_terms.csv
  - published_images.csv

Outputs:
  - nga_paintings_subset/all/<objectid>.jpg  (ImageFolder-friendly)
  - nga_paintings_subset/metadata.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Paths (relative to this script — not /mnt/data)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OBJECTS_PATH = BASE_DIR / "objects.csv"
TERMS_PATH = BASE_DIR / "objects_terms.csv"
IMAGES_PATH = BASE_DIR / "published_images.csv"

SAMPLE_SIZE = 2000
MIN_EDGE_PX = 500
IIIF_WIDTH = 800

# NGA CDN may reject requests without a User-Agent
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NGA-research-download/1.0; +https://github.com/NationalGalleryOfArt/opendata)",
}


def main() -> int:
    for p in (OBJECTS_PATH, TERMS_PATH, IMAGES_PATH):
        if not p.is_file():
            print(f"Missing required file: {p}", file=sys.stderr)
            return 1

    # -------------------------------------------------------------------------
    # 1. Load CSVs (NGA exports use lowercase column names)
    # -------------------------------------------------------------------------
    objects = pd.read_csv(OBJECTS_PATH, low_memory=False)
    terms = pd.read_csv(TERMS_PATH, low_memory=False)
    images = pd.read_csv(IMAGES_PATH, low_memory=False)

    print("Objects:", len(objects))
    print("Terms:", len(terms))
    print("Images:", len(images))

    # -------------------------------------------------------------------------
    # 2. Filter: physical paintings only (not virtual / deaccessioned placeholders)
    # -------------------------------------------------------------------------
    cls = objects["classification"].fillna("").astype(str).str.strip().str.lower()
    paintings = objects[(cls == "painting") & (objects["isvirtual"] == 0)].copy()

    print("Paintings after filtering:", len(paintings))

    # -------------------------------------------------------------------------
    # 3. Join published_images (primary view only)
    # -------------------------------------------------------------------------
    primary_images = images[images["viewtype"].astype(str).str.lower() == "primary"].copy()

    merged = paintings.merge(
        primary_images,
        left_on="objectid",
        right_on="depictstmsobjectid",
        how="inner",
        suffixes=("_obj", "_img"),
    )

    print("Paintings with primary images:", len(merged))

    # -------------------------------------------------------------------------
    # 4. Image size filter (published image pixel dimensions)
    # -------------------------------------------------------------------------
    merged = merged[
        (pd.to_numeric(merged["width"], errors="coerce") >= MIN_EDGE_PX)
        & (pd.to_numeric(merged["height"], errors="coerce") >= MIN_EDGE_PX)
    ]

    print("After size filtering:", len(merged))

    # -------------------------------------------------------------------------
    # 5. Drop rows without IIIF base URL; dedupe by object
    # -------------------------------------------------------------------------
    iiif_col = "iiifurl" if "iiifurl" in merged.columns else "iiifURL"
    merged = merged[merged[iiif_col].notna() & (merged[iiif_col].astype(str).str.len() > 0)]
    merged = merged.drop_duplicates(subset=["objectid"], keep="first")

    print("After cleaning:", len(merged))

    # -------------------------------------------------------------------------
    # 6. Sample
    # -------------------------------------------------------------------------
    if len(merged) > SAMPLE_SIZE:
        merged_sample = merged.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        merged_sample = merged.copy()

    print("Final dataset size:", len(merged_sample))

    # -------------------------------------------------------------------------
    # 7. Optional: attach AAT / browse terms (School, Style, …) for richer metadata
    # -------------------------------------------------------------------------
    term_types = {"School", "Style", "Subject"}
    if "termtype" in terms.columns and "objectid" in terms.columns:
        t = terms[terms["termtype"].isin(term_types)].copy()
        terms_agg = (
            t.groupby("objectid", as_index=False)["term"]
            .agg(lambda s: " | ".join(sorted({str(x).strip() for x in s.dropna() if str(x).strip()})))
            .rename(columns={"term": "style_terms"})
        )
        merged_sample = merged_sample.merge(terms_agg, on="objectid", how="left")
    else:
        merged_sample["style_terms"] = pd.NA

    # -------------------------------------------------------------------------
    # 8. Download directory: ImageFolder needs one class folder
    # -------------------------------------------------------------------------
    save_root = BASE_DIR / "nga_paintings_subset"
    save_dir = save_root / "all"
    save_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 9. IIIF URL (NGA iiifurl is the image service base, no trailing slash issues)
    # -------------------------------------------------------------------------
    def build_iiif_url(base_url: str, width: int = IIIF_WIDTH) -> str:
        base = str(base_url).rstrip("/")
        return f"{base}/full/{width},/0/default.jpg"

    print("Downloading images...")

    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)

    for _, row in tqdm(merged_sample.iterrows(), total=len(merged_sample)):
        oid = int(row["objectid"])
        base = row[iiif_col]
        img_url = build_iiif_url(base, IIIF_WIDTH)
        save_path = save_dir / f"{oid}.jpg"

        try:
            r = session.get(img_url, timeout=30)
            if r.status_code == 200 and len(r.content) > 0:
                save_path.write_bytes(r.content)
            else:
                print(f"Failed HTTP {r.status_code}: objectid={oid}")
        except Exception as e:
            print(f"Error downloading objectid={oid}: {e}")

    print("Download complete.")

    # -------------------------------------------------------------------------
    # 10. Metadata for Task 2 notebook (filename relative to nga_paintings_subset)
    # -------------------------------------------------------------------------
    meta = pd.DataFrame(
        {
            "filename": merged_sample["objectid"].astype(int).map(lambda x: f"all/{x}.jpg"),
            "objectid": merged_sample["objectid"].astype(int),
            "title": merged_sample.get("title", pd.Series([pd.NA] * len(merged_sample))),
            "artist": merged_sample.get(
                "attribution",
                merged_sample.get("attributioninverted", pd.Series([pd.NA] * len(merged_sample))),
            ),
            "classification": merged_sample.get("classification", pd.Series([pd.NA] * len(merged_sample))),
            "beginyear": merged_sample.get("beginyear", pd.Series([pd.NA] * len(merged_sample))),
            "style_terms": merged_sample.get("style_terms", pd.Series([pd.NA] * len(merged_sample))),
            "iiifurl": merged_sample[iiif_col],
            "width": merged_sample.get("width", pd.Series([pd.NA] * len(merged_sample))),
            "height": merged_sample.get("height", pd.Series([pd.NA] * len(merged_sample))),
        }
    )

    out_csv = save_root / "metadata.csv"
    meta.to_csv(out_csv, index=False)
    print(f"Metadata saved: {out_csv}")
    print(f"Set Task 2 notebook IMAGES_ROOT to: {save_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
