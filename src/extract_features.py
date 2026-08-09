"""
Feature extraction for the Amfitrite Inland Waters HAB (Sentinel-2) dataset.

For every tile (uid folder) we:
  1. Load the Scene Classification Layer (SCL) and keep only pixels labeled
     "water" (SCL == 6), per the dataset README's recommendation to mask out
     land/cloud pixels before training.
  2. Load the raw spectral bands and compute water-masked summary statistics
     (mean/std/median) per band.
  3. Compute a handful of physically-motivated spectral indices commonly used
     for water quality / cyanobacteria remote sensing (NDCI, 3BDA/phycocyanin
     proxy, FAI, NDVI, NDWI, blue:green ratio) and summarize them the same way.

Deliberately excluded: any column from dataset_summary.xlsx that is derived
from the CyFi model's own predictions (high_pred/mod_pred/low_pred,
roughness_median, outlier_fraction, compatible_severities, HAB_status, abun,
abun_class). Those are literally the inputs used to build the `indicative_class`
label, so using them as model features would leak the label. Only raw pixel
values (which we read and aggregate ourselves) and independent metadata
(cloud %, water pixel count, acquisition month) are used as features.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import traceback

import numpy as np
import pandas as pd
import rasterio

WATER_CLASS = 6

# Sentinel-2 band -> approximate center wavelength (nm), used for FAI interpolation
WAVELENGTH_NM = {
    "B02": 490, "B03": 560, "B04": 665, "B05": 705, "B06": 740,
    "B07": 783, "B08": 842, "B8A": 865, "B11": 1610, "B12": 2190,
}

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


def read_band(tile_dir, name):
    with rasterio.open(os.path.join(tile_dir, f"{name}_raw.tif")) as src:
        return src.read(1).astype(np.float32)


def safe_ratio_index(a, b):
    """(a-b)/(a+b) with divide-by-zero guarded."""
    denom = a + b
    out = np.full_like(a, np.nan, dtype=np.float32)
    valid = denom != 0
    out[valid] = (a[valid] - b[valid]) / denom[valid]
    return out


def extract_one(uid):
    tile_dir = os.path.join(DATA_ROOT, uid)
    try:
        scl = read_band(tile_dir, "SCL")
        water_mask = scl == WATER_CLASS
        n_water = int(water_mask.sum())
        if n_water < 50:
            return {"uid": uid, "error": f"too few water pixels ({n_water})"}

        bands = {b: read_band(tile_dir, b) for b in BANDS}
        # Sentinel-2 L2A surface reflectance is scaled by 10000
        refl = {b: bands[b] / 10000.0 for b in BANDS}

        feats = {"uid": uid, "n_water_px": n_water}

        for b in BANDS:
            vals = refl[b][water_mask]
            feats[f"{b}_mean"] = float(np.mean(vals))
            feats[f"{b}_std"] = float(np.std(vals))
            feats[f"{b}_median"] = float(np.median(vals))

        # --- spectral indices (computed pixel-wise, then summarized over water) ---
        ndci = safe_ratio_index(refl["B05"], refl["B04"])  # chlorophyll (red-edge vs red)
        ndvi = safe_ratio_index(refl["B08"], refl["B04"])  # vegetation/algae biomass
        ndwi = safe_ratio_index(refl["B03"], refl["B08"])  # water extent/turbidity
        blue_green = np.divide(refl["B02"], refl["B03"], out=np.full_like(refl["B02"], np.nan), where=refl["B03"] != 0)
        green_red = np.divide(refl["B03"], refl["B04"], out=np.full_like(refl["B03"], np.nan), where=refl["B04"] != 0)

        # Floating Algae Index (Hu 2009): NIR baseline-subtracted vs linear interp of RED/SWIR
        lam_r, lam_nir, lam_swir = WAVELENGTH_NM["B04"], WAVELENGTH_NM["B08"], WAVELENGTH_NM["B11"]
        interp_frac = (lam_nir - lam_r) / (lam_swir - lam_r)
        rf_baseline = refl["B04"] + (refl["B11"] - refl["B04"]) * interp_frac
        fai = refl["B08"] - rf_baseline

        # 3-band phycocyanin proxy (Simis et al. 2005), sensitive to cyanobacteria pigment
        with np.errstate(divide="ignore", invalid="ignore"):
            three_bda = (1.0 / refl["B04"] - 1.0 / refl["B05"]) * refl["B06"]
        three_bda[~np.isfinite(three_bda)] = np.nan

        for name, arr in [
            ("ndci", ndci), ("ndvi", ndvi), ("ndwi", ndwi),
            ("blue_green", blue_green), ("green_red", green_red),
            ("fai", fai), ("three_bda", three_bda),
        ]:
            vals = arr[water_mask]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                feats[f"{name}_mean"] = np.nan
                feats[f"{name}_std"] = np.nan
                feats[f"{name}_median"] = np.nan
            else:
                feats[f"{name}_mean"] = float(np.mean(vals))
                feats[f"{name}_std"] = float(np.std(vals))
                feats[f"{name}_median"] = float(np.median(vals))

        return feats
    except Exception as e:
        return {"uid": uid, "error": f"{type(e).__name__}: {e}"}


def _init_worker(data_root):
    global DATA_ROOT
    DATA_ROOT = data_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="amfitrite-inland-waters-hab-sentinel2/data")
    ap.add_argument("--summary-xlsx", default="amfitrite-inland-waters-hab-sentinel2/dataset_summary.xlsx")
    ap.add_argument("--out", default="outputs/features.csv")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--limit", type=int, default=None, help="only process first N uids (debug)")
    args = ap.parse_args()

    summary = pd.read_excel(args.summary_xlsx)
    uids = summary["uid"].tolist()
    if args.limit:
        uids = uids[: args.limit]

    print(f"Extracting features for {len(uids)} tiles using {args.workers} workers...")
    t0 = time.time()
    results = []
    errors = []
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(args.data_root,)) as pool:
        for i, res in enumerate(pool.imap_unordered(extract_one, uids, chunksize=16)):
            if "error" in res:
                errors.append(res)
            else:
                results.append(res)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(uids)} done ({time.time() - t0:.1f}s elapsed)")

    print(f"Done in {time.time() - t0:.1f}s. OK={len(results)} errors={len(errors)}")

    feat_df = pd.DataFrame(results)
    merged = feat_df.merge(
        summary[["uid", "case", "date", "lat", "lon", "per_clouds", "water_pixels",
                 "indicative_class", "category", "training_priority"]],
        on="uid", how="left",
    )
    merged["date"] = pd.to_datetime(merged["date"])
    merged["month"] = merged["date"].dt.month

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"Saved {merged.shape} to {args.out}")

    if errors:
        err_path = args.out.replace(".csv", "_errors.json")
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"Saved {len(errors)} errors to {err_path}")


if __name__ == "__main__":
    main()
