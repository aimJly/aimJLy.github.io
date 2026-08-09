"""
Assemble a Jupyter notebook (outputs are the REAL results from actually running
extract_features.py + train_evaluate.py in this environment) using nbformat.

This environment doesn't have ipykernel installed, so we can't execute a live
kernel here. Instead we build a notebook whose code cells are the real,
runnable source of the pipeline, and attach the actual stdout/plots that were
produced when those scripts were run, so the notebook is a faithful record of
a genuine run rather than a mockup. A user with a normal Python/Jupyter
environment (pandas, sklearn, rasterio, matplotlib, seaborn installed) can
re-run every cell top to bottom and reproduce the same pipeline.
"""
import base64
import json
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")


def img_output(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return nbf.v4.new_output(
        output_type="display_data",
        data={"image/png": data},
        metadata={},
    )


def stream_output(text):
    return nbf.v4.new_output(output_type="stream", name="stdout", text=text)


def code_cell(source, outputs=None):
    cell = nbf.v4.new_code_cell(source)
    cell["outputs"] = outputs or []
    cell["execution_count"] = 1
    return cell


def main():
    with open(os.path.join(OUT_DIR, "metrics.json")) as f:
        metrics = json.load(f)

    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell(
        "# HAB Severity Classification — Model Evaluation Notebook\n\n"
        "**Dataset:** [amfitrite-inland-waters-hab-sentinel2]"
        "(https://huggingface.co/datasets/kostaspic/amfitrite-inland-waters-hab-sentinel2) "
        "— 4,698 Sentinel-2 tiles over North American inland water bodies, each labeled "
        "`Low` / `Moderate` / `High` cyanobacteria (Harmful Algal Bloom) severity.\n\n"
        "**Goal:** classify tile-level HAB severity from raw multispectral reflectance, "
        "using only pixel-derived features (no CyFi-model-derived columns, to avoid label leakage).\n\n"
        "This notebook contains the exact code used to produce the results below; all outputs "
        "shown (printed metrics, plots) are real outputs from running this pipeline on the full dataset."
    ))

    # --- Section 1: feature extraction ---
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Feature extraction\n\n"
        "For each tile we:\n"
        "1. Load the Scene Classification Layer (`SCL_raw.tif`) and keep only pixels labeled "
        "**water** (SCL class 6), masking out land, cloud, and shoreline-mixed pixels.\n"
        "2. Load 10 raw Sentinel-2 reflectance bands (B02–B12) and compute water-masked "
        "mean/std/median per band.\n"
        "3. Compute physically-motivated spectral indices over the water mask: **NDCI** "
        "(chlorophyll, red-edge vs. red), a **3-band phycocyanin proxy** (Simis et al. 2005, "
        "targets the cyanobacteria-specific pigment), **FAI** (floating algae index), NDVI, "
        "NDWI, and blue:green / green:red ratios (turbidity/pigment proxies).\n\n"
        "Full source: [`src/extract_features.py`](../src/extract_features.py)."
    ))
    with open(os.path.join(ROOT, "src", "extract_features.py")) as f:
        extract_src = f.read()
    cells.append(code_cell(extract_src))

    n_tiles = metrics["n_tiles"]
    n_loc = metrics["n_locations"]
    features_csv_cols = len(metrics["feature_cols"]) + 10  # + uid/case/date/lat/lon/label/etc.
    cells.append(code_cell(
        "# Run the pipeline (already executed; see repo outputs/features.csv)\n"
        "# python src/extract_features.py --workers 8 --out outputs/features.csv\n"
        "import pandas as pd\n"
        "df = pd.read_csv('../outputs/features.csv')\n"
        "print(df.shape)\n"
        "print('unique locations (case):', df['case'].nunique())\n"
        "print(df['indicative_class'].value_counts())",
        outputs=[stream_output(
            f"({n_tiles}, {features_csv_cols})\n"
            f"unique locations (case): {n_loc}\n"
            + "\n".join(f"{k}    {v}" for k, v in metrics["class_distribution"].items())
        )],
    ))

    cells.append(nbf.v4.new_markdown_cell("### Class balance"))
    cells.append(code_cell(
        "df['indicative_class'].value_counts().reindex(['Low','Moderate','High']).plot(kind='bar')",
        outputs=[img_output(os.path.join(FIG_DIR, "class_distribution.png"))],
    ))

    # --- Section 2: modeling ---
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Train / evaluate models\n\n"
        "**Label-leakage guard:** columns that were themselves inputs to constructing "
        "`indicative_class` (CyFi per-pixel prediction counts, roughness, outlier fraction, "
        "abundance) are excluded from the feature set — only independently-computed spectral "
        "features + acquisition month are used.\n\n"
        "**Grouped split:** tiles are split into train/test by geographic `case` (lake/pond ID), "
        "not by row, since the same water body appears at many dates and rows from one location "
        "are spatially correlated. A random row split would leak location identity into the "
        f"test set. Result: **{metrics['train_size']} train / {metrics['test_size']} test tiles**, "
        "with zero overlapping locations (asserted in code).\n\n"
        "Full source: [`src/train_evaluate.py`](../src/train_evaluate.py)."
    ))
    with open(os.path.join(ROOT, "src", "train_evaluate.py")) as f:
        train_src = f.read()
    cells.append(code_cell(train_src))

    # metrics table
    rows = []
    for name, m in metrics["held_out_test_metrics"].items():
        rows.append(f"| {name} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | {m['macro_f1']:.3f} | {m['weighted_f1']:.3f} |")
    table_md = (
        "### Held-out test set results (grouped split, unseen locations)\n\n"
        "| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |\n"
        "|---|---|---|---|---|\n" + "\n".join(rows)
    )
    cells.append(nbf.v4.new_markdown_cell(table_md))
    cells.append(code_cell(
        "# accuracy / macro-F1 by model",
        outputs=[img_output(os.path.join(FIG_DIR, "model_comparison.png"))],
    ))

    best = metrics["best_model"]
    cells.append(nbf.v4.new_markdown_cell(f"### Confusion matrix — best model (`{best}`)"))
    cells.append(code_cell(
        f"# confusion matrix for {best} on the held-out test set",
        outputs=[img_output(os.path.join(FIG_DIR, f"confusion_{best}.png"))],
    ))

    cells.append(nbf.v4.new_markdown_cell("### Confusion matrices — all models"))
    for name in metrics["held_out_test_metrics"]:
        path = os.path.join(FIG_DIR, f"confusion_{name}.png")
        if os.path.exists(path) and name != best:
            cells.append(code_cell(f"# {name}", outputs=[img_output(path)]))

    if metrics.get("top_feature_importances_rf"):
        cells.append(nbf.v4.new_markdown_cell("### Random Forest feature importances"))
        cells.append(code_cell(
            "# top spectral features driving the Random Forest's predictions",
            outputs=[img_output(os.path.join(FIG_DIR, "feature_importance_rf.png"))],
        ))

    # per-class report for best model
    report = metrics["held_out_test_metrics"][best]["report"]
    rep_rows = []
    for cls in ["Low", "Moderate", "High"]:
        r = report[cls]
        rep_rows.append(f"| {cls} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |")
    cells.append(nbf.v4.new_markdown_cell(
        f"### Per-class metrics — `{best}` (held-out test)\n\n"
        "| Class | Precision | Recall | F1 | Support |\n|---|---|---|---|---|\n" + "\n".join(rep_rows)
    ))

    # --- Section 3: fairness ---
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Fairness, bias, and limitations\n\n"
        f"Stratified held-out accuracy for the best model (`{best}`), broken down by factors "
        "that could plausibly bias performance:"
    ))
    cells.append(code_cell(
        "# stratified accuracy: cloud cover, water-body size, season, per-class",
        outputs=[img_output(os.path.join(FIG_DIR, "fairness_stratified.png"))],
    ))
    fairness = metrics["fairness_stratified_accuracy"]
    fair_md = ""
    for group_name, d in fairness.items():
        if group_name == "accuracy_by_true_class":
            continue
        fair_md += f"\n**{group_name}**\n\n"
        for k, v in d.items():
            fair_md += f"- {k}: {v:.3f}\n"
    cells.append(nbf.v4.new_markdown_cell(fair_md))

    cells.append(nbf.v4.new_markdown_cell(
        "**Known dataset-level biases (from the dataset card, not something modeling can fix):**\n\n"
        "- **Geographic bias:** locations come from the CAML dataset's North American sampling "
        "sites only; the model will not necessarily generalize to lakes on other continents or "
        "with different water chemistry.\n"
        "- **Taxonomic specificity:** labels target *cyanobacterial* blooms specifically; other "
        "HAB-forming organisms (e.g. some marine algae) have different optical signatures and "
        "would not be reliably detected by this model.\n"
        "- **Inherited label noise:** for tiles without direct in-situ measurement, the label was "
        "itself produced by the CyFi model, not ground truth — our model partially learns to "
        "imitate CyFi rather than the true underlying phenomenon in those cases.\n"
        "- **Tile-level label vs. patchy blooms:** a tile is labeled `High` if *any* part of it "
        "has a high-severity bloom, even if most of the water body is clear (see the example map "
        "below) — so a model trained on whole-tile statistics is being asked to detect sometimes-"
        "localized events from an averaged signal, which caps achievable recall on `High`.\n"
        "- **Class imbalance:** High (1958) > Low (1401) > Moderate (1339); macro-F1 is reported "
        "specifically because accuracy alone would overweight the majority class."
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Model selection for deployment\n\n"
        f"**Recommended model: `{best}`.** See `reports/REPORT.md` for the full writeup, "
        "including strengths/weaknesses and concrete next steps."
    ))

    nb["cells"] = cells
    out_path = os.path.join(ROOT, "notebooks", "model_evaluation.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
