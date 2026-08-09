# Credits 
Group Members: Dani Jiang, Abigail Huang, Kelly Garcia, Duy Nghiem, Raquel Quintanilla


# HAB Classifier — Sentinel-2 Harmful Algal Bloom Severity Classification

Classifies inland-water Harmful Algal Bloom (HAB) severity (`Low` / `Moderate` / `High`) from
Sentinel-2 multispectral satellite imagery, using the
[amfitrite-inland-waters-hab-sentinel2](https://huggingface.co/datasets/kostaspic/amfitrite-inland-waters-hab-sentinel2)
dataset (4,698 tiles, 955 locations).

**Start here:** [`reports/REPORT.md`](reports/REPORT.md) for the full write-up (metrics, findings,
fairness analysis, deployment recommendation), or
[`notebooks/model_evaluation.ipynb`](notebooks/model_evaluation.ipynb) for the same content with
inline code and real run outputs.

## Results at a glance

Best model: **Random Forest** — macro F1 = 0.741, accuracy = 0.750 on a held-out set of 843 tiles
from 191 locations never seen in training (vs. 0.192 macro F1 for a majority-class baseline).

## Repo layout

```
src/extract_features.py   # reads raw GeoTIFF bands, water-masks with SCL, computes spectral
                           # features per tile -> outputs/features.csv
src/train_evaluate.py     # grouped train/test split, trains 4 classifiers, evaluates, saves
                           # outputs/metrics.json + outputs/figures/*.png
src/extra_plots.py        # a couple of summary charts built from metrics.json
src/build_notebook.py     # packages the above into notebooks/model_evaluation.ipynb
outputs/                  # features.csv, metrics.json, figures/
notebooks/                # model_evaluation.ipynb
reports/REPORT.md         # written report
amfitrite-inland-waters-hab-sentinel2/   # the cloned dataset (git+lfs)
```

## Reproducing

```bash
pip install -r requirements.txt
python src/extract_features.py --workers 8      # ~4 min on 8 cores; writes outputs/features.csv
python src/train_evaluate.py                    # writes outputs/metrics.json + figures
python src/extra_plots.py                       # a couple of extra summary charts
python src/build_notebook.py                    # rebuilds notebooks/model_evaluation.ipynb
```

## Key design decisions

- **No label leakage:** features are computed independently from raw pixel bands; none of the
  dataset's CyFi-derived summary columns (which were used to build the label itself) are used as
  inputs. See `reports/REPORT.md` §1.
- **Grouped train/test split by location (`case`)**, not by row, since the same lake appears at
  multiple dates and rows from one location are spatially correlated.
- **Macro-F1** is the primary metric because severity classes are imbalanced and all three matter
  for an early-warning use case.


# HAB Severity Classification — Model Evaluation Report

**Project:** Detecting and classifying Harmful Algal Bloom (HAB) severity in inland waters from Sentinel-2 multispectral imagery.
**Dataset:** [amfitrite-inland-waters-hab-sentinel2](https://huggingface.co/datasets/kostaspic/amfitrite-inland-waters-hab-sentinel2) — 4,698 Sentinel-2 L2A tiles over North American lakes/ponds, each labeled `Low` / `Moderate` / `High` cyanobacterial bloom severity (built from the CAML in-situ dataset + CyFi model predictions).
**Code:** [`src/extract_features.py`](../src/extract_features.py), [`src/train_evaluate.py`](../src/train_evaluate.py) · **Notebook:** [`notebooks/model_evaluation.ipynb`](../notebooks/model_evaluation.ipynb)

---

## 1. Approach

Rather than training a CNN directly on raw pixel grids (the dataset's stated primary use case, but the heaviest option), this first checkpoint uses a **feature-based classical ML pipeline**: for every tile we mask out non-water pixels using the Scene Classification Layer, compute reflectance statistics (mean/std/median) for 10 raw Sentinel-2 bands, and derive spectral indices with known links to water quality and cyanobacteria — NDCI (chlorophyll), a 3-band phycocyanin proxy (Simis et al. 2005, targets the pigment specific to cyanobacteria), FAI (floating algae), NDVI, NDWI, and blue:green / green:red ratios. This gives 55 engineered features per tile.

This is a deliberate, justified choice for a first model: it's fast to train and evaluate (minutes, not hours, on a CPU-only machine), highly interpretable (feature importances map directly onto known bio-optical signals, which is a strong sanity check — see §4), and gives a real performance floor to beat with a CNN later. It also mirrors how the CyFi tool referenced in this dataset itself works (pixel/point-based spectral classification, not deep CNNs).

**Label-leakage guard:** the dataset's summary spreadsheet includes several columns (`high_pred`/`mod_pred`/`low_pred`, `roughness_median`, `outlier_fraction`, `HAB_status`, `abun`, `abun_class`) that were themselves used to *construct* the `indicative_class` label. None of these were used as model inputs — every feature is either computed independently from the raw pixel bands (by us) or is basic, pre-label metadata (cloud %, acquisition month).

**Preventing location leakage:** 4,698 tiles cover only 955 unique locations (`case`), averaging ~5 dates per lake. A row-level train/test split would put the same lake's shoreline/bathymetry in both train and test, inflating scores. We instead used a **grouped split by `case`** (`GroupShuffleSplit`), so no lake in the test set was ever seen in training — verified with an explicit assertion in code. Cross-validation during training also used `GroupKFold` for the same reason.

---

## 2. Models trained and evaluated

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| Majority-class baseline | 0.406 | 0.333 | 0.192 | 0.234 |
| Logistic Regression | 0.675 | 0.680 | 0.679 | 0.681 |
| Gradient Boosting | 0.745 | 0.734 | 0.732 | 0.740 |
| **Random Forest** | **0.750** | **0.739** | **0.741** | **0.748** |

*(Held-out test set: 843 tiles from 191 locations never seen during training. 5-fold grouped cross-validation on the training set gave consistent numbers — e.g. Random Forest macro-F1 0.732 in-CV vs. 0.741 on the held-out test — indicating the result is stable, not a lucky split.)*

**Why macro-F1 as the primary metric, not accuracy:** classes are imbalanced (High 1958 / Low 1401 / Moderate 1339), and all three severity levels matter for an early-warning use case — missing a `High` bloom is much costlier than the class's frequency would suggest, and accuracy alone would let a model that ignores `Moderate` look artificially good. The majority-baseline row makes this concrete: guessing `High` every time yields 40.6% accuracy but a macro-F1 of only 0.19, because it scores zero on the other two classes.

### Per-class performance — Random Forest (best model)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low | 0.833 | 0.789 | 0.810 | 246 |
| Moderate | 0.607 | 0.580 | 0.593 | 255 |
| High | 0.792 | 0.848 | 0.819 | 342 |

Confusion matrix (rows = true, columns = predicted):

```
              Low  Moderate  High
Low           194        49     3
Moderate       34       148    73
High            5        47   290
```

---

## 3. Strengths and weaknesses

**Strengths**
- Clearly beats the majority-class baseline (macro-F1 0.74 vs. 0.19) and a linear model (Random Forest vs. Logistic Regression: +0.06 macro-F1), showing the spectral features carry real, non-linear signal.
- `Low` and `High` are the two classes that matter most for an early-warning system, and they're the strongest: recall of 0.79 (Low) and 0.85 (High), with very few High blooms predicted as Low (5 of 342) or vice versa (3 of 246) — the model essentially never makes the worst kind of mistake (confusing the two extremes).
- Feature importances are physically sensible: the top predictors are `blue_green_mean/median` (a classic chlorophyll/pigment ratio — high pigment concentration absorbs blue light and lowers this ratio), `NDCI` (red-edge chlorophyll index), and the phycocyanin-targeted `three_bda` index — exactly the bands remote-sensing literature would point to for cyanobacteria detection. This is a meaningful sanity check that the model is learning bloom-relevant optics, not spurious correlations.

**Weaknesses**
- `Moderate` is the clear weak point: F1 of 0.59, versus ~0.81 for the other two classes. 73 of 255 true-Moderate tiles were predicted `High`, and 34 were predicted `Low` — it's the "boundary" class sitting between two others on a continuous severity gradient, and confusable with both. This is consistent across every model tried, so it looks like a genuine signal ceiling given tile-level, single-date spectral summaries, not a model-selection artifact.
- Overall accuracy (75%) still leaves 1 in 4 tiles misclassified, which is a helpful screening tool but not yet reliable enough to fully automate a response decision — see §5.
- Random Forest and Gradient Boosting perform almost identically (macro-F1 0.741 vs. 0.732); Logistic Regression trails by 6 points, suggesting the class boundaries in feature space are meaningfully non-linear (interactions between bands/indices), which tree ensembles capture better than a linear separator.

---

## 4. Fairness, bias, and limitations

We stratified the Random Forest's held-out accuracy by factors that could plausibly bias real-world performance:

| Stratum | Accuracy |
|---|---|
| **Cloud cover** — low (<0.5%) | 0.820 |
| **Cloud cover** — medium (0.5–2%) | 0.737 |
| **Cloud cover** — high (>2%) | 0.638 |
| **Water body size** — small (bottom tercile of water pixels) | 0.662 |
| **Water body size** — medium | 0.772 |
| **Water body size** — large | 0.815 |
| **Season** — winter | 0.667 |
| **Season** — spring | 0.689 |
| **Season** — fall | 0.743 |
| **Season** — summer | 0.817 |
| **Label source** — tiles with real in-situ measurement | 0.801 |
| **Label source** — tiles augmented from CyFi-only labels | 0.713 |

**Reading these results:**
- **Cloud cover and small water bodies both hurt accuracy** — both reduce the number of clean water pixels available to compute stable statistics from, so noisier per-tile aggregates are the likely cause. This means the model will be least reliable on small ponds and hazy imagery — exactly the conditions where a monitoring tool is often needed most (small water bodies are common and rarely otherwise monitored).
- **Summer performs best, winter/spring worst.** This tracks real bloom biology (cyanobacteria blooms are warm-season phenomena), but it also means winter/spring predictions should be trusted less — the model has seen fewer, less-typical examples in those seasons.
- **Real in-situ-labeled tiles score ~9 points higher than CyFi-only-labeled tiles.** This is expected: CyFi-only labels are themselves model predictions, not ground truth, so our model is partly learning to imitate another model's errors on that subset. It also means our reported accuracy is a slight overestimate of true-ground-truth accuracy, since 53% of tiles (2,477 of 4,698) fall in this "without_measurements" (CyFi-labeled) category.

**Dataset-level biases inherited from the source data (documented in the dataset card, not fixable by modeling choices):**
- **Geographic bias:** all locations come from the CAML study's North American sampling sites. The model has no guarantee of generalizing to lakes in other regions with different water chemistry, turbidity baselines, or algal species composition.
- **Taxonomic specificity:** labels target cyanobacterial blooms specifically. HABs caused by other organisms (e.g., certain marine algae) have different optical signatures and would likely not be detected reliably by this model.
- **Tile-level label vs. patchy blooms:** a tile is labeled `High` if *any* part of the water body has a high-severity bloom, even if most of it is clear. Our model is trained on whole-tile averaged statistics but asked to detect what can be a spatially localized event — this "averaging away" of localized signal is a plausible contributor to the `Moderate` class's weaker performance and caps achievable performance on `High` recall for small blooms.
- **Only 5 years of data (2017–2022), all cloud-filtered (<7.5%)** — the model has not seen extreme-weather or heavy-haze conditions that occur in practice.

---

## 5. Recommended model for deployment

**Random Forest** is the model we'd move forward with: best macro-F1 and accuracy of the four, similar training/inference cost to Gradient Boosting, more robust to unscaled/skewed features than Logistic Regression, and it doubles as an interpretability tool via feature importances (useful for building trust with domain scientists and for further feature engineering).

**Recommended deployment framing, given current performance:** as a **triage / early-warning screening tool**, not a fully automated severity determination. Concretely:
- High confidence in flagging the `Low` vs. `High` boundary (the two extremes are rarely confused with each other), which is the highest-value distinction for a "should someone go check this lake" alert.
- Predictions on small water bodies, high-cloud-cover imagery, and winter/spring dates should be flagged as lower-confidence and prioritized for human/in-situ follow-up rather than acted on directly.
- `Moderate` predictions should be treated as "could be Low or could be High" and always escalated for follow-up rather than trusted at face value.

**Concrete next steps** (natural continuation of this checkpoint):
1. Train a CNN directly on the 256×256 water-masked image crops (using the dataset's provided `crop_row`/`crop_col` and `new_indicative_class` to avoid the cropping/label-shift pitfalls the dataset card explicitly warns about) and compare against this tabular baseline — this is the dataset's intended primary use case and may better capture the spatial patchiness that hurts the `Moderate` class here.
2. Investigate the `Moderate` class specifically: is confusion concentrated in tiles with mixed severity within the same lake (multiple CyFi grid points disagreeing), where the single tile-level label is inherently more ambiguous?
3. Re-run the fairness stratification restricted only to `with_measurements` (true ground-truth) tiles to get an accuracy estimate less influenced by inherited CyFi label noise.

