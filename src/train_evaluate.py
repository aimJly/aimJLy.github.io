"""
Train and evaluate baseline classifiers for HAB severity classification
(Low / Moderate / High) from water-masked Sentinel-2 spectral features.

Key design choices (see reports/REPORT.md for full rationale):
  * Features are derived purely from raw pixel reflectance (computed in
    extract_features.py) plus acquisition month. No column that was itself
    used to construct the `indicative_class` label (CyFi per-pixel
    predictions, roughness, outlier fraction, etc.) is used as input.
  * Train/val/test are split by geographic `case`, not by row, since the
    same lake/pond appears at many dates and rows from the same case are
    spatially correlated (near-duplicate shoreline/bathymetry). A random
    row-level split would leak location identity into the test set.
  * Metrics: macro-F1 (primary, since classes are imbalanced and all three
    severity levels matter), per-class precision/recall/F1, confusion
    matrix, and accuracy vs. a majority-class baseline.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

CLASS_ORDER = ["Low", "Moderate", "High"]

LEAKY_COLS = {
    "uid", "case", "date", "indicative_class", "category", "training_priority",
    "lat", "lon",  # exact coordinates would let the model memorize specific ponds
}


def load_features(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["indicative_class"])
    feature_cols = [c for c in df.columns if c not in LEAKY_COLS]
    return df, feature_cols


def make_models():
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }


def evaluate_predictions(y_true, y_pred, labels=CLASS_ORDER):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=labels),
        "report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def plot_confusion(cm, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_feature_importance(model, feature_cols, out_path, top_n=20):
    if not hasattr(model, "feature_importances_"):
        return None
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top = importances.head(top_n)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.barplot(x=top.values, y=top.index, ax=ax, color="#2b7a78")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest — top feature importances")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/features.csv")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    df, feature_cols = load_features(args.features)
    X = df[feature_cols].values
    y = df["indicative_class"].values
    groups = df["case"].values

    print(f"Loaded {df.shape[0]} tiles, {len(feature_cols)} features, {df['case'].nunique()} unique locations")
    print("Class distribution:\n", df["indicative_class"].value_counts())

    # ---- grouped held-out test split (by geographic case) ----
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    print(f"Train: {len(train_idx)} tiles ({df.iloc[train_idx]['case'].nunique()} locations) | "
          f"Test: {len(test_idx)} tiles ({df.iloc[test_idx]['case'].nunique()} locations)")

    overlap = set(df.iloc[train_idx]["case"]) & set(df.iloc[test_idx]["case"])
    assert not overlap, f"Location leakage between train/test: {overlap}"

    models = make_models()
    results = {}
    cv_results = {}
    gkf = GroupKFold(n_splits=args.cv_folds)

    for name, model in models.items():
        print(f"\n=== {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)
        results[name] = metrics
        print(f"  held-out test: acc={metrics['accuracy']:.3f}  "
              f"macro_f1={metrics['macro_f1']:.3f}  bal_acc={metrics['balanced_accuracy']:.3f}")

        plot_confusion(
            np.array(metrics["confusion_matrix"]), CLASS_ORDER,
            f"{name} — held-out test confusion matrix",
            os.path.join(fig_dir, f"confusion_{name}.png"),
        )

        # grouped cross-validation on the training portion for a more robust estimate
        cv_pred = cross_val_predict(model, X_train, y_train, groups=groups_train, cv=gkf, n_jobs=-1)
        cv_metrics = evaluate_predictions(y_train, cv_pred)
        cv_results[name] = cv_metrics
        print(f"  {args.cv_folds}-fold grouped CV (train): macro_f1={cv_metrics['macro_f1']:.3f} "
              f"(+/- across folds not shown, see report)")

    # feature importance from the random forest (best interpretable model)
    rf = models["random_forest"]
    top_importances = plot_feature_importance(rf, feature_cols, os.path.join(fig_dir, "feature_importance_rf.png"))

    # class balance plot
    fig, ax = plt.subplots(figsize=(4, 3.5))
    df["indicative_class"].value_counts().reindex(CLASS_ORDER).plot(kind="bar", ax=ax, color="#3b6ea5")
    ax.set_ylabel("tile count")
    ax.set_title("Class distribution (all tiles)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "class_distribution.png"), dpi=130)
    plt.close(fig)

    # ---- fairness / stratified error analysis on the best model (random_forest) ----
    best_name = max(results, key=lambda k: results[k]["macro_f1"] if k != "majority_baseline" else -1)
    best_model = models[best_name]
    y_pred_best = best_model.predict(X_test)
    test_df = df.iloc[test_idx].copy()
    test_df["y_true"] = y_test
    test_df["y_pred"] = y_pred_best
    test_df["correct"] = test_df["y_true"] == test_df["y_pred"]

    fairness = {}
    # by cloud cover bucket
    test_df["cloud_bucket"] = pd.cut(test_df["per_clouds"], [-0.01, 0.5, 2, 100], labels=["low(<0.5%)", "med(0.5-2%)", "high(>2%)"])
    fairness["by_cloud_cover"] = test_df.groupby("cloud_bucket", observed=True)["correct"].mean().to_dict()
    # by water pixel count (tile size / lake size proxy)
    test_df["water_size_bucket"] = pd.qcut(test_df["n_water_px"], 3, labels=["small", "medium", "large"])
    fairness["by_water_body_size"] = test_df.groupby("water_size_bucket", observed=True)["correct"].mean().to_dict()
    # by season
    test_df["season"] = test_df["month"].map({12: "winter", 1: "winter", 2: "winter",
                                                3: "spring", 4: "spring", 5: "spring",
                                                6: "summer", 7: "summer", 8: "summer",
                                                9: "fall", 10: "fall", 11: "fall"})
    fairness["by_season"] = test_df.groupby("season", observed=True)["correct"].mean().to_dict()
    # by data category (real in-situ measurement vs. augmented same-site sample)
    fairness["by_category"] = test_df.groupby("category", observed=True)["correct"].mean().to_dict()
    # accuracy per true class
    fairness["accuracy_by_true_class"] = test_df.groupby("y_true", observed=True)["correct"].mean().to_dict()

    print("\n=== Fairness / stratified accuracy (best model =", best_name, ") ===")
    print(json.dumps(fairness, indent=2, default=str))

    # ---- save everything ----
    summary = {
        "n_tiles": int(df.shape[0]),
        "n_locations": int(df["case"].nunique()),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "class_distribution": df["indicative_class"].value_counts().to_dict(),
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "held_out_test_metrics": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
        "grouped_cv_train_metrics": cv_results,
        "best_model": best_name,
        "fairness_stratified_accuracy": fairness,
        "top_feature_importances_rf": top_importances.to_dict() if top_importances is not None else None,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved metrics.json and figures to {args.out_dir}/")


if __name__ == "__main__":
    main()
