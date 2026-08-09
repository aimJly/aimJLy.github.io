"""Additional summary plots built from outputs/metrics.json (model comparison, fairness bars)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")

with open(os.path.join(OUT_DIR, "metrics.json")) as f:
    metrics = json.load(f)

# --- model comparison ---
rows = []
for name, m in metrics["held_out_test_metrics"].items():
    rows.append({"model": name, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
comp = pd.DataFrame(rows).set_index("model").reindex(
    ["majority_baseline", "logistic_regression", "gradient_boosting", "random_forest"]
)
fig, ax = plt.subplots(figsize=(6, 4))
comp.plot(kind="bar", ax=ax, color=["#9db4c0", "#2b7a78"])
ax.set_ylabel("score")
ax.set_title("Held-out test performance by model")
ax.set_ylim(0, 1)
ax.legend(["Accuracy", "Macro F1"])
plt.xticks(rotation=25, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "model_comparison.png"), dpi=130)
plt.close(fig)

# --- fairness stratified accuracy ---
fairness = metrics["fairness_stratified_accuracy"]
groups_to_plot = ["by_cloud_cover", "by_water_body_size", "by_season", "accuracy_by_true_class"]
fig, axes = plt.subplots(1, len(groups_to_plot), figsize=(16, 3.5))
titles = {
    "by_cloud_cover": "Accuracy by cloud cover",
    "by_water_body_size": "Accuracy by water body size",
    "by_season": "Accuracy by season",
    "accuracy_by_true_class": "Accuracy by true class",
}
for ax, g in zip(axes, groups_to_plot):
    d = fairness[g]
    ax.bar(d.keys(), d.values(), color="#3b6ea5")
    ax.set_ylim(0, 1)
    ax.set_title(titles[g], fontsize=10)
    ax.tick_params(axis="x", labelrotation=30)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fairness_stratified.png"), dpi=130)
plt.close(fig)

print("Saved model_comparison.png and fairness_stratified.png")
