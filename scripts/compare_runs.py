"""Task 3.5 - pull all runs of the experiment into one table + chart."""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()
EXP = os.getenv("MLFLOW_EXPERIMENT", "23L-0548_A2_Ops_VisionGuard")
df = mlflow.search_runs(experiment_names=[EXP], filter_string="attributes.status = 'FINISHED'")
df = df[df["tags.mlflow.parentRunId"].isna()] if "tags.mlflow.parentRunId" in df else df
cols = {"tags.mlflow.runName": "run", "params.model": "model", "params.lr": "lr",
        "params.img_size": "img", "params.no_aug": "no_aug", "params.finetune": "finetune",
        "metrics.best_epoch": "best_ep", "metrics.decision_threshold": "thr",
        "metrics.test_accuracy": "acc", "metrics.test_precision": "prec", "metrics.test_recall": "rec",
        "metrics.test_f1": "f1", "metrics.test_roc_auc": "auc", "metrics.test_false_positive_rate": "fpr",
        "run_id": "run_id"}
t = df[[c for c in cols if c in df]].rename(columns=cols).sort_values("auc", ascending=False)
Path("reports").mkdir(exist_ok=True)
t.to_csv("reports/run_comparison.csv", index=False)
print(t.drop(columns=["run_id"]).to_markdown(index=False, floatfmt=".3f"))

fig, ax = plt.subplots(figsize=(10, 4.5))
x = range(len(t))
w = 0.27
ax.bar([i - w for i in x], t["auc"], w, label="ROC-AUC")
ax.bar(list(x), t["f1"], w, label="F1")
ax.bar([i + w for i in x], 1 - t["fpr"], w, label="1 - FPR (real kept real)")
ax.set_xticks(list(x), t["run"], rotation=30, ha="right")
ax.set_ylim(0, 1.05)
ax.legend()
ax.set_title("VisionGuard experiment comparison (test set)")
fig.tight_layout()
fig.savefig("reports/run_comparison.png", dpi=120)
print("\nsaved reports/run_comparison.csv and .png")
