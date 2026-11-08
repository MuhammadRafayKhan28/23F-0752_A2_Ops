"""Task 2.4 / 2.5 - threshold selection, metrics and plots."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def choose_threshold(y_true, prob, beta: float = 0.5):
    """Pick the validation threshold that maximises F-beta with beta=0.5.

    Positive class = AI-generated. beta<1 weights precision more than recall, i.e. it punishes
    false positives (a REAL photo called FAKE) harder - the error we consider worse for VisionGuard.
    """
    grid = np.linspace(0.05, 0.95, 91)
    scores = [fbeta_score(y_true, (prob >= t).astype(int), beta=beta, zero_division=0) for t in grid]
    best = int(np.argmax(scores))
    return round(float(grid[best]), 2), float(scores[best])


def compute_metrics(y_true, prob, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob) if len(set(y_true)) > 1 else float("nan"),
        "false_positive_rate": fp / max(fp + tn, 1),   # real photos flagged as AI
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def plot_confusion(y_true, prob, threshold, path, title="Confusion matrix (test)"):
    cm = confusion_matrix(y_true, (prob >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1], ["Real", "AI"])
    ax.set_yticks([0, 1], ["Real", "AI"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title}\nthreshold={threshold:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc(y_true, prob, path, title="ROC curve (test)"):
    fpr, tpr, _ = roc_curve(y_true, prob)
    auc = roc_auc_score(y_true, prob)
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate (real called AI)")
    ax.set_ylabel("True positive rate (AI caught)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_history(history: dict, path, best_epoch: int, stage_boundary: int | None = None, title=""):
    """history: dict of lists (combined over both training stages). best_epoch is 1-based."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, history[key], label=f"train {key}")
        ax.plot(epochs, history[f"val_{key}"], label=f"val {key}")
        ax.axvline(best_epoch, color="red", ls="--", label=f"best epoch = {best_epoch}")
        if stage_boundary:
            ax.axvline(stage_boundary + 0.5, color="grey", ls=":", label="start fine-tuning")
        ax.set_xlabel("epoch")
        ax.set_title(key)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
