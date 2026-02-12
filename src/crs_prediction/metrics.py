"""Metrics computation for binary classification."""

import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, roc_auc_score, roc_curve

try:
    import wandb
except ImportError:
    wandb = None


def log_roc_curve(
    all_logits: torch.Tensor,
    all_gts: torch.Tensor,
    prefix: str = "",
) -> float:
    """Calculate ROC curve, AUC, and log to wandb.

    Args:
        all_logits: Model logits (raw outputs before sigmoid).
        all_gts: Ground truth labels (0 or 1).
        prefix: Prefix for metric names in wandb.

    Returns:
        The ROC AUC score.
    """
    all_logits = all_logits.detach().cpu()
    all_gts = all_gts.detach().cpu()

    probabilities = torch.sigmoid(all_logits)
    probs_np = probabilities.flatten().numpy()
    gts_np = all_gts.flatten().numpy()

    fpr, tpr, thresholds = roc_curve(gts_np, probs_np)
    roc_auc = auc(fpr, tpr)

    if wandb is not None and wandb.run is not None:
        table = wandb.Table(
            columns=["False Positive Rate", "True Positive Rate", "Threshold"]
        )
        for i in range(len(fpr)):
            table.add_data(
                float(fpr[i]),
                float(tpr[i]),
                float(thresholds[i]) if i < len(thresholds) else float("nan"),
            )

        wandb.log(
            {
                f"{prefix}roc_auc": roc_auc,
                f"{prefix}roc_curve": wandb.plot.line(
                    table,
                    "False Positive Rate",
                    "True Positive Rate",
                    title=f"ROC Curve (AUC = {roc_auc:.4f})",
                ),
            }
        )

    return roc_auc


def calculate_metrics(
    logits: torch.Tensor,
    ground_truth: torch.Tensor,
    threshold: float = 0.5,
    split: str | None = None,
) -> dict:
    """Calculate a suite of metrics for binary classification.

    Includes accuracy, precision, recall, F1 score, ROC AUC, PR AUC,
    confusion matrix components, and Cohen's Kappa.

    Args:
        logits: Tensor of shape (B, 1) containing model logits.
        ground_truth: Tensor of shape (B, 1) containing ground truth (0 or 1).
        threshold: Threshold for converting probabilities to binary predictions.
        split: A string identifier for the data split (e.g., 'Training', 'Validation', 'Test').

    Returns:
        Dictionary of computed metrics.
    """
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).int()

    predictions = predictions.view(-1).cpu().numpy()
    ground_truth = ground_truth.view(-1).cpu().numpy()
    probabilities = probabilities.view(-1).cpu().numpy()

    TP = ((predictions == 1) & (ground_truth == 1)).sum()
    TN = ((predictions == 0) & (ground_truth == 0)).sum()
    FP = ((predictions == 1) & (ground_truth == 0)).sum()
    FN = ((predictions == 0) & (ground_truth == 1)).sum()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    cohen_kappa = (
        2 * (TP * TN - FP * FN) / ((TP + FP) * (FP + TN) + (TP + FN) * (FN + TN))
        if ((TP + FP) * (FP + TN) + (TP + FN) * (FN + TN)) > 0
        else 0.0
    )

    roc_auc = (
        roc_auc_score(ground_truth, probabilities)
        if len(set(ground_truth)) > 1
        else None
    )
    pr_auc = (
        average_precision_score(ground_truth, probabilities)
        if len(set(ground_truth)) > 1
        else None
    )

    prefix = f"{split}/" if split is not None else ""
    return {
        f"{prefix}accuracy": accuracy,
        f"{prefix}precision": precision,
        f"{prefix}recall": recall,
        f"{prefix}f1_score": f1_score,
        f"{prefix}roc_auc": roc_auc,
        f"{prefix}pr_auc": pr_auc,
        f"{prefix}cohen_kappa": cohen_kappa,
        f"{prefix}confusion_matrix/TP": TP,
        f"{prefix}confusion_matrix/TN": TN,
        f"{prefix}confusion_matrix/FP": FP,
        f"{prefix}confusion_matrix/FN": FN,
    }
