"""Model evaluation for CRS prediction."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from crs_prediction import metrics as metrics_module
from crs_prediction.engine import log_attention_weights

try:
    import wandb

    _wandb_available = True
except ImportError:
    wandb = None
    _wandb_available = False

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates a trained model on a given dataset and computes metrics.

    Handles loading from checkpoint, device management, and optional wandb logging.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        test_loader: Optional[DataLoader] = None,
        checkpoint_path: Optional[str] = None,
        config: Optional[Dict] = None,
        enable_wandb: bool = True,
    ):
        self.device = device
        self.config = config or {}
        self.test_loader = test_loader
        self.enable_wandb = enable_wandb and _wandb_available

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint from {checkpoint_path}")

        self.model = model.to(device)
        self.model.eval()

    def predict(
        self,
        loader: DataLoader,
        optimal_threshold: float = 0.5,
    ) -> Dict:
        """Run model inference and compute evaluation metrics.

        Args:
            loader: DataLoader for evaluation (e.g., test set).
            optimal_threshold: Threshold for binary classification.

        Returns:
            Comprehensive test metrics dictionary.
        """
        all_logits = []
        all_gts = []

        self.model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                images = batch["image"].to(self.device)
                clinical_features = (
                    batch["clinical_features"].to(self.device)
                    if self.config.get("clinical_features", False)
                    else None
                )
                labels = batch["label"][:, 1].to(self.device)

                logits, att_weights, _ = self.model(images, clinical_features)

                if att_weights is not None:
                    log_attention_weights(att_weights, step=batch_index, prefix="Test")

                all_logits.append(logits.detach())
                all_gts.append(labels.detach())

        all_logits = torch.cat(all_logits, dim=0)
        all_gts = torch.cat(all_gts, dim=0)
        probabilities = torch.sigmoid(all_logits)
        preds = (probabilities >= optimal_threshold).int()

        comprehensive_test_metrics = metrics_module.calculate_metrics(
            all_logits, all_gts, optimal_threshold, split="Test"
        )
        metrics_module.log_roc_curve(all_logits, all_gts, prefix="Test/")

        if self.enable_wandb and wandb is not None and wandb.run is not None:
            wandb.log(comprehensive_test_metrics)

        # Save predictions to CSV
        df = pd.DataFrame(
            {
                "targets": all_gts.view(-1).cpu().numpy(),
                "logits": all_logits.view(-1).cpu().numpy(),
                "probabilities": probabilities.view(-1).cpu().numpy(),
                "preds": preds.view(-1).cpu().numpy(),
            }
        )
        save_dir = Path("test").absolute()
        save_dir.mkdir(parents=True, exist_ok=True)
        run_id = (
            wandb.run.id if wandb is not None and wandb.run is not None else "local"
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"test/test_results_{run_id}_{timestamp}.csv")

        return comprehensive_test_metrics
