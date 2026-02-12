"""Training engines for CRS prediction models.

Contains:
- log_attention_weights: visualize and log attention matrices
- Engine: training/validation loop for ViT classifier
- MLPEngine: training/validation loop for MLP classifier
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rich import print
from rich.console import Console
from rich.theme import Theme

from crs_prediction import metrics as metrics_module

try:
    import wandb
except ImportError:
    wandb = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

custom_theme = Theme(
    {
        "epoch": "#d7af00",
        "batch": "cyan",
        "train": "cyan",
        "val": "dark_magenta",
        "val_log": "violet",
    }
)


def log_attention_weights(
    att_weights,
    step: int,
    prefix: str = "Training",
    save: bool = False,
    save_path: str = "",
):
    """Log attention matrix visualization to W&B and optionally save to disk."""
    if plt is None or sns is None:
        return
    if torch.is_tensor(att_weights):
        att_weights = att_weights.cpu().detach().numpy()

    num_samples = min(att_weights.shape[0], 4)
    fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 5))
    if num_samples == 1:
        axes = [axes]
    for sample_idx in range(num_samples):
        ax = axes[sample_idx]
        sns.heatmap(
            att_weights[sample_idx],
            ax=ax,
            cmap="viridis",
            xticklabels=False,
            yticklabels=False,
        )
        ax.set_title(f"Sample {sample_idx + 1}")
    plt.tight_layout()

    if wandb is not None and wandb.run is not None:
        wandb.log({f"{prefix}/attention_matrix": wandb.Image(fig), "epoch": step})
    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


class Engine:
    """Training and evaluation engine for ViT-based CRS classifier."""

    def __init__(self, model, config, train_loader, val_loader, device, run_dir=None):
        self.console = Console(theme=custom_theme)
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_dir = run_dir

        self.pos_weights = self._compute_pos_weight()
        self.criterion = self._get_criterion()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        self.warmup_steps = config.get("warmup_steps", 100)
        self.main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=config["scheduler_patience"],
            min_lr=1.0e-8,
            threshold=1.0e-4,
        )

        schedulers = {
            "cosine": optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config["epochs"] * len(train_loader) / config["scheduler_step"],
                eta_min=0,
            ),
            "step": optim.lr_scheduler.StepLR(
                self.optimizer,
                config["epochs"] * len(train_loader) / config["scheduler_step"],
                gamma=config["scheduler_gamma"],
            ),
        }
        self.secondary_scheduler = schedulers[config["scheduler_type"]]

        self.best_val_loss = float("inf")
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.current_step = 0
        self.initial_lr = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.clinical_features = config["clinical_features"]
        self.gradient_accumulation_steps = config.get("accumulation_steps", 1)

        if wandb is not None and wandb.run is not None:
            wandb.watch(self.model, log_freq=2, log="all")
            wandb.define_metric("epoch")
            wandb.define_metric("train", step_metric="epoch")
            wandb.define_metric("val", step_metric="epoch")
            wandb.define_metric("Training", step_metric="epoch")
            wandb.define_metric("Validation", step_metric="epoch")
            wandb.define_metric("batch")
            wandb.define_metric("hyperp", step_metric="batch")

    def _compute_pos_weight(self):
        all_labels = []
        for batch in self.train_loader:
            labels = batch["label"]
            binary_labels = (labels[:, 1] > 0.5).cpu().numpy()
            all_labels.extend(binary_labels)
        labels_array = np.array(all_labels)
        class_counts = np.bincount(labels_array.astype(int), minlength=2)
        class_counts = np.maximum(class_counts, 1e-8)
        pos_weight = class_counts[0] / class_counts[1]
        self.console.print(
            f"Class distribution: Negative={class_counts[0]}, Positive={class_counts[1]}",
            style="epoch",
        )
        self.console.print(f"Computed positive weight: {pos_weight:.4f}", style="epoch")
        return torch.tensor([pos_weight], device=self.device)

    def _get_criterion(self):
        if self.config["loss_function"] == "BCE":
            return nn.BCEWithLogitsLoss()
        elif self.config["loss_function"] == "WBCE":
            return nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
        raise ValueError(f"Unknown loss function: {self.config['loss_function']}")

    def _format_tr_values(self, row):
        formatted = []
        for val in row:
            val = round(float(val), 1)
            tag = "CRS 1" if val == 1.0 else "CRS 0"
            color = "orange1" if val == 1.0 else "violet"
            formatted.append(f"[{color}]{tag:<12}[/{color}]")
        return formatted

    def _format_truefalse_values(self, row):
        formatted = []
        for val in row:
            val = round(float(val), 1)
            tag = "True" if val == 1.0 else "False"
            color = "green" if val == 1.0 else "red"
            formatted.append(f"[{color}]{tag:<12}[/{color}]")
        return formatted

    def _format_probs(self, probs):
        return [f"{round(float(p.item()), 2):<12}, " for p in probs]

    def train_epoch(self, epoch, num_epochs):
        self.model.train()
        self.console.print(
            f"Epoch [yellow]{epoch + 1}/{num_epochs}[/yellow]", style="epoch"
        )
        self.console.print("TRAINING", style="train")

        epoch_metrics = {"Training/loss": 0.0}
        total_batches = len(self.train_loader)
        all_logits, all_gts = [], []

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            clinical_features = (
                batch["clinical_features"].to(self.device)
                if self.clinical_features
                else None
            )
            labels = batch["label"][:, 1].to(self.device).unsqueeze(1)

            logits, att_weights, _ = self.model(images, clinical_features)
            loss = self.criterion(logits, labels)
            scaled_loss = loss / self.gradient_accumulation_steps
            scaled_loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                if att_weights is not None:
                    log_attention_weights(att_weights, step=epoch, prefix="Training")
                if wandb is not None and wandb.run is not None:
                    wandb.log({"LR": self.optimizer.param_groups[0]["lr"]})

            all_logits.append(logits.detach())
            all_gts.append(labels.detach())

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                binary_preds = (probs > 0.5).float()
                epoch_metrics["Training/loss"] += loss.item()

            self.secondary_scheduler.step()

            self.console.print(
                f"Batch [cyan]{batch_idx + 1}/{total_batches}[/cyan]", style="batch"
            )
            self.console.print(
                f">> Probs: {[self._format_probs(probs)]}", style="bright_cyan"
            )
            self.console.print(
                f">> Preds: {[self._format_tr_values(row) for row in binary_preds]}",
                style="bright_cyan",
            )
            self.console.print(
                f">> Targs: {[self._format_tr_values(row) for row in labels]}",
                style="bright_cyan",
            )
            self.console.print(
                f">> Acc  : {[self._format_truefalse_values(row) for row in (binary_preds == labels).tolist()]}",
                style="bright_cyan",
            )
            self.console.print(
                f">> Loss : [white]{loss.item():.3f}[/white]", style="bright_cyan"
            )
            self.console.print()

            if wandb is not None and wandb.run is not None:
                wandb.log(
                    {
                        "batch": batch_idx,
                        "batch/learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )

            del images, labels, logits, loss

        return (
            {k: v / total_batches for k, v in epoch_metrics.items()},
            att_weights,
            torch.cat(all_logits, dim=0),
            torch.cat(all_gts, dim=0),
        )

    def validate(self, epoch, num_epochs):
        self.model.eval()
        self.console.print(
            f"Epoch [yellow]{epoch + 1}/{num_epochs}[/yellow]", style="epoch"
        )
        self.console.print("VALIDATION", style="val")

        epoch_metrics = {"Validation/loss": 0.0}
        total_batches = len(self.val_loader)
        all_logits, all_gts = [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                images = batch["image"].to(self.device)
                clinical_features = (
                    batch["clinical_features"].to(self.device)
                    if self.clinical_features
                    else None
                )
                labels = batch["label"][:, 1].to(self.device).unsqueeze(1)

                logits, att_weights, _ = self.model(images, clinical_features)
                loss = self.criterion(logits, labels)

                if att_weights is not None:
                    log_attention_weights(att_weights, step=epoch, prefix="Validation")

                probs = torch.sigmoid(logits)
                binary_preds = (probs > 0.5).float()
                all_logits.append(logits.detach())
                all_gts.append(labels.detach())
                epoch_metrics["Validation/loss"] += loss.item()

                self.console.print(
                    f"Batch [magenta]{batch_idx + 1}/{total_batches}[/magenta]",
                    style="val_log",
                )
                self.console.print(
                    f">> Probs: {[self._format_probs(probs)]}", style="magenta"
                )
                self.console.print(
                    f">> Preds: {[self._format_tr_values(row) for row in binary_preds]}",
                    style="magenta",
                )
                self.console.print(
                    f">> Targs: {[self._format_tr_values(row) for row in labels]}",
                    style="magenta",
                )
                self.console.print(
                    f">> Loss : [white]{loss.item():.3f}[/white]", style="magenta"
                )
                self.console.print()

                del images, labels, logits, loss

        return (
            {k: v / total_batches for k, v in epoch_metrics.items()},
            torch.cat(all_logits, dim=0),
            torch.cat(all_gts, dim=0),
        )

    def _update_scheduler(self, val_loss):
        old_lr = self.optimizer.param_groups[0]["lr"]
        if self.current_step < self.warmup_steps and self.warmup_steps > 0:
            self.current_step += 1
        else:
            self.main_scheduler.step(val_loss)
        new_lr = self.optimizer.param_groups[0]["lr"]
        if new_lr != old_lr:
            print(f"[red]Learning rate changed: {old_lr:.2e} -> {new_lr:.2e}[/red]")
        else:
            print(f"[green]Current Learning rate: {new_lr:.2e}[/green]")
        print()

    def train(self, num_epochs: int):
        best_model_state = None
        no_improvement_epochs = 0
        best_val_loss = float("inf")
        prev_train_metrics = {"Training/loss": 0.0}
        prev_val_metrics = {"Validation/loss": 0.0}

        self.console.print(
            f"Initial learning rate: [yellow]{self.initial_lr}[/yellow]", style="epoch"
        )
        self.console.print(
            f"Batch size: [yellow]{self.batch_size}[/yellow]", style="epoch"
        )
        self.console.print(
            f"Gradient accumulation steps: [yellow]{self.gradient_accumulation_steps}[/yellow]",
            style="epoch",
        )

        trainable = sum(p.requires_grad for p in self.model.parameters())
        total = sum(1 for _ in self.model.parameters())
        self.console.print(
            f"Trainable parameters: [yellow]{trainable}/{total}[/yellow]", style="epoch"
        )
        print()

        if trainable == 0:
            raise ValueError("No trainable parameters found in the model!")

        for epoch in range(num_epochs):
            train_metrics, att_weights, logits, gts = self.train_epoch(
                epoch, num_epochs
            )
            comp_train = metrics_module.calculate_metrics(
                logits, gts, 0.5, split="Training"
            )
            comp_train["epoch"] = epoch
            if wandb is not None and wandb.run is not None:
                wandb.log(comp_train)

            val_metrics, logits, gts = self.validate(epoch, num_epochs)
            comp_val = metrics_module.calculate_metrics(
                logits, gts, 0.5, split="Validation"
            )
            comp_val["epoch"] = epoch
            if wandb is not None and wandb.run is not None:
                wandb.log(comp_val)

            if wandb is not None and wandb.run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "hyperp/learning_rate": self.optimizer.param_groups[0]["lr"],
                        **train_metrics,
                        **val_metrics,
                    }
                )

            self.console.print(
                f"SUMMARY EPOCH [yellow]{epoch + 1}/{num_epochs}[/yellow]",
                style="epoch",
            )
            self.console.print(
                f"TRAIN[cyan] Loss:{prev_train_metrics.get('Training/loss', -1):.4f} -> {train_metrics.get('Training/loss', -1):.4f}[/cyan]",
                style="batch",
            )
            self.console.print(
                f"VAL  [magenta] Loss:{prev_val_metrics.get('Validation/loss', -1):.4f} -> {val_metrics.get('Validation/loss', -1):.4f}[/magenta]",
                style="val_log",
            )
            print()

            prev_train_metrics = train_metrics
            prev_val_metrics = val_metrics

            self._update_scheduler(val_metrics["Validation/loss"])

            if val_metrics["Validation/loss"] < best_val_loss:
                best_val_loss = val_metrics["Validation/loss"]
                best_model_state = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
                no_improvement_epochs = 0
                self._save_checkpoint(epoch, val_metrics)
            else:
                no_improvement_epochs += 1

            if comp_val["Validation/f1_score"] > self.best_val_f1:
                self.best_val_f1 = comp_val["Validation/f1_score"]

            if no_improvement_epochs >= self.config["early_stopping_patience"]:
                self.console.print(
                    f"STOP TRAINING: early stopping after [red]{epoch + 1}[/red] epochs",
                    style="red",
                )
                self.console.print(
                    f"Best validation loss: [red]{best_val_loss:.4f}[/red]", style="red"
                )
                self.console.print(
                    f"Best validation F1: [red]{self.best_val_f1:.4f}[/red]",
                    style="red",
                )
                if best_model_state is not None:
                    self.console.print(
                        f"Restoring best model from epoch [red]{best_model_state['epoch']}[/red]",
                        style="red",
                    )
                    self._restore_best_model(best_model_state, att_weights, epoch)
                break

        if epoch == num_epochs - 1 and best_model_state is not None:
            self.console.print(
                f"Training completed. Restoring best model from epoch [yellow]{best_model_state['epoch']}[/yellow]",
                style="epoch",
            )
            self._restore_best_model(best_model_state, att_weights, epoch)

    def _restore_best_model(self, best_model_state, att_weights, epoch):
        self.model.load_state_dict(best_model_state["model_state_dict"])
        self.optimizer.load_state_dict(best_model_state["optimizer_state_dict"])
        log_attention_weights(att_weights, step=epoch, save=False)

    def _save_checkpoint(self, epoch: int, checkpoint_metrics: Dict[str, float]):
        try:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": checkpoint_metrics,
                "best_val_loss": self.best_val_loss,
                "best_val_f1": self.best_val_f1,
                "current_step": self.current_step,
                "current_lr": self.optimizer.param_groups[0]["lr"],
            }
            if self.run_dir is not None:
                from crs_prediction.utils import save_model_artifact

                save_path = save_model_artifact(
                    checkpoint, self.config, self.run_dir, filename="best_model.pt"
                )
            else:
                save_dir = Path("checkpoints").absolute()
                save_dir.mkdir(parents=True, exist_ok=True)
                run_id = (
                    wandb.run.id
                    if wandb is not None and wandb.run is not None
                    else "local"
                )
                save_path = save_dir / f"best_model_{run_id}.pt"
                torch.save(checkpoint, str(save_path))
            self.console.print(f"Saved checkpoint to {save_path}", style="epoch")
            print()
        except Exception as e:
            self.console.print(f"Warning: Failed to save checkpoint: {e}", style="red")


class MLPEngine:
    """Training engine for MLP classifier on clinical features."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        train_loader,
        val_loader,
        device: torch.device,
        run_dir=None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_dir = run_dir

        self.pos_weight = self._compute_pos_weight()

        if config.get("loss_function", "WBCE") == "WBCE":
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-6),
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=config.get("scheduler_patience", 20),
            min_lr=1e-8,
        )

        self.best_val_loss = float("inf")
        self.best_val_f1 = 0.0

    def _compute_pos_weight(self) -> torch.Tensor:
        all_labels = []
        for batch in self.train_loader:
            labels = batch["label"][:, 1].numpy()
            all_labels.extend(labels)
        labels_array = np.array(all_labels)
        class_counts = np.bincount(labels_array.astype(int), minlength=2)
        class_counts = np.maximum(class_counts, 1e-8)
        pos_weight = class_counts[0] / class_counts[1]
        return torch.tensor([pos_weight], device=self.device)

    def train_epoch(self, epoch: int) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        self.model.train()
        total_loss = 0.0
        all_logits, all_gts = [], []

        for batch in self.train_loader:
            features = batch["features"].to(self.device)
            labels = batch["label"][:, 1].to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_logits.append(logits.detach())
            all_gts.append(labels.detach())

        return (
            {"Training/loss": total_loss / len(self.train_loader)},
            torch.cat(all_logits),
            torch.cat(all_gts),
        )

    def validate(self, epoch: int) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        self.model.eval()
        total_loss = 0.0
        all_logits, all_gts = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)
                labels = batch["label"][:, 1].to(self.device).unsqueeze(1)

                logits = self.model(features)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                all_logits.append(logits.detach())
                all_gts.append(labels.detach())

        return (
            {"Validation/loss": total_loss / len(self.val_loader)},
            torch.cat(all_logits),
            torch.cat(all_gts),
        )

    def train(self, num_epochs: int):
        import logging

        logger = logging.getLogger(__name__)
        best_model_state = None
        no_improvement = 0

        for epoch in range(num_epochs):
            train_metrics, train_logits, train_gts = self.train_epoch(epoch)
            train_comp = metrics_module.calculate_metrics(
                train_logits, train_gts, 0.5, split="Training"
            )
            train_comp.update(train_metrics)
            train_comp["epoch"] = epoch
            if wandb is not None and wandb.run is not None:
                wandb.log(train_comp)

            val_metrics, val_logits, val_gts = self.validate(epoch)
            val_comp = metrics_module.calculate_metrics(
                val_logits, val_gts, 0.5, split="Validation"
            )
            val_comp.update(val_metrics)
            val_comp["epoch"] = epoch
            if wandb is not None and wandb.run is not None:
                wandb.log(val_comp)

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['Training/loss']:.4f}, "
                f"Val Loss: {val_metrics['Validation/loss']:.4f}, "
                f"Val F1: {val_comp['Validation/f1_score']:.4f}, "
                f"Val AUC: {val_comp.get('Validation/roc_auc', 0):.4f}"
            )

            self.scheduler.step(val_metrics["Validation/loss"])
            if wandb is not None and wandb.run is not None:
                wandb.log({"LR": self.optimizer.param_groups[0]["lr"]})

            if val_metrics["Validation/loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["Validation/loss"]
                self.best_val_f1 = val_comp["Validation/f1_score"]
                best_model_state = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": self.best_val_loss,
                    "val_f1": self.best_val_f1,
                }
                no_improvement = 0
                self._save_checkpoint(epoch, val_comp)
            else:
                no_improvement += 1

            if no_improvement >= self.config.get("early_stopping_patience", 50):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state["model_state_dict"])
            logger.info(
                f"Restored best model from epoch {best_model_state['epoch'] + 1}"
            )

    def _save_checkpoint(self, epoch: int, metrics_dict: Dict):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics_dict,
        }
        if self.run_dir is not None:
            from crs_prediction.utils import save_model_artifact

            save_model_artifact(
                checkpoint, self.config, self.run_dir, filename="best_mlp_model.pt"
            )
        else:
            save_dir = Path("checkpoints").absolute()
            save_dir.mkdir(parents=True, exist_ok=True)
            run_id = (
                wandb.run.id if wandb is not None and wandb.run is not None else "local"
            )
            save_path = save_dir / f"best_mlp_model_{run_id}.pt"
            torch.save(checkpoint, save_path)
