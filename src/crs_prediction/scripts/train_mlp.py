"""Training script for MLP classifier using clinical features.

Features: tumor volume (from segmentation), age, CA125.

Usage:
    crs-train-mlp --config config_mlp.yaml [--run_name my_run]
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import wandb
import yaml
from rich.console import Console
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from crs_prediction import metrics as metrics_module
from crs_prediction.dataset import MLPDataset
from crs_prediction.engine import MLPEngine
from crs_prediction.model import MLP
from crs_prediction.utils import make_run_dir, save_model_artifact

console = Console(force_terminal=True)
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
)
logger = logging.getLogger("rich")


def predict_test(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    """Run predictions on test set.

    Returns:
        Tuple of (metrics_dict, results_list).
    """
    model.eval()
    all_logits, all_gts = [], []
    results_list = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            labels = batch["label"][:, 1].to(device)

            logits = model(features)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int()

            all_logits.append(logits)
            all_gts.append(labels)

            for i in range(len(batch["record_id"])):
                results_list.append(
                    {
                        "record_id": batch["record_id"][i],
                        "age": batch["age"][i].item()
                        if torch.is_tensor(batch["age"][i])
                        else batch["age"][i],
                        "ca125": batch["ca125"][i].item()
                        if torch.is_tensor(batch["ca125"][i])
                        else batch["ca125"][i],
                        "tumor_volume": batch["tumor_volume"][i].item()
                        if torch.is_tensor(batch["tumor_volume"][i])
                        else batch["tumor_volume"][i],
                        "original_crs": batch["original_crs"][i].item()
                        if torch.is_tensor(batch["original_crs"][i])
                        else batch["original_crs"][i],
                        "crs_label": batch["crs_label"][i],
                        "ground_truth": int(labels[i].cpu().item()),
                        "predicted_prob": float(probs[i].cpu().item()),
                        "predicted_class": int(preds[i].cpu().item()),
                    }
                )

    all_logits = torch.cat(all_logits, dim=0)
    all_gts = torch.cat(all_gts, dim=0)
    test_metrics = metrics_module.calculate_metrics(
        all_logits, all_gts, 0.5, split="Test"
    )
    return test_metrics, results_list


def load_config(config_path: str) -> dict:
    """Load and validate config from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "parameters" in config:
        flat_config = {}
        for key, value in config["parameters"].items():
            if isinstance(value, dict) and "value" in value:
                flat_config[key] = value["value"]
            else:
                flat_config[key] = value
        config = flat_config

    # Defaults
    config.setdefault("hidden_dims", [32, 16, 8])
    config.setdefault("dropout", 0.10)
    config.setdefault("learning_rate", 1e-4)
    config.setdefault("batch_size", 4)
    config.setdefault("epochs", 1000)
    config.setdefault("early_stopping_patience", 800)
    config.setdefault("loss_function", "WBCE")
    config.setdefault("weight_decay", 1e-3)
    config.setdefault("scheduler_patience", 200)
    config.setdefault("disease_sites", "om")
    config.setdefault("excluded_years", [])
    config.setdefault("use_volume", True)
    config.setdefault("use_age", True)
    config.setdefault("use_ca125", True)

    required = ["train_paths", "val_paths", "test_paths"]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    return config


def main():
    parser = argparse.ArgumentParser(description="Train MLP on clinical features")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--run_name", default="", help="Optional run name for outputs")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MLP TRAINING - CLINICAL FEATURES")
    logger.info("=" * 60)

    config = load_config(args.config)

    device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    if "cuda" in device_str and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable, falling back to CPU")
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    run_name = args.run_name or f"mlp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = make_run_dir(run_name)
    wandb.init(project="CARS26", name=run_name, config=config)
    logger.info(f"Wandb run ID: {wandb.run.id}")
    logger.info(f"Run directory: {run_dir}")

    # Feature flags
    use_volume = config.get("use_volume", True)
    use_age = config.get("use_age", True)
    use_ca125 = config.get("use_ca125", True)

    # Datasets
    train_dataset = MLPDataset(
        data_file=config["train_paths"],
        is_training=True,
        excluded_years=config.get("excluded_years", []),
        disease_sites=config.get("disease_sites", "om"),
        normalize=True,
        use_volume=use_volume,
        use_age=use_age,
        use_ca125=use_ca125,
    )
    train_mean, train_std = train_dataset.get_normalization_stats()

    val_dataset = MLPDataset(
        data_file=config["val_paths"],
        is_training=False,
        excluded_years=config.get("excluded_years", []),
        disease_sites=config.get("disease_sites", "om"),
        normalize=True,
        use_volume=use_volume,
        use_age=use_age,
        use_ca125=use_ca125,
    )
    val_dataset.set_normalization_stats(train_mean, train_std)

    test_dataset = MLPDataset(
        data_file=config["test_paths"],
        is_training=False,
        excluded_years=[],
        disease_sites=config.get("disease_sites", "om"),
        normalize=True,
        use_volume=use_volume,
        use_age=use_age,
        use_ca125=use_ca125,
    )
    test_dataset.set_normalization_stats(train_mean, train_std)

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # Model
    model = MLP(
        input_dim=train_dataset.input_dim,
        hidden_dims=config.get("hidden_dims", [64, 32, 16]),
        dropout=config.get("dropout", 0.25),
        num_classes=1,
    ).to(device)

    logger.info(f"Model: {model}")
    logger.info(
        f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Train
    engine = MLPEngine(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        run_dir=run_dir,
    )
    engine.train(num_epochs=config["epochs"])

    # Test
    logger.info("Running test evaluation...")
    test_metrics, results_list = predict_test(model, test_loader, device)
    test_metrics["epoch"] = config["epochs"]
    wandb.log(test_metrics)

    logger.info(f"Test Accuracy: {test_metrics['Test/accuracy']:.4f}")
    logger.info(f"Test F1: {test_metrics['Test/f1_score']:.4f}")
    logger.info(f"Test AUC: {test_metrics.get('Test/roc_auc', 'N/A')}")

    # Save outputs
    output_dir = Path("outputs").absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = wandb.run.id

    pd.DataFrame([test_metrics]).to_csv(
        output_dir / f"METRICS_mlp_{run_name}_{timestamp}.csv", index=False
    )
    pd.DataFrame(results_list).to_csv(
        output_dir / f"RESULTS_mlp_{run_name}_{timestamp}.csv", index=False
    )

    # Save final model into the run directory
    save_model_artifact(
        {
            "model_state_dict": model.state_dict(),
            "normalization_stats": {
                "mean": train_mean.tolist(),
                "std": train_std.tolist(),
            },
        },
        config,
        run_dir,
        filename="final_mlp_model.pt",
    )

    wandb.finish()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
