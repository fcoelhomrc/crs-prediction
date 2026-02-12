"""Bootstrap test evaluation script.

Loads a trained model checkpoint and evaluates on a test dataset
with bootstrapped resampling for robust metric estimation.

Usage:
    crs-test --config config.yaml --checkpoint best_model.pt --data_file test.json
"""

import argparse
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, Subset

from crs_prediction import metrics as metrics_module
from crs_prediction.dataset import PCSDataset
from crs_prediction.model import VitClassifier
from crs_prediction.utils import detect_available_cpus

console = Console(force_terminal=True)
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
)
logger = logging.getLogger("rich")


def predict(model, loader, device, config, progress, task):
    """Run inference and collect per-sample results with metadata.

    Returns:
        Tuple of (metrics_dict, results_list).
    """
    all_logits, all_gts = [], []
    record_ids, ages, ca125s, original_crss, crs_labels = [], [], [], [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            clinical_features = (
                batch["clinical_features"].to(device)
                if config.get("clinical_features", False)
                else None
            )
            labels = batch["label"][:, 1].to(device)
            logits, _, _ = model(images, clinical_features)
            all_logits.append(logits.detach())
            all_gts.append(labels.detach())

            record_ids.append(batch["record_id"][0])
            ages.append(batch["age"].item())
            ca125s.append(batch["ca125"].item())
            original_crss.append(batch["original_crs"].item())
            crs_labels.append(batch["crs_label"][0])

            progress.update(task, advance=1)

    all_logits = torch.cat(all_logits, dim=0)
    all_gts = torch.cat(all_gts, dim=0)
    probabilities = torch.sigmoid(all_logits)
    preds = (probabilities >= 0.5).int().view(-1)

    test_metrics = metrics_module.calculate_metrics(
        all_logits, all_gts, 0.5, split="Test"
    )

    results_list = []
    for i in range(len(record_ids)):
        results_list.append(
            {
                "record_id": record_ids[i],
                "age": ages[i],
                "ca125": ca125s[i],
                "original_crs": original_crss[i],
                "crs_label": crs_labels[i],
                "ground_truth": int(all_gts[i].cpu().item()),
                "predicted_prob": float(probabilities[i].cpu().item()),
                "predicted_class": int(preds[i].cpu().item()),
            }
        )

    return test_metrics, results_list


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on test dataset with bootstrapping."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config YAML"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_file", type=str, required=True, help="Path to test data JSON"
    )
    parser.add_argument("--seg", action="store_true", default=False)
    parser.add_argument("--no_seg_as_im", action="store_true", default=False)
    parser.add_argument("--no_clinical_fe", action="store_true", default=False)
    parser.add_argument("--disease_sites", type=str, default="om")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--cohort", type=str, default="IEO", choices=["IEO", "NEOV"])
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args()

    logger.info("Loading dataset...")
    test_dataset = PCSDataset(
        data_file=args.data_file,
        is_training=False,
        seg=args.seg,
        seg_as_im=not args.no_seg_as_im,
        clinical_fe=not args.no_clinical_fe,
        disease_sites=args.disease_sites,
        excluded_years=[],
        cohort=args.cohort,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=detect_available_cpus(),
        pin_memory=True,
        drop_last=False,
    )

    logger.info(f"Found {len(test_dataset)} samples")
    logger.info(f"Stats: {test_dataset.stats}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model = VitClassifier(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Model loaded!")

    # Bootstrap
    n_iterations = args.n_bootstrap
    n_size = len(test_dataset)
    bootstrap_metrics = []
    bootstrap_results = []

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[metrics]}"),
        console=console,
        refresh_per_second=10,
    )

    logger.info("Starting bootstrap...")
    with progress:
        task_bootstrap = progress.add_task("Bootstrap", total=n_iterations, metrics="")
        task_eval = progress.add_task("Evaluating...", total=n_size, metrics="")

        for i in range(n_iterations):
            indices = np.random.choice(np.arange(n_size), size=n_size, replace=True)
            bootstrap_subset = Subset(test_dataset, indices)
            bootstrap_loader = DataLoader(bootstrap_subset, batch_size=1, shuffle=False)

            test_metrics, results_list = predict(
                model, bootstrap_loader, device, config, progress, task_eval
            )
            bootstrap_metrics.append(test_metrics)

            for result in results_list:
                result["bootstrap_iter"] = i
            bootstrap_results.extend(results_list)

            f1 = test_metrics.get("Test/f1_score", 0.0)
            roc_auc = test_metrics.get("Test/roc_auc", 0.0)
            auc_str = f"{roc_auc:.3f}" if roc_auc is not None else "--"
            progress.update(
                task_bootstrap, advance=1, metrics=f"F1: {f1:.3f} | AUC: {auc_str}"
            )
            progress.reset(task_eval)

    # Save results
    name = args.run_name or "test"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs", exist_ok=True)

    pd.DataFrame(bootstrap_metrics).to_csv(
        f"outputs/METRICS_{n_iterations}_bootstrap_{name}_{timestamp}.csv", index=False
    )
    pd.DataFrame(bootstrap_results).to_csv(
        f"outputs/RESULTS_{n_iterations}_bootstrap_{name}_{timestamp}.csv", index=False
    )

    logger.info("All done!")


if __name__ == "__main__":
    main()
