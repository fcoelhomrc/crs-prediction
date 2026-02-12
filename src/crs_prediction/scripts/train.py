"""Main training script for ViT-based CRS classifier.

Usage:
    crs-train --config config.yaml
    # or: python -m crs_prediction.scripts.train --config config.yaml
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from rich import print
from torch.utils.data import DataLoader

import wandb
from crs_prediction.dataset import PCSDataset
from crs_prediction.engine import Engine
from crs_prediction.eval import ModelEvaluator
from crs_prediction.model import VitClassifier
from crs_prediction.utils import (
    detect_available_cpus,
    load_config,
    make_run_dir,
    save_model_artifact,
)


def run_training(config: dict):
    """Run a full training + evaluation cycle with the given config."""
    run_name = config.get("run_name", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_dir = make_run_dir(run_name)

    with wandb.init(
        config=config,
        name=run_name,
    ):
        device_str = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        device = torch.device(device_str)

        model = VitClassifier(config).to(device)
        print("Model loaded")

        num_workers = detect_available_cpus()

        train_dataset = PCSDataset(
            data_file=config["train_paths"],
            excluded_years=[2015],
            transform=None,
            is_training=True,
            seg=False,
            seg_as_im=True,
            clinical_fe=config.get("clinical_features", False),
            eager=config.get("eager_loading", False),
        )
        val_dataset = PCSDataset(
            data_file=config["val_paths"],
            excluded_years=[2015],
            transform=None,
            is_training=False,
            seg=False,
            seg_as_im=True,
            clinical_fe=config.get("clinical_features", False),
            eager=config.get("eager_loading", False),
        )
        test_dataset = PCSDataset(
            data_file=config["test_paths"],
            excluded_years=[2015],
            transform=None,
            is_training=False,
            seg=False,
            seg_as_im=True,
            clinical_fe=config.get("clinical_features", False),
            eager=config.get("eager_loading", False),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        print("Datasets loaded")

        engine = Engine(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            run_dir=run_dir,
        )

        print("Engine loaded. Start training...")

        engine.train(num_epochs=config["epochs"])

        # Save final model into the same run directory
        save_path = save_model_artifact(
            {
                "model_state_dict": model.state_dict(),
                "learning_rate": config.get("learning_rate"),
                "batch_size": config.get("batch_size"),
                "epochs": config.get("epochs"),
            },
            config,
            run_dir,
            filename="final_model.pt",
        )
        print(f"[yellow]Saved final model to: {save_path}[/yellow]")

        # Evaluate on test set
        print()
        print("[orange3]TEST[/orange3]")
        evaluator = ModelEvaluator(model, device, config=config)
        test_dir = Path("test").absolute()
        test_dir.mkdir(parents=True, exist_ok=True)
        test_metrics = evaluator.predict(test_loader)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pd.DataFrame([test_metrics]).to_csv(
            f"test/test_metrics_{wandb.run.id}_{timestamp}.csv"
        )
        print(f"[orange3]Test metrics saved[/orange3]")


def main():
    parser = argparse.ArgumentParser(description="Train ViT-based CRS classifier.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--run_name", default=None, help="Run name (used for model directory)"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.run_name:
        config["run_name"] = args.run_name
    run_training(config)


if __name__ == "__main__":
    main()
