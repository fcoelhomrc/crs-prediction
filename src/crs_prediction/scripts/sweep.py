"""Weights & Biases sweep launcher.

Usage:
    crs-sweep --config config.yaml
"""

import argparse

import wandb

from crs_prediction.scripts.train import run_training
from crs_prediction.utils import load_config


def sweep_agent(config: dict):
    with wandb.init(config=config):
        config = wandb.config
        run_training(dict(config))


def main():
    parser = argparse.ArgumentParser(description="Launch W&B sweep for CRS training.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    config = load_config(args.config)
    sweep_agent(config)


if __name__ == "__main__":
    main()
