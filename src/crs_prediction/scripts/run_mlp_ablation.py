"""MLP ablation pipeline.

Runs train_mlp with all feature combinations, then generates a results table.

Usage:
    crs-mlp-ablation
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()

ABLATION_CONFIGS = [
    ("volume_only", True, False, False, "Volume only"),
    ("volume_age_ca125", True, True, True, "Volume + Age + CA-125"),
]

BASE_CONFIG = {
    "batch_size": 4,
    "epochs": 200,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "loss_function": "WBCE",
    "scheduler_patience": 20,
    "early_stopping_patience": 50,
    "disease_sites": "om",
    "excluded_years": [],
    "train_paths": "inputs/json/ieo_train.json",
    "val_paths": "inputs/json/ieo_val.json",
    "test_paths": "inputs/json/ieo_test.json",
}


def create_config_file(
    key: str, use_volume: bool, use_age: bool, use_ca125: bool
) -> Path:
    config = BASE_CONFIG.copy()
    config["use_volume"] = use_volume
    config["use_age"] = use_age
    config["use_ca125"] = use_ca125

    config_dir = Path("ablation/mlp_configs")
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / f"config_{key}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


def run_training(config_path: Path, run_name: str) -> bool:
    cmd = [
        sys.executable,
        "-m",
        "crs_prediction.scripts.train_mlp",
        "--config",
        str(config_path),
        "--run_name",
        run_name,
    ]
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return False


def main():
    console.print("[bold blue]MLP ABLATION PIPELINE[/bold blue]")

    table = Table(title="Ablation Configurations")
    table.add_column("Key", style="cyan")
    table.add_column("Volume", justify="center")
    table.add_column("Age", justify="center")
    table.add_column("CA-125", justify="center")
    table.add_column("Description")

    for key, use_vol, use_age, use_ca125, desc in ABLATION_CONFIGS:
        table.add_row(
            key,
            "[green]\u2713[/green]" if use_vol else "[red]\u2717[/red]",
            "[green]\u2713[/green]" if use_age else "[red]\u2717[/red]",
            "[green]\u2713[/green]" if use_ca125 else "[red]\u2717[/red]",
            desc,
        )
    console.print(table)

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for key, use_vol, use_age, use_ca125, desc in ABLATION_CONFIGS:
            task = progress.add_task(f"[cyan]Training {key}...", total=None)

            config_path = create_config_file(key, use_vol, use_age, use_ca125)
            console.print(f"\n[bold]Running: {desc}[/bold]")

            run_name = f"ablation_{key}"
            success = run_training(config_path, run_name)

            status = "Success" if success else "Failed"
            style = "[green]\u2713[/green]" if success else "[red]\u2717[/red]"
            console.print(f"  {style} {status}")
            results.append((key, desc, status))
            progress.remove_task(task)

    # Summary
    summary = Table(title="Results Summary")
    summary.add_column("Configuration", style="cyan")
    summary.add_column("Description")
    summary.add_column("Status", justify="center")
    for key, desc, status in results:
        style = "[green]\u2713[/green]" if status == "Success" else "[red]\u2717[/red]"
        summary.add_row(key, desc, style)
    console.print(summary)

    console.print("\n[bold green]MLP ablation complete![/bold green]")


if __name__ == "__main__":
    main()
