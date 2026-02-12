"""Utility functions for device detection and environment setup."""

import os
import stat
import subprocess
from pathlib import Path

import torch
import yaml


def detect_available_cpus() -> int:
    """Detect the number of available CPUs for data loading.

    If running under SLURM, attempts to detect allocated CPUs for the current job.
    Otherwise, falls back to the number of CPUs available to the process.

    Returns:
        Number of available CPUs (at least 1).
    """
    try:
        job_id = os.environ["SLURM_JOB_ID"]
        command = f"squeue --Format=JobID,cpus-per-task | grep {int(job_id)}"
        output = subprocess.check_output(command, shell=True)[5:-4].replace(b" ", b"")
        cpus = int(output.decode("utf-8"))
        cpus_affinity = len(os.sched_getaffinity(0))
        return min(cpus, cpus_affinity)
    except Exception:
        return 1


def get_device(config: dict | None = None) -> torch.device:
    """Get the torch device from config or auto-detect.

    Args:
        config: Optional config dict with a 'device' key.

    Returns:
        torch.device instance.
    """
    if config and "device" in config:
        device_str = config["device"]
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


def make_run_dir(
    run_name: str | None = None,
    base_dir: str | Path = "models",
) -> Path:
    """Create a timestamped run directory under base_dir.

    Directory name format: ``YYYY-MM-DD_<run_name>``
    (or just the date if no run_name is given).

    Returns:
        Absolute path to the created directory.
    """
    from datetime import datetime

    date_str = datetime.now().strftime("%Y-%m-%d")
    dirname = f"{date_str}_{run_name}" if run_name else date_str
    run_dir = Path(base_dir).absolute() / dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_model_artifact(
    checkpoint: dict,
    config: dict,
    run_dir: str | Path,
    filename: str = "best_model.pt",
) -> Path:
    """Save a model checkpoint alongside a read-only copy of its config.

    Creates ``<run_dir>/<filename>`` and ``<run_dir>/config.yaml`` (read-only).

    Args:
        checkpoint: State dict / checkpoint to ``torch.save``.
        config: Training config to freeze as YAML.
        run_dir: Directory to write into (created if needed).
        filename: Name for the ``.pt`` file.

    Returns:
        Path to the saved checkpoint file.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    pt_path = run_dir / filename
    torch.save(checkpoint, str(pt_path))

    # Save read-only config (overwrite if exists)
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        config_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    config_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # 0o444

    return pt_path


def load_config(config_path: str) -> dict:
    """Load a YAML config file and validate required fields.

    Handles both flat configs and nested 'parameters' structures (for sweep configs).

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Loaded configuration dictionary.

    Raises:
        ValueError: If any required field is missing.
    """
    with open(config_path, "r") as f:
        config_raw = yaml.safe_load(f)

    if "parameters" in config_raw:
        config = {}
        for key, value in config_raw["parameters"].items():
            if isinstance(value, dict) and "value" in value:
                config[key] = value["value"]
            elif isinstance(value, dict) and "values" in value:
                config[key] = value["values"][0] if value["values"] else None
            else:
                config[key] = value
    else:
        config = config_raw

    required_fields = [
        "train_paths",
        "val_paths",
        "test_paths",
        "batch_size",
        "epochs",
        "learning_rate",
    ]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    return config
