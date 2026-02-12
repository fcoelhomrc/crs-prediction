"""Sklearn ablation study with bootstrap evaluation.

Trains sklearn classifiers (logistic regression, random forest, SVM, gradient boosting)
on all 7 feature combinations and evaluates with bootstrap resampling.

Usage:
    crs-sklearn-ablation --model logistic --n-bootstrap 1000
    crs-sklearn-ablation --model random_forest --test-only neov_ds.json --run-name NEOV
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

console = Console()

ABLATION_CONFIGS = [
    ("age_only", False, True, False, "Age only"),
    ("ca125_only", False, False, True, "CA-125 only"),
    ("volume_only", True, False, False, "Volume only"),
    ("age_ca125", False, True, True, "Age + CA-125"),
    ("age_volume", True, True, False, "Age + Volume"),
    ("ca125_volume", True, False, True, "CA-125 + Volume"),
    ("volume_age_ca125", True, True, True, "Volume + Age + CA-125"),
]

MODELS = {
    "logistic": lambda: LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(
        n_estimators=100,
        random_state=42,
    ),
    "svm": lambda: SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=42,
    ),
}


def load_patients_from_json(
    data_file: str,
    disease_sites: List[int],
    excluded_years: List[int],
    use_volume: bool,
    use_age: bool,
    use_ca125: bool,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Load patient features, labels, and metadata from JSON."""
    sites_str = "_".join(map(str, sorted(disease_sites)))
    volume_cache_key = f"tumor_volume_mm3_sites_{sites_str}"

    with open(data_file, "r") as f:
        data = json.load(f)

    features_list, labels_list, metadata_list = [], [], []

    for patient in data["patients"]:
        if patient["year"] in excluded_years:
            continue

        seg_path = str(patient.get("segmentation_path", ""))
        if seg_path in ("-1", "-1.0", "") or not seg_path.strip():
            continue
        if volume_cache_key not in patient:
            continue

        tumor_volume = patient.get(volume_cache_key, 0.0)
        age = float(patient.get("Age at diagnosis", 0.0))
        ca125_raw = patient.get("CA125 (U/mL)", 0.0)
        ca125 = float(ca125_raw) if ca125_raw not in (-1, "-1") else 0.0

        feature_vec = []
        if use_volume:
            feature_vec.append(tumor_volume)
        if use_age:
            feature_vec.append(age)
        if use_ca125:
            feature_vec.append(ca125)
        if not feature_vec:
            continue

        features_list.append(feature_vec)
        original_label = patient["CRS"]
        labels_list.append(1 if original_label >= 1 else 0)
        metadata_list.append(
            {
                "record_id": str(patient.get("Record ID", "")),
                "age": age,
                "ca125": ca125,
                "tumor_volume": tumor_volume,
                "original_crs": original_label,
                "crs_label": str(patient.get("CRS: notes", "")),
            }
        )

    return np.array(features_list), np.array(labels_list), metadata_list


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    split: str = "Test",
) -> Dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    roc_auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else None
    pr_auc = average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else None

    prefix = f"{split}/"
    return {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}f1_score": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}roc_auc": roc_auc,
        f"{prefix}pr_auc": pr_auc,
        f"{prefix}cohen_kappa": cohen_kappa_score(y_true, y_pred),
        f"{prefix}confusion_matrix/TP": int(tp),
        f"{prefix}confusion_matrix/TN": int(tn),
        f"{prefix}confusion_matrix/FP": int(fp),
        f"{prefix}confusion_matrix/FN": int(fn),
    }


def train_and_evaluate_with_bootstrap(
    model_name: str,
    use_volume: bool,
    use_age: bool,
    use_ca125: bool,
    config: Dict,
    n_bootstrap: int,
    progress: Progress,
    bootstrap_task,
    test_only_path: str | None = None,
) -> Tuple[List[Dict], List[Dict]]:
    disease_site_map = {
        "om": [1],
        "pod_om": [1, 9],
        "abdominal_lesions": [1, 2, 3, 5, 6, 7],
        "lymph_nodes": [13, 14, 15, 17],
        "all": [1, 2, 3, 5, 6, 7, 9, 13, 14, 15, 17],
    }
    disease_sites = disease_site_map.get(config.get("disease_sites", "om"), [1])
    excluded_years = config.get("excluded_years", [])

    X_train, y_train, _ = load_patients_from_json(
        config["train_paths"],
        disease_sites,
        excluded_years,
        use_volume,
        use_age,
        use_ca125,
    )
    X_val, y_val, _ = load_patients_from_json(
        config["val_paths"],
        disease_sites,
        excluded_years,
        use_volume,
        use_age,
        use_ca125,
    )
    test_path = test_only_path if test_only_path else config["test_paths"]
    X_test, y_test, test_metadata = load_patients_from_json(
        test_path,
        disease_sites,
        [],
        use_volume,
        use_age,
        use_ca125,
    )

    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

    model = MODELS[model_name]()
    model.fit(X_train_scaled, y_train_full)

    y_prob_full = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_full = (y_prob_full >= 0.5).astype(int)

    n_test = len(y_test)
    bootstrap_metrics, bootstrap_results = [], []
    rng = np.random.default_rng(42)

    for i in range(n_bootstrap):
        indices = rng.choice(n_test, size=n_test, replace=True)
        metrics = calculate_metrics(
            y_test[indices],
            y_pred_full[indices],
            y_prob_full[indices],
            split="Test",
        )
        bootstrap_metrics.append(metrics)

        for idx in indices:
            meta = test_metadata[idx]
            bootstrap_results.append(
                {
                    "bootstrap_iter": i,
                    **meta,
                    "ground_truth": int(y_test[idx]),
                    "predicted_prob": float(y_prob_full[idx]),
                    "predicted_class": int(y_pred_full[idx]),
                }
            )

        f1 = metrics.get("Test/f1_score", 0.0)
        roc_auc = metrics.get("Test/roc_auc", 0.0)
        auc_str = f"{roc_auc:.3f}" if roc_auc is not None else "--"
        progress.update(
            bootstrap_task, advance=1, metrics=f"F1: {f1:.3f} | AUC: {auc_str}"
        )

    return bootstrap_metrics, bootstrap_results


def main():
    parser = argparse.ArgumentParser(description="Run sklearn ablation with bootstrap")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="logistic")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablation"))
    parser.add_argument("--test-only", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config with train/val/test paths. "
        "If not provided, uses hardcoded defaults.",
    )
    args = parser.parse_args()

    # Load config or use defaults
    if args.config:
        import yaml

        with open(args.config) as f:
            default_config = yaml.safe_load(f)
    else:
        default_config = {
            "disease_sites": "om",
            "excluded_years": [],
            "train_paths": "inputs/json/ieo_train.json",
            "val_paths": "inputs/json/ieo_val.json",
            "test_paths": "inputs/json/ieo_test.json",
        }

    console.print(f"[bold blue]SKLEARN ABLATION ({args.model.upper()})[/bold blue]")

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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_summary = []

    for key, use_vol, use_age, use_ca125, desc in ABLATION_CONFIGS:
        console.print(f"\n[bold]Running: {desc}[/bold]")
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[metrics]}"),
            console=console,
        ) as progress:
            bootstrap_task = progress.add_task(
                f"Bootstrap ({key})",
                total=args.n_bootstrap,
                metrics="",
            )
            try:
                bootstrap_metrics, bootstrap_results = (
                    train_and_evaluate_with_bootstrap(
                        args.model,
                        use_vol,
                        use_age,
                        use_ca125,
                        default_config,
                        args.n_bootstrap,
                        progress,
                        bootstrap_task,
                        test_only_path=args.test_only,
                    )
                )

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name_part = (
                    f"{args.run_name}_{args.model}" if args.run_name else args.model
                )
                pd.DataFrame(bootstrap_metrics).to_csv(
                    args.output_dir
                    / f"METRICS_{args.n_bootstrap}_bootstrap_{name_part}_{key}_{timestamp}.csv",
                    index=False,
                )
                pd.DataFrame(bootstrap_results).to_csv(
                    args.output_dir
                    / f"RESULTS_{args.n_bootstrap}_bootstrap_{name_part}_{key}_{timestamp}.csv",
                    index=False,
                )

                df = pd.DataFrame(bootstrap_metrics)
                for col in ["Test/roc_auc", "Test/f1_score"]:
                    vals = df[col].dropna().values
                    if len(vals) > 0:
                        med = np.median(vals)
                        lo, hi = np.percentile(vals, [2.5, 97.5])
                        console.print(f"  {col}: {med:.3f} [{lo:.3f}-{hi:.3f}]")

                results_summary.append((key, desc, "Success"))
            except Exception as e:
                console.print(f"  [red]Failed: {e}[/red]")
                results_summary.append((key, desc, "Failed"))

    console.print(
        f"\n[bold green]Ablation complete! Results in {args.output_dir}[/bold green]"
    )


if __name__ == "__main__":
    main()
