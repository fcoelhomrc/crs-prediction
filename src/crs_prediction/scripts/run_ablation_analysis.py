"""Ablation analysis: generates comparison tables from bootstrap results.

Reads METRICS CSVs produced by sklearn or MLP ablation runs, computes
median + 95% CI, and outputs both Rich console tables and LaTeX.

Usage:
    crs-ablation-analysis --model logistic
    crs-ablation-analysis --model mlp --ablation-dir outputs/ablation
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

ABLATION_CONFIGS = [
    ("noclinical", True, False, False, False),
    ("age", True, False, True, False),
    ("ca125", True, False, False, True),
    ("full", True, False, True, True),
    ("age_only", False, False, True, False),
    ("ca125_only", False, False, False, True),
    ("volume_only", False, True, False, False),
    ("age_ca125", False, False, True, True),
    ("age_volume", False, True, True, False),
    ("ca125_volume", False, True, False, True),
    ("volume_age_ca125", False, True, True, True),
]

TICK = r"\ding{51}"
CROSS = r"\ding{55}"


def compute_ci(data: np.ndarray, ci: float = 0.95) -> tuple[float, float, float]:
    lower_p = (1 - ci) / 2 * 100
    upper_p = (1 + ci) / 2 * 100
    return np.median(data), np.percentile(data, lower_p), np.percentile(data, upper_p)


def load_ablation_metrics(
    ablation_dir: Path,
    model_filter: str | None = None,
) -> dict[str, pd.DataFrame]:
    metrics = {}
    for key, *_ in ABLATION_CONFIGS:
        if model_filter:
            patterns = [
                f"METRICS_*_bootstrap_{model_filter}_{key}_*.csv",
                f"METRICS_{model_filter}_{key}_*.csv",
            ]
        else:
            patterns = [
                f"METRICS_*_bootstrap_*_{key}_*.csv",
                f"METRICS_*_{key}_*.csv",
            ]

        matches = []
        for pattern in patterns:
            matches.extend(ablation_dir.glob(pattern))

        if not matches:
            continue

        metrics_file = sorted(matches)[-1]
        metrics[key] = pd.read_csv(metrics_file)
        console.print(f"[green]Loaded {key}: {metrics_file.name}[/green]")

    return metrics


def extract_metrics_summary(df: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    return {
        "acc": compute_ci(df["Test/accuracy"].values),
        "prec": compute_ci(df["Test/precision"].values),
        "rec": compute_ci(df["Test/recall"].values),
        "f1": compute_ci(df["Test/f1_score"].values),
        "auc": compute_ci(df["Test/roc_auc"].values),
        "pr_auc": compute_ci(df["Test/pr_auc"].values),
    }


def generate_rich_table(metrics: dict[str, pd.DataFrame]):
    metric_keys = ["acc", "prec", "rec", "f1", "auc", "pr_auc"]
    metric_names = {
        "acc": "Accuracy",
        "prec": "Precision",
        "rec": "Recall",
        "f1": "F1",
        "auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
    }

    table = Table(title="Ablation Study Results (Median [95% CI])", show_header=True)
    table.add_column("CT", justify="center", style="cyan")
    table.add_column("Volume", justify="center", style="cyan")
    table.add_column("Age", justify="center", style="cyan")
    table.add_column("CA125", justify="center", style="cyan")
    for key in metric_keys:
        table.add_column(metric_names[key], justify="center")

    summaries = {}
    for key, _, _, _, _ in ABLATION_CONFIGS:
        if key in metrics:
            summaries[key] = extract_metrics_summary(metrics[key])

    if not summaries:
        console.print("[yellow]No metrics to display[/yellow]")
        return

    best_values = {}
    for mk in metric_keys:
        best_values[mk] = max(s[mk][0] for s in summaries.values())

    for key, ct, volume, age, ca125 in ABLATION_CONFIGS:
        if key not in summaries:
            continue
        summary = summaries[key]
        row = [
            "[green]\u2713[/green]" if ct else "[red]\u2717[/red]",
            "[green]\u2713[/green]" if volume else "[red]\u2717[/red]",
            "[green]\u2713[/green]" if age else "[red]\u2717[/red]",
            "[green]\u2713[/green]" if ca125 else "[red]\u2717[/red]",
        ]
        for mk in metric_keys:
            med, lo, hi = summary[mk]
            is_best = abs(med - best_values[mk]) < 1e-6
            formatted = f"{med:.3f} [{lo:.3f}-{hi:.3f}]"
            row.append(
                f"[bold green]{formatted}[/bold green]" if is_best else formatted
            )
        table.add_row(*row)

    console.print(table)


def generate_latex_table(metrics: dict[str, pd.DataFrame], output_path: Path):
    summaries = {}
    for key, _, _, _, _ in ABLATION_CONFIGS:
        if key in metrics:
            summaries[key] = extract_metrics_summary(metrics[key])

    if not summaries:
        return

    best_auc = max(s["auc"][0] for s in summaries.values())
    best_f1 = max(s["f1"][0] for s in summaries.values())

    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{Ablation study. Best values in bold.}",
        r"    \label{tab:ablation}",
        r"    \begin{tabular}{ccc|cc}",
        r"        \toprule",
        r"        Volume & Age & CA125 & AUC & F1 \\",
        r"        \midrule",
    ]

    def fmt(med, lo, hi, is_best):
        s = f"{med:.3f} ({lo:.3f}--{hi:.3f})"
        return r"\textbf{" + s + "}" if is_best else s

    for key, ct, volume, age, ca125 in ABLATION_CONFIGS:
        if key not in summaries:
            continue
        s = summaries[key]
        vol_sym = TICK if volume else CROSS
        age_sym = TICK if age else CROSS
        ca_sym = TICK if ca125 else CROSS
        auc_s = fmt(*s["auc"], abs(s["auc"][0] - best_auc) < 1e-6)
        f1_s = fmt(*s["f1"], abs(s["f1"][0] - best_f1) < 1e-6)
        lines.append(
            f"        {vol_sym} & {age_sym} & {ca_sym} & {auc_s} & {f1_s} \\\\"
        )

    lines.extend([r"        \bottomrule", r"    \end{tabular}", r"\end{table}"])
    latex = "\n".join(lines)
    output_path.write_text(latex)
    console.print(f"\n[green]LaTeX table saved to: {output_path}[/green]")
    console.print(f"\n[bold]LaTeX Code:[/bold]\n{latex}")


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument("--model", type=str, default=None, help="Filter by model type")
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=Path("outputs/ablation"),
    )
    args = parser.parse_args()

    if not args.ablation_dir.exists():
        console.print(f"[red]Directory not found: {args.ablation_dir}[/red]")
        return

    model_str = f" (model={args.model})" if args.model else ""
    console.print(f"[bold blue]Loading ablation metrics{model_str}...[/bold blue]")
    metrics = load_ablation_metrics(args.ablation_dir, model_filter=args.model)

    if not metrics:
        console.print("[red]No metrics files found[/red]")
        return

    generate_rich_table(metrics)
    generate_latex_table(metrics, args.ablation_dir / "ablation_table.tex")
    console.print("\n[bold green]Analysis complete![/bold green]")


if __name__ == "__main__":
    main()
