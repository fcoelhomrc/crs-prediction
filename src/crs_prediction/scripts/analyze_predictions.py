"""Analyze predictions by confusion matrix category.

Merges model predictions with cohort clinical data and compares
clinical features (age, CA125) across TP/TN/FP/FN categories.

Usage:
    crs-analyze predictions.csv cohort_ds.json
"""

import argparse
import json
import re

import numpy as np
import pandas as pd
from rich import print
from rich.table import Table
from scipy import stats


def load_cohort(path: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data["patients"])
    for col in ["Age at diagnosis", "CA125 (U/mL)", "CRS"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == -1, col] = np.nan
    return df


def classify_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Add confusion matrix category column."""
    conditions = [
        (df["targets"] == 1) & (df["preds"] == 1),
        (df["targets"] == 0) & (df["preds"] == 0),
        (df["targets"] == 1) & (df["preds"] == 0),
        (df["targets"] == 0) & (df["preds"] == 1),
    ]
    choices = ["TP", "TN", "FN", "FP"]
    df["category"] = np.select(conditions, choices, default="")
    return df


def summarize_by_category(merged: pd.DataFrame):
    """Print clinical feature summary by confusion matrix category."""
    table = Table(title="Clinical Features by Prediction Category")
    table.add_column("Category")
    table.add_column("n")
    table.add_column("Age median [IQR]")
    table.add_column("CA125 median [IQR]")

    for cat in ["TP", "TN", "FP", "FN"]:
        subset = merged[merged["category"] == cat]
        n = len(subset)
        if n == 0:
            table.add_row(cat, "0", "-", "-")
            continue

        age = subset["Age at diagnosis"].dropna()
        ca125 = subset["CA125 (U/mL)"].dropna()

        age_str = (
            f"{age.median():.1f} [{age.quantile(0.25):.1f}--{age.quantile(0.75):.1f}]"
            if len(age) > 0
            else "-"
        )
        ca125_str = (
            f"{ca125.median():.0f} [{ca125.quantile(0.25):.0f}--{ca125.quantile(0.75):.0f}]"
            if len(ca125) > 0
            else "-"
        )

        table.add_row(cat, str(n), age_str, ca125_str)

    print(table)


def compare_categories(merged: pd.DataFrame):
    """Statistical comparison between categories."""
    print("\n[bold]Statistical Comparisons[/bold]")

    comparisons = [
        ("TP", "FN", "TP vs FN (why were some positives missed?)"),
        ("TN", "FP", "TN vs FP (why were some negatives misclassified?)"),
    ]

    for cat1, cat2, title in comparisons:
        g1 = merged[merged["category"] == cat1]
        g2 = merged[merged["category"] == cat2]
        if len(g1) == 0 or len(g2) == 0:
            continue
        print(f"\n[underline]{title}[/underline]")
        for col in ["Age at diagnosis", "CA125 (U/mL)"]:
            v1 = g1[col].dropna()
            v2 = g2[col].dropna()
            if len(v1) > 1 and len(v2) > 1:
                _, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
                print(
                    f"  {col}: {cat1} median={v1.median():.1f}, "
                    f"{cat2} median={v2.median():.1f}, p={p:.4f}"
                )

    # Correct vs incorrect
    correct = merged[merged["category"].isin(["TP", "TN"])]
    incorrect = merged[merged["category"].isin(["FP", "FN"])]
    if len(correct) > 0 and len(incorrect) > 0:
        print("\n[underline]Correct vs Incorrect predictions[/underline]")
        for col in ["Age at diagnosis", "CA125 (U/mL)"]:
            v1 = correct[col].dropna()
            v2 = incorrect[col].dropna()
            if len(v1) > 1 and len(v2) > 1:
                _, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
                print(
                    f"  {col}: Correct median={v1.median():.1f}, "
                    f"Incorrect median={v2.median():.1f}, p={p:.4f}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze predictions by confusion matrix category"
    )
    parser.add_argument("predictions", type=str, help="Path to predictions CSV")
    parser.add_argument("cohort", type=str, help="Path to cohort ds.json")
    args = parser.parse_args()

    preds = pd.read_csv(args.predictions, index_col=0)
    preds = classify_predictions(preds)

    cohort = load_cohort(args.cohort)

    if len(preds) != len(cohort):
        print(
            f"[yellow]Warning: predictions ({len(preds)}) and cohort ({len(cohort)}) "
            f"have different sizes[/yellow]"
        )

    merged = pd.concat(
        [preds.reset_index(drop=True), cohort.reset_index(drop=True)], axis=1
    )

    print(f"\n[bold]Prediction Analysis (n={len(merged)})[/bold]\n")
    print("[bold]Confusion Matrix:[/bold]")
    for cat in ["TP", "TN", "FP", "FN"]:
        n = (merged["category"] == cat).sum()
        print(f"  {cat}: {n}")
    print()

    summarize_by_category(merged)
    compare_categories(merged)


if __name__ == "__main__":
    main()
