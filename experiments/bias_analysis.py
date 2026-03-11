"""Model bias analysis across clinical and morphological subgroups.

Loads bootstrap RESULTS CSV (from crs-test) and the corresponding dataset JSON,
computes morphological features from segmentation masks, then tests whether
model performance differs systematically across subgroups.

Clinical features tested:  age, CA-125
Morphological features:    total lesion volume, number of connected components,
                           largest CC diameter, surface-to-volume ratio

Outputs:
    - Rich console table
    - LaTeX table  (<output_dir>/bias_table.tex)
    - Matplotlib figure  (<output_dir>/bias_analysis.png / .pdf)

Usage:
    uv run python experiments/bias_analysis.py \\
        --results outputs/RESULTS_1000_bootstrap_IEO_test_20260101_120000.csv \\
        --data    inputs/json/ieo_test.json \\
        --output_dir outputs/bias
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import stats
from sklearn.metrics import roc_auc_score
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore", category=RuntimeWarning)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette (pastel)
# ---------------------------------------------------------------------------
PALETTE = {
    "low":  "#AEC6CF",   # pastel blue
    "high": "#FFB694",   # pastel peach-orange
    "bg":   "#F8F8F8",
    "grid": "#E8E8E8",
    "text": "#333333",
    "sig":  "#C0392B",   # annotation colour for p-values
}

FEATURE_META = {
    "age":                    ("Age",                     "years",  "Age (years)"),
    "ca125":                  ("CA-125",                  "U/mL",   "CA-125 (U/mL)"),
    "volume_mm3":             ("Lesion volume",           "mm³",    "Lesion volume (mm³)"),
    "n_components":           ("Connected components",    "n",      "No. connected components"),
    "largest_cc_diameter_mm": ("Largest CC diameter",     "mm",     "Largest CC diameter (mm)"),
    "surface_volume_ratio":   ("Surface-to-volume ratio", "mm⁻¹",  "Surface-to-volume ratio (mm⁻¹)"),
}

# ---------------------------------------------------------------------------
# Morphological feature computation
# ---------------------------------------------------------------------------

DISEASE_SITES_MAP = {
    "om":               [1],
    "pod_om":           [1, 9],
    "abdominal_lesions":[1, 2, 3, 5, 6, 7],
    "all":              [1, 2, 3, 5, 6, 7, 9, 13, 14, 15, 17],
}


def compute_morphological_features(
    seg_path: str,
    disease_sites: list[int] | None = None,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, float]:
    """Compute morphological features from a segmentation NIfTI mask.

    Returns dict with keys:
        volume_mm3, n_components, largest_cc_diameter_mm, surface_volume_ratio
    Returns all-zero dict on failure.
    """
    zero = {
        "volume_mm3": 0.0,
        "n_components": 0.0,
        "largest_cc_diameter_mm": 0.0,
        "surface_volume_ratio": 0.0,
    }
    if disease_sites is None:
        disease_sites = [1]

    try:
        img = nib.load(seg_path)
        data = img.get_fdata(dtype=np.float32)
        affine = img.affine
        current_spacing = np.abs(np.diag(affine)[:3])

        if not np.allclose(current_spacing, target_spacing, atol=0.01):
            zoom_factors = current_spacing / np.array(target_spacing)
            data = ndi.zoom(data, zoom_factors, order=0, mode="nearest")

        mask = np.isin(data.astype(np.int32), disease_sites)

        volume = float(np.sum(mask))
        if volume == 0:
            return zero

        # Connected components
        labeled, n_labels = ndi.label(mask)

        # Largest CC diameter (max bounding-box extent in mm at 1mm/vox)
        sizes = np.bincount(labeled.ravel())[1:]
        largest_label = int(np.argmax(sizes)) + 1
        slices = ndi.find_objects(labeled)[largest_label - 1]
        largest_cc_diameter = float(max(s.stop - s.start for s in slices))

        # Surface-to-volume ratio
        eroded = ndi.binary_erosion(mask)
        surface = np.sum(mask & ~eroded)
        svr = float(surface) / volume

        return {
            "volume_mm3": volume,
            "n_components": float(n_labels),
            "largest_cc_diameter_mm": largest_cc_diameter,
            "surface_volume_ratio": svr,
        }

    except Exception as exc:
        logger.warning(f"Morphological features failed for {seg_path}: {exc}")
        return zero


def load_or_compute_morph_features(
    patients: list[dict],
    disease_sites: list[int],
    cache_path: Path,
) -> dict[str, dict[str, float]]:
    """Load cached morphological features or compute them from scratch."""
    if cache_path.exists():
        console.print(f"[cyan]Loading cached morphological features from {cache_path}[/cyan]")
        with open(cache_path) as f:
            return json.load(f)

    console.print("[yellow]Computing morphological features (this may take a few minutes)...[/yellow]")
    features: dict[str, dict[str, float]] = {}

    for i, p in enumerate(patients):
        rid = str(p.get("Record ID", i))
        seg_path = str(p.get("segmentation_path", ""))
        if seg_path in ("-1", "-1.0", "") or not Path(seg_path).exists():
            logger.warning(f"Skipping {rid}: missing segmentation")
            features[rid] = {
                "volume_mm3": p.get("tumor_volume_mm3_sites_1", 0.0),
                "n_components": 0.0,
                "largest_cc_diameter_mm": 0.0,
                "surface_volume_ratio": 0.0,
            }
            continue

        feat = compute_morphological_features(seg_path, disease_sites)
        # Use cached volume from JSON if already there, for consistency
        cached_vol = p.get("tumor_volume_mm3_sites_1")
        if cached_vol is not None and cached_vol > 0:
            feat["volume_mm3"] = float(cached_vol)
        features[rid] = feat

        if (i + 1) % 5 == 0:
            console.print(f"  [{i+1}/{len(patients)}] done")

    with open(cache_path, "w") as f:
        json.dump(features, f, indent=2)
    console.print(f"[green]Saved morphological feature cache to {cache_path}[/green]")
    return features


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def bootstrap_auc(
    probs: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap AUC: returns (median, lo_95, hi_95)."""
    if rng is None:
        rng = np.random.default_rng(42)
    if len(np.unique(labels)) < 2:
        return (float("nan"),) * 3
    aucs = []
    n = len(probs)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], probs[idx]))
    if not aucs:
        return (float("nan"),) * 3
    return float(np.median(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def pvalue_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def analyse_feature(
    df: pd.DataFrame,
    feature: str,
    n_boot: int = 1000,
) -> dict:
    """Run subgroup bias analysis for one feature.

    Returns a dict with keys: threshold, low, high (each with sub-dict of stats).
    """
    vals = df[feature].dropna()
    threshold = float(np.median(vals))

    low_mask  = df[feature] <= threshold
    high_mask = df[feature] >  threshold

    rng = np.random.default_rng(42)
    result = {"feature": feature, "threshold": threshold, "groups": {}}

    for label, mask in [("low", low_mask), ("high", high_mask)]:
        sub = df[mask].dropna(subset=["predicted_prob", "ground_truth"])
        probs  = sub["predicted_prob"].values
        labels = sub["ground_truth"].values
        preds  = sub["predicted_class"].values

        n = len(sub)
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        correct = int((preds == labels).sum())
        errors  = n - correct

        auc_med, auc_lo, auc_hi = bootstrap_auc(probs, labels, n_boot=n_boot, rng=rng)

        result["groups"][label] = {
            "n": n, "n_pos": n_pos, "n_neg": n_neg,
            "correct": correct, "errors": errors,
            "auc": auc_med, "auc_lo": auc_lo, "auc_hi": auc_hi,
            "probs": probs.tolist(),
            "labels": labels.tolist(),
        }

    # Between-group statistical tests
    low_probs  = np.array(result["groups"]["low"]["probs"])
    high_probs = np.array(result["groups"]["high"]["probs"])

    if len(low_probs) > 1 and len(high_probs) > 1:
        mwu_stat, mwu_p = stats.mannwhitneyu(low_probs, high_probs, alternative="two-sided")
    else:
        mwu_stat, mwu_p = float("nan"), float("nan")

    # Fisher's exact: [[correct_low, errors_low], [correct_high, errors_high]]
    c_low  = result["groups"]["low"]["correct"]
    e_low  = result["groups"]["low"]["errors"]
    c_high = result["groups"]["high"]["correct"]
    e_high = result["groups"]["high"]["errors"]
    if (c_low + e_low) > 0 and (c_high + e_high) > 0:
        _, fisher_p = stats.fisher_exact([[c_low, e_low], [c_high, e_high]])
    else:
        fisher_p = float("nan")

    result["mwu_p"] = mwu_p
    result["fisher_p"] = fisher_p
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_rich_table(analyses: list[dict]):
    table = Table(
        title="Subgroup Bias Analysis",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Feature", style="bold")
    table.add_column("Group", justify="center")
    table.add_column("n", justify="right")
    table.add_column("AUC [95% CI]", justify="center")
    table.add_column("Correct / n", justify="center")
    table.add_column("MWU p", justify="center")
    table.add_column("Fisher p", justify="center")

    for res in analyses:
        feat = FEATURE_META[res["feature"]][0]
        thr  = res["threshold"]
        mwu_p    = res["mwu_p"]
        fisher_p = res["fisher_p"]

        fmt_p = lambda p: (
            f"[bold red]{p:.4f}[/bold red]" if not np.isnan(p) and p < 0.05
            else (f"{p:.4f}" if not np.isnan(p) else "—")
        )

        for i, grp_label in enumerate(["low", "high"]):
            g = res["groups"][grp_label]
            thr_str = f"≤{thr:.1f}" if grp_label == "low" else f">{thr:.1f}"
            auc_str = (
                f"{g['auc']:.3f} [{g['auc_lo']:.3f}–{g['auc_hi']:.3f}]"
                if not np.isnan(g["auc"]) else "—"
            )
            corr_str = f"{g['correct']}/{g['n']}"

            if i == 0:
                table.add_row(
                    f"{feat} ({thr_str})",
                    grp_label,
                    str(g["n"]),
                    auc_str,
                    corr_str,
                    fmt_p(mwu_p),
                    fmt_p(fisher_p),
                )
            else:
                table.add_row(
                    f"  ({thr_str})",
                    grp_label,
                    str(g["n"]),
                    auc_str,
                    corr_str,
                    "",
                    "",
                )

    console.print(table)


def write_latex_table(analyses: list[dict], output_path: Path):
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{Subgroup bias analysis. AUC reported as median [95\% bootstrap CI]. "
        r"MWU: Mann--Whitney U test on predicted probabilities; Fisher: Fisher's exact test on error rate. "
        r"$p < 0.05$ in bold.}",
        r"    \label{tab:bias}",
        r"    \begin{tabular}{ll r c c r r}",
        r"        \toprule",
        r"        Feature & Group & $n$ & AUC [95\% CI] & Correct/$n$ & $p_\text{MWU}$ & $p_\text{Fisher}$ \\",
        r"        \midrule",
    ]

    def fmt_p_latex(p):
        if np.isnan(p):
            return "—"
        s = f"{p:.4f}"
        return r"\textbf{" + s + "}" if p < 0.05 else s

    clinical = [r for r in analyses if r["feature"] in ("age", "ca125")]
    morphological = [r for r in analyses if r["feature"] not in ("age", "ca125")]

    for section, section_analyses in [("Clinical features", clinical), ("Morphological features", morphological)]:
        lines.append(r"        \multicolumn{7}{l}{\textit{" + section + r"}} \\")
        for res in section_analyses:
            feat_name, unit, _ = FEATURE_META[res["feature"]]
            thr = res["threshold"]
            mwu_p    = fmt_p_latex(res["mwu_p"])
            fisher_p = fmt_p_latex(res["fisher_p"])

            for i, grp_label in enumerate(["low", "high"]):
                g = res["groups"][grp_label]
                thr_str = rf"$\leq${thr:.1f}" if grp_label == "low" else rf"$>${thr:.1f}"
                auc_str = (
                    rf"{g['auc']:.3f} [{g['auc_lo']:.3f}--{g['auc_hi']:.3f}]"
                    if not np.isnan(g["auc"]) else "—"
                )
                corr_str = rf"{g['correct']}/{g['n']}"

                if i == 0:
                    lines.append(
                        rf"        {feat_name} ({thr_str}) & low & {g['n']} & {auc_str} & {corr_str} & {mwu_p} & {fisher_p} \\"
                    )
                else:
                    lines.append(
                        rf"         & high & {g['n']} & {auc_str} & {corr_str} & & \\"
                    )

        lines.append(r"        \addlinespace")

    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ])

    latex = "\n".join(lines)
    output_path.write_text(latex)
    console.print(f"\n[green]LaTeX table saved → {output_path}[/green]")
    console.print(f"\n[bold]LaTeX:[/bold]\n{latex}\n")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(analyses: list[dict], output_dir: Path):
    n_features = len(analyses)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4.5 * n_rows),
        facecolor="white",
    )
    axes = np.array(axes).flatten()

    rng = np.random.default_rng(0)

    for ax_idx, res in enumerate(analyses):
        ax = axes[ax_idx]
        ax.set_facecolor(PALETTE["bg"])
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        _, _, feat_label = FEATURE_META[res["feature"]]
        thr = res["threshold"]

        low_probs  = np.array(res["groups"]["low"]["probs"])
        high_probs = np.array(res["groups"]["high"]["probs"])
        low_labels  = np.array(res["groups"]["low"]["labels"])
        high_labels = np.array(res["groups"]["high"]["labels"])

        positions = [1, 2]
        data_arrays = [low_probs, high_probs]
        colors      = [PALETTE["low"], PALETTE["high"]]
        edge_colors = ["#5B8FA8", "#C87941"]

        parts = ax.violinplot(
            data_arrays,
            positions=positions,
            widths=0.6,
            showmedians=True,
            showextrema=False,
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor(edge_colors[i])
            pc.set_alpha(0.75)
            pc.set_linewidth(1.2)
        parts["cmedians"].set_color(PALETTE["text"])
        parts["cmedians"].set_linewidth(2)

        # Jittered scatter, colour by ground truth
        for pos, probs, labs in [(1, low_probs, low_labels), (2, high_probs, high_labels)]:
            jitter = rng.uniform(-0.12, 0.12, size=len(probs))
            for cls, marker, ms in [(0, "o", 6), (1, "^", 6)]:
                sel = labs == cls
                ax.scatter(
                    pos + jitter[sel],
                    probs[sel],
                    s=ms ** 2,
                    marker=marker,
                    color=PALETTE["crs0"] if cls == 0 else PALETTE["crs1"],
                    edgecolors=PALETTE["text"],
                    linewidths=0.4,
                    zorder=3,
                    alpha=0.85,
                )

        # AUC annotations inside each violin
        for pos, grp_label in [(1, "low"), (2, "high")]:
            g = res["groups"][grp_label]
            auc = g["auc"]
            auc_str = f"AUC={auc:.2f}" if not np.isnan(auc) else "AUC=—"
            ax.text(
                pos, 0.02, f"n={g['n']}\n{auc_str}",
                ha="center", va="bottom",
                fontsize=7.5, color=PALETTE["text"],
                transform=ax.get_xaxis_transform(),
            )

        # p-value annotation (MWU)
        mwu_p = res["mwu_p"]
        stars = pvalue_stars(mwu_p) if not np.isnan(mwu_p) else ""
        p_str = f"p={mwu_p:.3f} {stars}" if not np.isnan(mwu_p) else ""
        y_top = max(
            (max(low_probs) if len(low_probs) else 0),
            (max(high_probs) if len(high_probs) else 0),
        )
        bracket_y = min(y_top + 0.08, 1.02)
        ax.annotate(
            "", xy=(2, bracket_y), xytext=(1, bracket_y),
            arrowprops=dict(arrowstyle="-", color=PALETTE["sig"], lw=1.2),
        )
        ax.text(
            1.5, bracket_y + 0.02, p_str,
            ha="center", va="bottom",
            fontsize=8, color=PALETTE["sig"], fontweight="bold",
        )

        ax.set_xlim(0.4, 2.6)
        ax.set_ylim(-0.05, 1.15)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(
            [f"Low\n(≤{thr:.1f})", f"High\n(>{thr:.1f})"],
            fontsize=9,
        )
        ax.set_ylabel("Predicted probability", fontsize=9)
        ax.set_title(feat_label, fontsize=10, fontweight="bold", pad=8)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # Turn off unused axes
    for ax in axes[n_features:]:
        ax.set_visible(False)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=PALETTE["low"],  edgecolor="#5B8FA8", label="Low group"),
        mpatches.Patch(facecolor=PALETTE["high"], edgecolor="#C87941", label="High group"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["crs0"],
                   markeredgecolor=PALETTE["text"], markersize=7, label="CRS 0 (responder)"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=PALETTE["crs1"],
                   markeredgecolor=PALETTE["text"], markersize=7, label="CRS 1/2 (non-responder)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Model Predictions Across Clinical and Morphological Subgroups",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.tight_layout()

    for ext in ("png", "pdf"):
        path = output_dir / f"bias_analysis.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        console.print(f"[green]Figure saved → {path}[/green]")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Model bias analysis across subgroups")
    parser.add_argument(
        "--results", required=True,
        help="Path to RESULTS_*_bootstrap_*.csv produced by crs-test",
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to the dataset JSON used during evaluation (e.g. inputs/json/ieo_test.json)",
    )
    parser.add_argument(
        "--disease_sites", default="om",
        choices=list(DISEASE_SITES_MAP.keys()),
        help="Disease sites used for morphological feature computation (default: om)",
    )
    parser.add_argument(
        "--output_dir", default="outputs/bias",
        help="Directory to write outputs (default: outputs/bias)",
    )
    parser.add_argument(
        "--n_boot", type=int, default=1000,
        help="Bootstrap iterations for per-group AUC CI (default: 1000)",
    )
    parser.add_argument(
        "--no_morph", action="store_true",
        help="Skip morphological feature computation (clinical features only)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load RESULTS CSV and aggregate per patient
    # ------------------------------------------------------------------
    console.print(f"\n[bold blue]Loading results: {args.results}[/bold blue]")
    results_df = pd.read_csv(args.results, index_col=0)
    console.print(f"  {len(results_df)} rows, {results_df['bootstrap_iter'].nunique()} bootstrap iterations")

    # Aggregate across bootstrap iterations → one row per patient
    agg = (
        results_df.groupby("record_id")
        .agg(
            age=("age", "first"),
            ca125=("ca125", "first"),
            ground_truth=("ground_truth", "first"),
            original_crs=("original_crs", "first"),
            predicted_prob=("predicted_prob", "mean"),
            predicted_class=("predicted_class", lambda x: int(round(x.mean()))),
        )
        .reset_index()
    )
    console.print(f"  Aggregated to {len(agg)} unique patients")

    # ------------------------------------------------------------------
    # 2. Load JSON and merge morphological features
    # ------------------------------------------------------------------
    console.print(f"\n[bold blue]Loading dataset JSON: {args.data}[/bold blue]")
    with open(args.data) as f:
        json_data = json.load(f)
    patients = json_data["patients"]

    disease_sites = DISEASE_SITES_MAP[args.disease_sites]
    cache_path = Path(args.data).with_suffix("").with_suffix("") / "_morph_cache.json"
    # Simpler: put cache next to the JSON
    cache_path = Path(args.data).parent / (Path(args.data).stem + "_morph_cache.json")

    if not args.no_morph:
        morph_features = load_or_compute_morph_features(patients, disease_sites, cache_path)
        morph_df = pd.DataFrame.from_dict(morph_features, orient="index").reset_index()
        morph_df.columns = ["record_id"] + list(morph_df.columns[1:])
        agg = agg.merge(morph_df, on="record_id", how="left")
    else:
        console.print("[yellow]Skipping morphological features (--no_morph)[/yellow]")

    # ------------------------------------------------------------------
    # 3. Run analysis for each feature
    # ------------------------------------------------------------------
    clinical_features = ["age", "ca125"]
    morph_feat_cols   = ["volume_mm3", "n_components", "largest_cc_diameter_mm", "surface_volume_ratio"]

    features_to_analyse = clinical_features.copy()
    if not args.no_morph:
        features_to_analyse += morph_feat_cols

    # Clean up: replace -1 sentinel values with NaN
    for col in ["age", "ca125"]:
        if col in agg.columns:
            agg.loc[agg[col] <= 0, col] = np.nan

    console.print(f"\n[bold blue]Running subgroup analyses ({args.n_boot} bootstrap iterations each)...[/bold blue]")
    analyses = []
    for feat in features_to_analyse:
        if feat not in agg.columns:
            logger.warning(f"Feature {feat} not found in data, skipping")
            continue
        n_valid = agg[feat].notna().sum()
        if n_valid < 4:
            logger.warning(f"Feature {feat}: only {n_valid} valid values, skipping")
            continue
        console.print(f"  Analysing: {feat} (n valid = {n_valid})")
        res = analyse_feature(agg, feat, n_boot=args.n_boot)
        analyses.append(res)

    if not analyses:
        console.print("[red]No features could be analysed.[/red]")
        return

    # ------------------------------------------------------------------
    # 4. Print Rich table
    # ------------------------------------------------------------------
    console.print()
    print_rich_table(analyses)

    # ------------------------------------------------------------------
    # 5. LaTeX table
    # ------------------------------------------------------------------
    write_latex_table(analyses, output_dir / "bias_table.tex")

    # ------------------------------------------------------------------
    # 6. Figure
    # ------------------------------------------------------------------
    make_figure(analyses, output_dir)

    console.print("\n[bold green]Bias analysis complete![/bold green]")
    console.print(f"Outputs written to: {output_dir}/")


if __name__ == "__main__":
    main()
