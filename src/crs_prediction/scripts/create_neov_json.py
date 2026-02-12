"""Generate a ds.json-compatible file for the NEOV dataset.

Reads the new per-case directory structure under inputs/neov/ and produces
a JSON file compatible with PCSDataset / MLPDataset.

New data layout (inputs/neov/ is a symlink to the dataset root):
    inputs/neov/
    ├── metadata.json          # root-level metadata for all cases
    ├── case_XXXX/
    │   ├── scans/case_XXXX_t0.nii.gz   # pre-treatment CT
    │   ├── masks/case_XXXX_t0.nii.gz   # pre-treatment segmentation
    │   └── metadata.json               # per-case clinical metadata
    └── ...

CRS mapping (from treatment.chemotherapy_response_score):
    1 -> 0  (CRS1, poor response)
    2 -> 1  (CRS2)
    3 -> 2  (CRS3, good response)
    None / no surgery -> skipped

Paths in the output JSON point to inputs/neov/... so that only the
symlink needs to be reconfigured when moving between machines.

Usage:
    crs-create-neov-json --data_root inputs/neov --output neov_ds.json
"""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# CRS integer values in the new metadata -> our label scheme
CRS_INT_MAP = {1: 0, 2: 1, 3: 2}


def main():
    parser = argparse.ArgumentParser(
        description="Generate ds.json for the NEOV dataset from per-case directory structure."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="inputs/neov",
        help="Root directory of NEOV data (default: inputs/neov)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="neov_ds.json",
        help="Output JSON file path (default: neov_ds.json)",
    )
    parser.add_argument(
        "--timepoint",
        type=str,
        default="t0",
        help="Timepoint to use: t0 (pre-treatment) or t1 (post-treatment). Default: t0",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    tp = args.timepoint

    # Discover case directories (sorted for reproducibility)
    case_dirs = sorted(
        [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("case_")]
    )
    logger.info(f"Found {len(case_dirs)} case directories in {data_root}")

    patients = []
    stats = {
        "total": 0,
        "skipped_no_crs": 0,
        "skipped_no_scan": 0,
        "skipped_no_mask": 0,
        "crs_counts": {0: 0, 1: 0, 2: 0},
    }

    for case_dir in case_dirs:
        case_id = case_dir.name

        # Load per-case metadata
        meta_path = case_dir / "metadata.json"
        if not meta_path.exists():
            logger.warning(f"Skipping {case_id}: no metadata.json")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        # Extract CRS
        crs_raw = meta.get("treatment", {}).get("chemotherapy_response_score")
        if crs_raw is None or crs_raw not in CRS_INT_MAP:
            logger.debug(f"Skipping {case_id}: no valid CRS (got {crs_raw})")
            stats["skipped_no_crs"] += 1
            continue

        # Check scan and mask files exist
        scan_path = case_dir / "scans" / f"{case_id}_{tp}.nii.gz"
        mask_path = case_dir / "masks" / f"{case_id}_{tp}.nii.gz"

        if not scan_path.exists():
            logger.warning(f"Skipping {case_id}: scan not found at {scan_path}")
            stats["skipped_no_scan"] += 1
            continue

        if not mask_path.exists():
            logger.warning(f"Skipping {case_id}: mask not found at {mask_path}")
            stats["skipped_no_mask"] += 1
            continue

        crs_label = CRS_INT_MAP[crs_raw]
        stats["crs_counts"][crs_label] += 1

        # Extract clinical features
        clinical = meta.get("clinical", {})
        ca125_data = meta.get("ca125", {})

        age = clinical.get("age_at_diagnosis", -1)
        age = float(age) if age is not None else -1.0

        ca125 = ca125_data.get("ca125_level_at_diagnosis", -1)
        ca125 = float(ca125) if ca125 is not None else -1.0

        # CRS notes string
        crs_notes_map = {1: "CRS1", 2: "CRS2", 3: "CRS3"}
        crs_notes = crs_notes_map.get(crs_raw, str(crs_raw))

        # Record ID from dataset metadata or case number
        dataset_meta = meta.get("dataset", {})
        record_id = str(dataset_meta.get("original_id", case_id.replace("case_", "")))

        # Build patient record matching ds.json schema
        patient_record = {
            "Record ID": record_id,
            "Age at diagnosis": age,
            "Charlson Comorbidity Index": -1,
            "BMI (kg/m^2)": -1,
            "CA125 (U/mL)": ca125,
            "HE4 (pmol/l)": -1,
            "Type of surgery": "NACT",
            "CRS": crs_label,
            "CRS: notes": crs_notes,
            "Ovarian/Peritoneum/Fallopian Tube Cancer FIGO Staging": -1,
            "year": -1,
            "image_path": str(scan_path),
            "segmentation_path": str(mask_path),
        }

        patients.append(patient_record)
        stats["total"] += 1

    # Write output
    output_data = {"patients": patients}
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    logger.info(f"Wrote {len(patients)} patients to {output_path}")
    logger.info(f"Stats: {stats}")


if __name__ == "__main__":
    main()
