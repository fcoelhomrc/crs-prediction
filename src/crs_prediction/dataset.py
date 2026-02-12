"""Dataset classes for CRS prediction.

Contains:
- PCSDataset: loads volumetric CT/segmentation data from JSON for ViT training
- MLPDataset: loads clinical features (tumor volume, age, CA125) for MLP training
- compute_tumor_volume: computes tumor volume from segmentation NIfTI files
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Disease site label mappings for segmentation masks
DISEASE_SITES = {
    "om": [1],
    "pod_om": [1, 9],
    "abdominal_lesions": [1, 2, 3, 5, 6, 7],
    "lymph_nodes": [13, 14, 15, 17],
    "all": [1, 2, 3, 5, 6, 7, 9, 13, 14, 15, 17],
}


def _filter_disease_sites(volume: np.ndarray, disease_sites: list[int]) -> np.ndarray:
    """Filter segmentation volume to keep only specified disease site labels."""
    mask = np.isin(volume.astype(np.int32), disease_sites)
    return np.where(mask, volume, 0.0).astype(np.float32)


def compute_tumor_volume(
    seg_path: str,
    disease_sites: list[int] | None = None,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Compute tumor volume from a segmentation mask in mm^3.

    Loads the NIfTI segmentation, resamples to isotropic spacing if needed,
    filters to keep only specified disease sites, and counts voxels.

    Args:
        seg_path: Path to the segmentation NIfTI file.
        disease_sites: List of label indices to include. Defaults to omentum [1].
        target_spacing: Target voxel spacing in mm.

    Returns:
        Volume in mm^3.
    """
    if disease_sites is None:
        disease_sites = [1]

    try:
        nifti_img = nib.load(seg_path)
        data = nifti_img.get_fdata(dtype=np.float32)
        affine = nifti_img.affine

        current_spacing = np.abs(np.diag(affine)[:3])

        if not np.allclose(current_spacing, target_spacing, atol=0.01):
            zoom_factors = current_spacing / np.array(target_spacing)
            data = ndimage.zoom(data, zoom_factors, order=0, mode="nearest")

        mask = np.isin(data.astype(np.int32), disease_sites)
        return float(np.sum(mask))  # Each voxel is 1 mm^3 after resampling

    except Exception as e:
        logger.warning(f"Error computing volume from {seg_path}: {e}")
        return 0.0


def _load_nifti_volume(
    path: str,
    is_segmentation: bool = False,
    disease_sites: list[int] | None = None,
    cohort: str = "IEO",
) -> torch.Tensor:
    """Load a NIfTI volume, resize to (1, 224, 224, 256), and normalize.

    Args:
        path: Path to the .nii.gz file.
        is_segmentation: If True, filter to keep only disease_sites labels.
        disease_sites: Labels to keep when filtering segmentation.
        cohort: 'NEOV' transposes axes 0,1 to match IEO orientation.

    Returns:
        Tensor of shape (1, 224, 224, 256) normalized to [0, 1].
    """
    nifti_img = nib.load(str(path))
    volume = nifti_img.get_fdata(dtype=np.float32)

    if cohort == "NEOV":
        volume = np.transpose(volume, (1, 0, 2))
        volume = np.ascontiguousarray(volume)

    if is_segmentation and disease_sites:
        volume = _filter_disease_sites(volume, disease_sites)

    x = torch.from_numpy(volume).unsqueeze(0)  # (1, H, W, D)

    # Interpolate to (256, 224, 224): permute to (N, C, D, H, W)
    x = x.permute(0, 3, 1, 2).unsqueeze(0)  # (1, 1, D, H, W)
    x = F.interpolate(x, size=(256, 224, 224), mode="nearest")
    x = x.squeeze(0).permute(0, 2, 3, 1)  # (1, 224, 224, 256)

    # Min-max normalize
    x_min = (
        x.min(dim=1, keepdim=True)[0]
        .min(dim=2, keepdim=True)[0]
        .min(dim=3, keepdim=True)[0]
    )
    x_max = (
        x.max(dim=1, keepdim=True)[0]
        .max(dim=2, keepdim=True)[0]
        .max(dim=3, keepdim=True)[0]
    )
    x = (x - x_min) / (x_max - x_min + 1e-6)

    return x


# ---------------------------------------------------------------------------
# PCSDataset - for ViT training with volumetric images
# ---------------------------------------------------------------------------


@dataclass
class PatientData:
    """Stores a single patient's data for ViT training."""

    image: Optional[torch.Tensor] = None
    label: Optional[torch.Tensor] = None
    clinical_features: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    record_id: Optional[str] = None
    age: Optional[float] = None
    ca125: Optional[float] = None
    original_crs: Optional[int] = None
    crs_label: Optional[str] = None
    # Paths for lazy loading (only set when eager=False)
    image_path: Optional[str] = None
    mask_path: Optional[str] = None


class PCSDataset(Dataset):
    """Dataset for CRS prediction from volumetric CT/segmentation data.

    Loads patient data from a JSON file with the standard schema:
    ```json
    {"patients": [{"Record ID": ..., "image_path": ..., "segmentation_path": ..., "CRS": ..., ...}]}
    ```
    """

    def __init__(
        self,
        data_file: str | Path,
        transform: Optional[Any] = None,
        is_training: bool = True,
        excluded_years: list[int] | None = None,
        seg: bool = False,
        seg_as_im: bool = True,
        clinical_fe: bool = False,
        disease_sites: str = "om",
        cohort: str = "IEO",
        eager: bool = False,
    ):
        super().__init__()
        self.transform = transform
        self.is_training = is_training
        self.seg = seg
        self.seg_as_im = seg_as_im
        self.clinical_fe = clinical_fe
        self.disease_sites_labels = DISEASE_SITES.get(
            disease_sites, DISEASE_SITES["om"]
        )
        self.cohort = cohort
        self.eager = eager
        self.stats = {"num_patients": 0, "residual_tumor_counts": {0: 0, 1: 0, 2: 0}}

        if excluded_years is None:
            excluded_years = [2015]
        self.data = self._load_data(data_file, excluded_years)

    def _load_data(
        self, data_file: str | Path, excluded_years: list[int]
    ) -> list[PatientData]:
        data_list = []
        with open(data_file, "r") as f:
            data = json.load(f)

        for patient in data["patients"]:
            if patient["year"] in excluded_years:
                continue

            if self.seg_as_im:
                image_path = str(patient.get("segmentation_path", ""))
            else:
                image_path = str(patient.get("image_path", ""))

            if image_path in ("-1", "-1.0", "") or not image_path.strip():
                logger.warning(
                    f"Skipping patient {patient.get('Record ID', 'unknown')}: invalid image path"
                )
                continue

            self.stats["num_patients"] += 1

            mask_path = None
            if self.seg:
                mp = str(patient.get("segmentation_path", ""))
                if mp not in ("-1", "-1.0", "") and mp.strip():
                    mask_path = mp

            image, mask = None, None
            if self.eager:
                try:
                    image = self._load_image(image_path, is_segmentation=self.seg_as_im)
                    if mask_path:
                        mask = self._load_image(mask_path, is_segmentation=True)
                except Exception as e:
                    logger.error(
                        f"Error loading data for patient {patient.get('Record ID', 'unknown')}: {e}"
                    )
                    continue

            original_label = patient["CRS"]
            self.stats["residual_tumor_counts"][original_label] = (
                self.stats["residual_tumor_counts"].get(original_label, 0) + 1
            )
            label = torch.tensor(original_label, dtype=torch.long)
            label = torch.where(label == 2, torch.tensor(1, dtype=torch.long), label)

            record_id = str(patient.get("Record ID", ""))
            age = float(patient.get("Age at diagnosis", -1.0))
            ca125 = float(patient.get("CA125 (U/mL)", -1.0))
            crs_label = str(patient.get("CRS: notes", ""))

            clinical_features = None
            if self.clinical_fe:
                clinical_features = torch.tensor(
                    [
                        float(patient.get("Age at diagnosis", 0.0)),
                        float(patient.get("CA125 (U/mL)", 0.0)),
                    ],
                    dtype=torch.float32,
                )
                mean = clinical_features.mean(dim=0)
                std = clinical_features.std(dim=0) + 1e-6
                clinical_features = (clinical_features - mean) / std

            data_list.append(
                PatientData(
                    image=image,
                    label=label,
                    clinical_features=clinical_features,
                    mask=mask,
                    record_id=record_id,
                    age=age,
                    ca125=ca125,
                    original_crs=original_label,
                    crs_label=crs_label,
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )
        return data_list

    def _load_image(
        self, image_path: str, is_segmentation: bool = False
    ) -> torch.Tensor:
        if image_path.endswith(".nii.gz") or image_path.endswith(".nii"):
            return _load_nifti_volume(
                image_path,
                is_segmentation=is_segmentation,
                disease_sites=self.disease_sites_labels,
                cohort=self.cohort,
            )
        else:
            return torch.load(image_path, weights_only=False)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            patient = self.data[idx]

            if patient.image is not None:
                image = patient.image.float()
            else:
                image = self._load_image(
                    patient.image_path, is_segmentation=self.seg_as_im
                ).float()

            if self.is_training and self.transform is not None:
                transformed = self.transform(image=image.numpy())
                image = torch.from_numpy(transformed["image"])

            label_one_hot = F.one_hot(patient.label, num_classes=2).float()
            sample = {
                "image": image,
                "label": label_one_hot,
                "record_id": patient.record_id,
                "age": patient.age,
                "ca125": patient.ca125,
                "original_crs": patient.original_crs,
                "crs_label": patient.crs_label,
            }
            if self.seg:
                if patient.mask is not None:
                    sample["mask"] = patient.mask.float()
                elif patient.mask_path:
                    sample["mask"] = self._load_image(
                        patient.mask_path, is_segmentation=True
                    ).float()
            if self.clinical_fe and patient.clinical_features is not None:
                sample["clinical_features"] = patient.clinical_features

            return sample
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            if idx != 0:
                return self.__getitem__(0)
            raise

    def get_stats(self) -> dict:
        return self.stats


# ---------------------------------------------------------------------------
# MLPDataset - for MLP training with clinical features
# ---------------------------------------------------------------------------


@dataclass
class MLPPatientData:
    """Stores a single patient's clinical features for MLP training."""

    features: torch.Tensor
    label: torch.Tensor
    record_id: str
    age: float
    ca125: float
    tumor_volume: float
    original_crs: int
    crs_label: str


class MLPDataset(Dataset):
    """Dataset for MLP classifier using clinical features.

    Loads from the same JSON schema as PCSDataset. Computes tumor volume
    from segmentation masks on first load and caches the result in the JSON.

    Features are configurable: any combination of [tumor_volume, age, CA125].
    """

    def __init__(
        self,
        data_file: str,
        is_training: bool = True,
        excluded_years: list[int] | None = None,
        disease_sites: str = "om",
        normalize: bool = True,
        use_volume: bool = True,
        use_age: bool = True,
        use_ca125: bool = True,
    ):
        self.is_training = is_training
        self.normalize = normalize
        self.use_volume = use_volume
        self.use_age = use_age
        self.use_ca125 = use_ca125

        self.input_dim = sum([use_volume, use_age, use_ca125])
        if self.input_dim == 0:
            raise ValueError("At least one feature must be enabled")

        feature_names = []
        if use_volume:
            feature_names.append("volume")
        if use_age:
            feature_names.append("age")
        if use_ca125:
            feature_names.append("CA125")
        self.feature_names = feature_names
        logger.info(f"Using features: {feature_names} (input_dim={self.input_dim})")

        self.disease_sites_labels = DISEASE_SITES.get(
            disease_sites, DISEASE_SITES["om"]
        )
        self.stats = {"num_patients": 0, "residual_tumor_counts": {0: 0, 1: 0, 2: 0}}

        if excluded_years is None:
            excluded_years = []
        self.data = self._load_data(data_file, excluded_years)

        if self.normalize and len(self.data) > 0:
            self._compute_normalization_stats()

    def _volume_cache_key(self) -> str:
        sites_str = "_".join(map(str, sorted(self.disease_sites_labels)))
        return f"tumor_volume_mm3_sites_{sites_str}"

    def _compute_normalization_stats(self):
        all_features = torch.stack([p.features for p in self.data])
        self.feature_mean = all_features.mean(dim=0)
        self.feature_std = all_features.std(dim=0) + 1e-6
        logger.info(f"Feature mean: {self.feature_mean.tolist()}")
        logger.info(f"Feature std:  {self.feature_std.tolist()}")

    def set_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.feature_mean = mean
        self.feature_std = std

    def get_normalization_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.feature_mean, self.feature_std

    def _load_data(
        self, data_file: str, excluded_years: list[int]
    ) -> list[MLPPatientData]:
        volume_cache_key = self._volume_cache_key()

        with open(data_file, "r") as f:
            data = json.load(f)

        patients = data["patients"]

        # Check if volumes need to be computed
        needs_computation = self.use_volume and any(
            volume_cache_key not in p
            for p in patients
            if str(p.get("segmentation_path", "")) not in ("-1", "-1.0", "")
            and str(p.get("segmentation_path", "")).strip()
        )

        if needs_computation:
            logger.info(
                f"Computing tumor volumes (sites={self.disease_sites_labels}) and caching to JSON..."
            )
            self._compute_and_cache_volumes(data_file, data, volume_cache_key)
            patients = data["patients"]
        elif self.use_volume:
            logger.info(f"Using cached tumor volumes (key: {volume_cache_key})")

        data_list = []
        for patient in patients:
            if patient["year"] in excluded_years:
                continue

            seg_path = str(patient.get("segmentation_path", ""))
            if seg_path in ("-1", "-1.0", "") or not seg_path.strip():
                continue

            if (
                self.use_volume
                and volume_cache_key not in patient
                and not Path(seg_path).exists()
            ):
                continue

            self.stats["num_patients"] += 1

            tumor_volume = patient.get(volume_cache_key, 0.0)
            age = float(patient.get("Age at diagnosis", 0.0))
            ca125_raw = patient.get("CA125 (U/mL)", 0.0)
            ca125 = float(ca125_raw) if ca125_raw not in (-1, "-1") else 0.0

            feature_list = []
            if self.use_volume:
                feature_list.append(tumor_volume)
            if self.use_age:
                feature_list.append(age)
            if self.use_ca125:
                feature_list.append(ca125)
            features = torch.tensor(feature_list, dtype=torch.float32)

            original_label = patient["CRS"]
            self.stats["residual_tumor_counts"][original_label] = (
                self.stats["residual_tumor_counts"].get(original_label, 0) + 1
            )
            label = torch.tensor(1 if original_label >= 1 else 0, dtype=torch.long)

            record_id = str(patient.get("Record ID", ""))
            crs_label = str(patient.get("CRS: notes", ""))

            data_list.append(
                MLPPatientData(
                    features=features,
                    label=label,
                    record_id=record_id,
                    age=age,
                    ca125=ca125,
                    tumor_volume=tumor_volume,
                    original_crs=original_label,
                    crs_label=crs_label,
                )
            )

        logger.info(f"Loaded {len(data_list)} patients")
        logger.info(f"Class distribution: {self.stats['residual_tumor_counts']}")
        return data_list

    def _compute_and_cache_volumes(
        self, data_file: str, data: dict, volume_cache_key: str
    ):
        """Compute tumor volumes and cache them in the JSON file."""
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        patients = data["patients"]
        computed = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                f"Computing volumes for {Path(data_file).stem}...",
                total=len(patients),
            )
            for patient in patients:
                progress.update(task, advance=1)
                if volume_cache_key in patient:
                    continue

                seg_path = str(patient.get("segmentation_path", ""))
                if (
                    seg_path in ("-1", "-1.0", "")
                    or not seg_path.strip()
                    or not Path(seg_path).exists()
                ):
                    patient[volume_cache_key] = 0.0
                    continue

                patient[volume_cache_key] = compute_tumor_volume(
                    seg_path, self.disease_sites_labels
                )
                computed += 1

        logger.info(f"Computed {computed} tumor volumes")
        with open(data_file, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Cached tumor volumes to {data_file}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        patient = self.data[idx]
        features = patient.features.clone()

        if self.normalize and hasattr(self, "feature_mean"):
            features = (features - self.feature_mean) / self.feature_std

        label_one_hot = F.one_hot(patient.label, num_classes=2).float()

        return {
            "features": features,
            "label": label_one_hot,
            "record_id": patient.record_id,
            "age": patient.age,
            "ca125": patient.ca125,
            "tumor_volume": patient.tumor_volume,
            "original_crs": patient.original_crs,
            "crs_label": patient.crs_label,
        }
