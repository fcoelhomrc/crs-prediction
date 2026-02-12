"""Model architectures for CRS prediction.

Contains:
- VolumeProcessor: preprocesses volumetric images for ViT feature extraction
- FeatureExtractor: loads pretrained ViT/DINOv2 backbones
- AttentionPooling: attention-based pooling across slices
- ClinicalDataEncoder: encodes tabular clinical data
- VitClassifier: main ViT-based volumetric classifier
- MLP: standalone MLP for clinical feature classification
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, ViTModel


class VolumeProcessor(nn.Module):
    """Preprocesses volumetric medical images for feature extraction.

    Handles normalization and channel conversion for input slices.
    """

    def __init__(self, config: dict):
        super().__init__()
        if config["feature_extractor"] in ("google-16", "google-32"):
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        elif config["feature_extractor"] in ("dinov2-small", "dinov2-base"):
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]

    def forward(self, x: torch.Tensor, num_slices: int) -> torch.Tensor:
        """Normalize and convert grayscale slices to RGB for transformer input.

        Args:
            x: Input tensor [batch_size * num_slices, 1, H, W].
            num_slices: Number of slices per volume.

        Returns:
            Normalized tensor [batch_size * num_slices, 3, H, W].
        """
        batch_slices = x.shape[0]
        original_batch_size = batch_slices // num_slices

        x = x.reshape(original_batch_size, num_slices, x.shape[2], x.shape[3])

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

        x = x.reshape(batch_slices, 1, x.shape[2], x.shape[3])
        x = x.repeat(1, 3, 1, 1)

        mean = torch.tensor(self.mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        return x


class FeatureExtractor:
    """Wrapper for selecting and loading a transformer-based feature extractor."""

    EXTRACTORS = {
        "google-16": ("google/vit-base-patch16-224-in21k", ViTModel),
        "google-32": ("google/vit-base-patch32-224-in21k", ViTModel),
        "dinov2-small": ("facebook/dinov2-small", AutoModel),
        "dinov2-base": ("facebook/dinov2-base", AutoModel),
    }

    def __init__(self, config: dict):
        self.config = config
        self.feature_extractor = self._load()
        self.input_dim = self.feature_extractor.config.hidden_size

    def _load(self) -> nn.Module:
        name = self.config["feature_extractor"]
        if name not in self.EXTRACTORS:
            raise ValueError(
                f"Invalid feature extractor: {name}. "
                f"Choose from: {list(self.EXTRACTORS.keys())}"
            )
        path, model_class = self.EXTRACTORS[name]
        device_str = self.config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        device = torch.device(device_str)
        return model_class.from_pretrained(path).to(device)

    def get_extractor(self) -> nn.Module:
        return self.feature_extractor

    def get_input_dim(self) -> int:
        return self.input_dim


class AttentionPooling(nn.Module):
    """Attention-based pooling across slices of a volume."""

    def __init__(self, input_dim: int, dropout_rate: float, attention_resc: int = 1):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim // attention_resc)
        self.key = nn.Linear(input_dim, input_dim // attention_resc)
        self.value = nn.Linear(input_dim, input_dim // attention_resc)
        self.norm = nn.LayerNorm(input_dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-pooled features across slices.

        Args:
            x: Tensor [batch_size, num_slices, input_dim].

        Returns:
            pooled: Tensor [batch_size, pooled_dim].
            weights: Attention weights [batch_size, num_slices, num_slices].
        """
        x = self.norm(x)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(x.size(-1))
        weights = self.drop(F.softmax(scores, dim=-1))
        attended = torch.matmul(weights, V)
        pooled = attended.mean(dim=1)
        return pooled, weights


class ClinicalDataEncoder(nn.Module):
    """Encodes tabular clinical data into a learned feature representation."""

    def __init__(self, clinical_dim: int, encoding_dim: int):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.encoder = nn.Sequential(
            nn.LayerNorm(clinical_dim),
            nn.Linear(clinical_dim, encoding_dim),
            nn.BatchNorm1d(encoding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def get_output_dim(self) -> int:
        return self.encoding_dim


class VitClassifier(nn.Module):
    """Main classifier for volumetric medical images using ViT and attention pooling.

    Optionally incorporates clinical tabular data (age, CA125, etc.).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        fe = FeatureExtractor(config)
        self.feature_extractor = fe.get_extractor()
        self.input_dim = fe.get_input_dim()

        self.clinical_features_enabled = config["clinical_features"]
        if self.clinical_features_enabled:
            self.clinical_encoder = ClinicalDataEncoder(
                config["clinical_input_dim"], config["clinical_encoding_dim"]
            )
            clinical_encoded_dim = self.clinical_encoder.get_output_dim()
        else:
            self.clinical_encoder = None
            clinical_encoded_dim = 0

        self.num_classes = 1
        self.dropout_rate = config["dropout"]
        self.num_heads = config["num_heads"]
        self.max_slices = 256
        self.attention_resc = config["attention_resc"]

        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.volume_processor = VolumeProcessor(config)
        self.attention_pool = AttentionPooling(
            self.input_dim, self.dropout_rate, self.attention_resc
        )

        ct_feature_dim = self.input_dim // self.attention_resc
        combined_dim = ct_feature_dim + clinical_encoded_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, combined_dim // 2),
            nn.BatchNorm1d(combined_dim // 2),
            nn.ReLU(),
            nn.Linear(combined_dim // 2, combined_dim // 8),
            nn.BatchNorm1d(combined_dim // 8),
            nn.ReLU(),
            nn.Linear(combined_dim // 8, combined_dim // 32),
            nn.BatchNorm1d(combined_dim // 32),
            nn.ReLU(),
            nn.Linear(combined_dim // 32, self.num_classes),
        )

    def forward(
        self, images: torch.Tensor, clinical_features: torch.Tensor | None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            images: Input tensor [batch_size, channels, height, width, slices].
            clinical_features: Optional clinical data tensor [batch_size, clinical_dim].

        Returns:
            Tuple of (logits, attention_weights, pooled_features).
        """
        batch_size, _, height, width, num_slices = images.shape

        if num_slices > self.max_slices:
            raise ValueError(f"Max slices exceeded: {num_slices} > {self.max_slices}")

        slices = images.permute(0, 4, 1, 2, 3)  # (B, S, 1, H, W)
        slices = slices.reshape(-1, 1, height, width)  # (B*S, 1, H, W)
        processed_slices = self.volume_processor(slices, num_slices)

        with torch.no_grad():
            features = self.feature_extractor(processed_slices).last_hidden_state[
                :, 0, :
            ]

        features = features.reshape(batch_size, num_slices, -1)
        pooled_features, attention_weights = self.attention_pool(features)

        if self.clinical_features_enabled and clinical_features is not None:
            clinical_encoded = self.clinical_encoder(clinical_features)
            combined = torch.cat([pooled_features, clinical_encoded], dim=1)
            logits = self.classifier(combined)
        else:
            logits = self.classifier(pooled_features)

        return logits, attention_weights, pooled_features


class MLP(nn.Module):
    """Configurable MLP for clinical feature classification.

    Args:
        input_dim: Number of input features.
        hidden_dims: List of hidden layer dimensions.
        dropout: Dropout rate.
        num_classes: Number of output classes (1 for binary with BCEWithLogitsLoss).
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.10,
        num_classes: int = 1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        layers: list[nn.Module] = [nn.BatchNorm1d(input_dim)]
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
