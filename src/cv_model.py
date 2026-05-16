"""CV inference module: load fine-tuned ViT and classify sneaker images."""

import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from src.config import (
    CV_HF_MODEL_ID,
    CV_MODEL_DIR,
    CV_CONFIDENCE_THRESHOLD,
    CV_HIGH_CONFIDENCE,
    CV_MEDIUM_CONFIDENCE,
    IMAGE_SIZE,
)
from src.preprocessing import get_eval_transforms, load_image


class SneakerClassifier:
    """Wraps a fine-tuned ViT model for sneaker classification inference.

    Prefers a locally trained model under CV_MODEL_DIR; falls back to the
    published checkpoint on the Hugging Face Hub (CV_HF_MODEL_ID) when no
    local weights are present, e.g. on a fresh Space or CI runner.
    """

    def __init__(self, model_dir: Path = CV_MODEL_DIR, device: Optional[str] = None):
        self.model_dir = model_dir
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: Optional[ViTForImageClassification] = None
        self.processor: Optional[ViTImageProcessor] = None
        self.id2label: dict[int, str] = {}
        self._loaded = False

    def _resolve_source(self) -> str:
        """Return a local path string if weights exist locally, else the HF Hub id."""
        has_local_weights = (
            self.model_dir.exists()
            and ((self.model_dir / "model.safetensors").exists()
                 or (self.model_dir / "pytorch_model.bin").exists())
        )
        return str(self.model_dir) if has_local_weights else CV_HF_MODEL_ID

    def load(self) -> None:
        """Load model weights, processor, and label mapping (local first, then HF Hub)."""
        if self._loaded:
            return

        source = self._resolve_source()
        self.processor = ViTImageProcessor.from_pretrained(source)
        self.model = ViTForImageClassification.from_pretrained(source)
        self.model.to(self.device)
        self.model.eval()

        # Prefer id2label.json next to the local weights; fall back to the value
        # embedded in the model config (populated from config.json on the Hub).
        label_map_path = self.model_dir / "id2label.json"
        if label_map_path.exists():
            with open(label_map_path) as f:
                raw = json.load(f)
            self.id2label = {int(k): v for k, v in raw.items()}
        elif self.model.config.id2label:
            self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        else:
            self.id2label = {i: str(i) for i in range(self.model.config.num_labels)}

        self._loaded = True

    @torch.no_grad()
    def predict(self, image: Image.Image) -> tuple[str, float]:
        """Classify a sneaker image.

        Args:
            image: PIL RGB image.

        Returns:
            Tuple of (predicted class name, confidence score 0-1).
        """
        self.load()

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_idx = int(probs.argmax().item())
        confidence = float(probs[pred_idx].item())
        label = self.id2label.get(pred_idx, f"class_{pred_idx}")

        return label, confidence

    def predict_topk(self, image: Image.Image, k: int = 3) -> list[tuple[str, float]]:
        """Return top-k predictions with probabilities."""
        self.load()

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k2: v.to(self.device) for k2, v in inputs.items()}

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        top_probs, top_idx = torch.topk(probs, k=min(k, len(probs)))

        return [
            (self.id2label.get(int(idx.item()), f"class_{idx.item()}"), float(p.item()))
            for idx, p in zip(top_idx, top_probs)
        ]


def get_confidence_level(confidence: float) -> str:
    """Map confidence score to human-readable level."""
    if confidence >= CV_HIGH_CONFIDENCE:
        return "high"
    elif confidence >= CV_MEDIUM_CONFIDENCE:
        return "medium"
    else:
        return "low"


# Singleton instance for app use
_classifier: Optional[SneakerClassifier] = None


def get_classifier() -> SneakerClassifier:
    """Return the shared SneakerClassifier instance (lazy-loaded)."""
    global _classifier
    if _classifier is None:
        _classifier = SneakerClassifier()
    return _classifier
