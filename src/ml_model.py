"""ML inference module: load trained price predictor and return resell price + ROI."""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ML_MODEL_PATH,
    LABEL_ENCODER_PATH,
    ML_FEATURE_COLS,
)


class PricePredictor:
    """Wraps a trained gradient boosting regressor for resell price prediction.

    Expects the same feature set used during training. Handles encoding of
    categorical inputs using the saved label encoders.
    """

    def __init__(
        self,
        model_path: Path = ML_MODEL_PATH,
        encoder_path: Path = LABEL_ENCODER_PATH,
    ):
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.model = None
        self.encoders: dict = {}
        self._loaded = False

    def load(self) -> None:
        """Load model and label encoders from disk."""
        if self._loaded:
            return

        if not self.model_path.exists():
            raise RuntimeError(
                f"ML model not found at {self.model_path}. "
                "Run notebooks/03_ml_training.ipynb first."
            )

        self.model = joblib.load(self.model_path)

        if self.encoder_path.exists():
            self.encoders = joblib.load(self.encoder_path)

        self._loaded = True

    def _encode_features(self, features: dict) -> np.ndarray:
        """Convert raw feature dict to model-ready array."""
        row = {}

        # Pass-through numeric features
        row["retail_price"] = float(features.get("retail_price", 0))
        row["days_since_release"] = float(features.get("days_since_release", 180))
        row["shoe_size"] = float(features.get("shoe_size", 10))
        row["month"] = int(features.get("month", 6))
        row["quarter"] = int(features.get("quarter", 2))

        # Encode categoricals using saved mappings
        for col in ("brand", "sneaker_name", "buyer_region", "size_category"):
            encoded_col = f"{col}_encoded"
            raw_val = features.get(col, "")
            mapping = self.encoders.get(col, {})
            row[encoded_col] = mapping.get(raw_val, -1)

        feature_vec = np.array([row.get(c, 0) for c in ML_FEATURE_COLS], dtype=np.float32)
        return feature_vec.reshape(1, -1)

    def predict(
        self,
        sneaker_name: str,
        retail_price: float,
        shoe_size: float = 10.0,
        days_since_release: int = 180,
        brand: str = "",
        region: str = "United States",
    ) -> tuple[float, float]:
        """Predict resell price and ROI for a given sneaker configuration.

        Args:
            sneaker_name: Name from the CV classifier output.
            retail_price: Original retail price in USD.
            shoe_size: US shoe size (default 10).
            days_since_release: Days from release date to sale (default 180).
            brand: Brand name (extracted from sneaker_name if empty).
            region: Buyer region (default United States).

        Returns:
            Tuple of (predicted_resell_price, roi) where roi is a fraction (0.5 = 50%).
        """
        self.load()

        if not brand:
            brand = _extract_brand(sneaker_name)

        size_category = _size_category(shoe_size)
        import datetime
        now = datetime.datetime.now()

        features = {
            "retail_price": retail_price,
            "days_since_release": days_since_release,
            "shoe_size": shoe_size,
            "month": now.month,
            "quarter": (now.month - 1) // 3 + 1,
            "brand": brand,
            "sneaker_name": sneaker_name,
            "buyer_region": region,
            "size_category": size_category,
        }

        X = self._encode_features(features)
        predicted_price = float(self.model.predict(X)[0])
        predicted_price = max(predicted_price, retail_price * 0.5)

        roi = (predicted_price - retail_price) / retail_price if retail_price > 0 else 0.0

        return round(predicted_price, 2), round(roi, 4)


def _extract_brand(sneaker_name: str) -> str:
    """Heuristically extract brand from sneaker name."""
    name_lower = sneaker_name.lower()
    if "yeezy" in name_lower or "adidas" in name_lower:
        return "Adidas"
    elif "jordan" in name_lower or "air jordan" in name_lower or "nike" in name_lower:
        return "Nike"
    elif "new balance" in name_lower:
        return "New Balance"
    elif "converse" in name_lower:
        return "Converse"
    elif "vans" in name_lower:
        return "Vans"
    elif "off-white" in name_lower or "off white" in name_lower:
        return "Off-White"
    return "Nike"  # default to most common in dataset


def _size_category(shoe_size: float) -> str:
    if shoe_size <= 8:
        return "small"
    elif shoe_size <= 11:
        return "medium"
    else:
        return "large"


# Singleton instance for app use
_predictor: Optional[PricePredictor] = None


def get_predictor() -> PricePredictor:
    """Return the shared PricePredictor instance (lazy-loaded)."""
    global _predictor
    if _predictor is None:
        _predictor = PricePredictor()
    return _predictor
