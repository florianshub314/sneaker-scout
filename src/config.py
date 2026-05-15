"""Central configuration: paths, constants, hyperparameters."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_KNOWLEDGE_BASE = ROOT / "data" / "knowledge_base"

SNEAKERS_IMAGE_DIR = DATA_RAW / "sneakers_images"
STOCKX_DIR = DATA_RAW / "stockx"
SHOE_PRICES_DIR = DATA_RAW / "shoe_prices"

KNOWLEDGE_BASE_FILE = DATA_KNOWLEDGE_BASE / "sneaker_market_context.json"

PROCESSED_TRAIN = DATA_PROCESSED / "train.parquet"
PROCESSED_VAL = DATA_PROCESSED / "val.parquet"
PROCESSED_TEST = DATA_PROCESSED / "test.parquet"

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
MODELS_DIR = ROOT / "models"
CV_MODEL_DIR = MODELS_DIR / "cv_model"
ML_MODEL_PATH = MODELS_DIR / "ml_model" / "price_predictor.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "ml_model" / "label_encoders.joblib"

# HuggingFace model IDs
VIT_BASE_MODEL = "google/vit-base-patch16-224"
RESNET_BASE_MODEL = "microsoft/resnet-50"

# ---------------------------------------------------------------------------
# CV constants
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224
CV_BATCH_SIZE = 32
CV_NUM_EPOCHS = 5
CV_LEARNING_RATE = 2e-5
CV_WEIGHT_DECAY = 0.01
CV_WARMUP_STEPS = 100
CV_CONFIDENCE_THRESHOLD = 0.50
CV_HIGH_CONFIDENCE = 0.90
CV_MEDIUM_CONFIDENCE = 0.70

# ---------------------------------------------------------------------------
# ML constants
# ---------------------------------------------------------------------------
ML_RANDOM_STATE = 42
ML_CV_FOLDS = 5
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Feature columns used for ML training
ML_FEATURE_COLS = [
    "retail_price",
    "days_since_release",
    "shoe_size",
    "brand_encoded",
    "month",
    "quarter",
    "size_category_encoded",
    "buyer_region_encoded",
    "sneaker_name_encoded",
]
ML_TARGET_COL = "sale_price"

# ---------------------------------------------------------------------------
# NLP constants
# ---------------------------------------------------------------------------
OPENAI_MODEL = "gpt-4o-mini"
MAX_TOKENS = 512
TEMPERATURE = 0.3

# ---------------------------------------------------------------------------
# App constants
# ---------------------------------------------------------------------------
APP_PORT = 7860
APP_TITLE = "SneakerScout"
APP_SUBTITLE = "AI-Powered Sneaker Recognition & Resell Advisor"

# ---------------------------------------------------------------------------
# Sanity check (run as script to verify data exists)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    paths = {
        "Sneaker images": SNEAKERS_IMAGE_DIR,
        "StockX data": STOCKX_DIR,
        "Shoe prices": SHOE_PRICES_DIR,
        "Knowledge base": KNOWLEDGE_BASE_FILE,
    }
    print("Data path status:")
    for name, path in paths.items():
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {name}: {path}")
