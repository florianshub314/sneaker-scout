"""Shared preprocessing utilities for CV, ML, and data loading."""

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Augmented transforms for training (random flips, color jitter, rotation)."""
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Deterministic transforms for validation and inference."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_image(path: str | Path) -> Image.Image:
    """Load an image from disk and convert to RGB."""
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# StockX data preprocessing
# ---------------------------------------------------------------------------

def load_stockx_data(data_dir: Path) -> pd.DataFrame:
    """Load raw StockX data from the given directory."""
    xlsx_files = list(data_dir.glob("*.xlsx"))
    csv_files = list(data_dir.glob("*.csv"))

    if xlsx_files:
        df = pd.read_excel(xlsx_files[0])
    elif csv_files:
        df = pd.read_csv(csv_files[0])
    else:
        raise FileNotFoundError(f"No StockX data found in {data_dir}")

    return df


def clean_stockx_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names to snake_case and strip BOM markers."""
    df = df.copy()
    df.columns = [
        re.sub(r"\s+", "_", col.strip().lstrip("﻿").lower().replace("-", "_"))
        for col in df.columns
    ]
    return df


def parse_currency(series: pd.Series) -> pd.Series:
    """Convert a currency-formatted column ('$1,097') to a float series."""
    if series.dtype.kind in {"f", "i"}:
        return series.astype(float)
    return (
        series.astype(str)
        .str.replace(r"[\$,\s]", "", regex=True)
        .replace({"": None, "nan": None})
        .astype(float)
    )


def parse_stockx_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse order_date and release_date columns to datetime."""
    df = df.copy()
    for col in ("order_date", "release_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def normalise_stockx_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Strip $ and commas from sale_price and retail_price."""
    df = df.copy()
    for col in ("sale_price", "retail_price"):
        if col in df.columns:
            df[col] = parse_currency(df[col])
    return df


def engineer_stockx_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from raw StockX columns.

    Adds: price_premium, days_since_release, month, quarter,
          size_category, brand_encoded, sneaker_name_encoded,
          region_encoded, size_category_encoded.
    """
    df = df.copy()

    # Price premium: percentage above retail
    df["price_premium"] = (df["sale_price"] - df["retail_price"]) / df["retail_price"]

    # Days between release and sale
    if "order_date" in df.columns and "release_date" in df.columns:
        df["days_since_release"] = (df["order_date"] - df["release_date"]).dt.days
        df["days_since_release"] = df["days_since_release"].clip(lower=0)
        df["month"] = df["order_date"].dt.month
        df["quarter"] = df["order_date"].dt.quarter

    # Shoe size category
    def size_category(s):
        if s <= 8:
            return "small"
        elif s <= 11:
            return "medium"
        else:
            return "large"

    if "shoe_size" in df.columns:
        df["size_category"] = df["shoe_size"].apply(size_category)

    return df


def encode_categorical_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns. Returns df and encoder mappings."""
    df = df.copy()
    encoders: dict[str, dict] = {}

    categorical_cols = ["brand", "sneaker_name", "buyer_region", "size_category"]
    for col in categorical_cols:
        if col not in df.columns:
            continue
        mapping = {v: i for i, v in enumerate(sorted(df[col].dropna().unique()))}
        df[f"{col}_encoded"] = df[col].map(mapping).fillna(-1).astype(int)
        encoders[col] = mapping

    return df, encoders


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split (by brand if available)."""
    from sklearn.model_selection import train_test_split

    test_ratio = 1.0 - train_ratio - val_ratio
    strat_col = "brand_encoded" if "brand_encoded" in df.columns else None

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=df[strat_col] if strat_col else None,
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - relative_val),
        random_state=random_state,
        stratify=temp_df[strat_col] if strat_col else None,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Shoe prices supplementary data
# ---------------------------------------------------------------------------

def load_shoe_prices(data_dir: Path) -> pd.DataFrame:
    """Load supplementary shoe prices dataset."""
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No shoe prices CSV found in {data_dir}")
    return pd.read_csv(csv_files[0])
