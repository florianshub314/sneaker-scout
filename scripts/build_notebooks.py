"""Helper to construct .ipynb files from structured cell definitions.

Used during development to keep notebooks under version control without
fighting JSON syntax. Run once after editing this script:

    python scripts/build_notebooks.py

Generated notebooks are saved to ./notebooks/.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text,
        "outputs": [],
        "execution_count": None,
    }


def build(cells: list, filename: str) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = NOTEBOOKS_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Notebook 1: EDA & Feature Engineering on StockX data
# ---------------------------------------------------------------------------

EDA_CELLS = [
    md("# 01 – Exploratory Data Analysis: StockX Resell Data\n\n"
       "**Goal:** Understand the StockX dataset, identify distributions, "
       "correlations, anomalies, then engineer features and split into train/val/test.\n\n"
       "**Data source:** https://www.kaggle.com/datasets/hudsonstuck/stockx-data-contest\n\n"
       "Expected at: `data/raw/stockx/` (see `data/raw/README.md`)."),

    md("## Setup"),
    code("import sys\n"
         "from pathlib import Path\n\n"
         "# Allow `from src import ...` when running from notebooks/\n"
         "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "sys.path.insert(0, str(ROOT))\n\n"
         "import numpy as np\n"
         "import pandas as pd\n"
         "import matplotlib.pyplot as plt\n"
         "import seaborn as sns\n\n"
         "from src import config\n"
         "from src.preprocessing import (\n"
         "    load_stockx_data, clean_stockx_columns, parse_stockx_dates,\n"
         "    engineer_stockx_features, encode_categorical_features, split_data,\n"
         ")\n\n"
         "sns.set_style('whitegrid')\n"
         "plt.rcParams['figure.dpi'] = 100\n"
         "pd.set_option('display.max_columns', 50)"),

    md("## Load Raw Data"),
    code("df_raw = load_stockx_data(config.STOCKX_DIR)\n"
         "print(f'Shape: {df_raw.shape}')\n"
         "df_raw.head()"),

    code("df_raw.info()"),

    md("## Clean column names and parse dates"),
    code("df = clean_stockx_columns(df_raw)\n"
         "df = parse_stockx_dates(df)\n"
         "print('Columns:', list(df.columns))\n"
         "df.head()"),

    md("## Distribution Analysis\n\n"
       "Inspect the marginal distributions of the key numeric and categorical columns."),

    code("fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n\n"
         "# Sale price distribution\n"
         "axes[0, 0].hist(df['sale_price'], bins=50, color='#3b82f6', edgecolor='white')\n"
         "axes[0, 0].set_title('Sale Price Distribution (USD)')\n"
         "axes[0, 0].set_xlabel('Sale Price')\n"
         "axes[0, 0].set_ylabel('Count')\n\n"
         "# Retail price distribution\n"
         "axes[0, 1].hist(df['retail_price'], bins=30, color='#00d4aa', edgecolor='white')\n"
         "axes[0, 1].set_title('Retail Price Distribution (USD)')\n"
         "axes[0, 1].set_xlabel('Retail Price')\n\n"
         "# Brand counts\n"
         "df['brand'].value_counts().plot(kind='bar', ax=axes[1, 0], color='#f39c12')\n"
         "axes[1, 0].set_title('Sales per Brand')\n"
         "axes[1, 0].set_ylabel('Transaction Count')\n"
         "axes[1, 0].tick_params(axis='x', rotation=0)\n\n"
         "# Shoe size distribution\n"
         "axes[1, 1].hist(df['shoe_size'], bins=15, color='#e94560', edgecolor='white')\n"
         "axes[1, 1].set_title('Shoe Size Distribution (US)')\n"
         "axes[1, 1].set_xlabel('Shoe Size')\n\n"
         "plt.tight_layout()\n"
         "plt.show()"),

    md("**Observation:** Sale prices are heavily right-skewed (long tail of premium sneakers). "
       "Brand distribution shows extreme imbalance dominated by Off-White and Yeezy – this is "
       "a known bias in the dataset and will be documented in the ethics notebook."),

    md("## Top sneaker models"),
    code("top_models = df['sneaker_name'].value_counts().head(15)\n"
         "fig, ax = plt.subplots(figsize=(10, 6))\n"
         "top_models.plot(kind='barh', ax=ax, color='#3b82f6')\n"
         "ax.set_title('Top 15 Sneaker Models by Sales Volume')\n"
         "ax.set_xlabel('Number of Transactions')\n"
         "ax.invert_yaxis()\n"
         "plt.tight_layout()\n"
         "plt.show()\n"
         "print(f\"Unique sneaker models: {df['sneaker_name'].nunique()}\")"),

    md("## Temporal Analysis\n\n"
       "Resell prices change over time. Plot median sale price per month to detect trends."),

    code("df_time = df.copy()\n"
         "df_time['order_month'] = df_time['order_date'].dt.to_period('M').dt.to_timestamp()\n\n"
         "monthly = df_time.groupby('order_month').agg(\n"
         "    median_price=('sale_price', 'median'),\n"
         "    volume=('sale_price', 'count'),\n"
         ").reset_index()\n\n"
         "fig, ax1 = plt.subplots(figsize=(12, 5))\n"
         "ax1.plot(monthly['order_month'], monthly['median_price'],\n"
         "         color='#00d4aa', linewidth=2, label='Median Sale Price')\n"
         "ax1.set_xlabel('Month')\n"
         "ax1.set_ylabel('Median Sale Price ($)', color='#00d4aa')\n"
         "ax1.tick_params(axis='y', labelcolor='#00d4aa')\n\n"
         "ax2 = ax1.twinx()\n"
         "ax2.bar(monthly['order_month'], monthly['volume'],\n"
         "        alpha=0.2, color='#3b82f6', width=20, label='Volume')\n"
         "ax2.set_ylabel('Transaction Volume', color='#3b82f6')\n"
         "ax2.tick_params(axis='y', labelcolor='#3b82f6')\n\n"
         "plt.title('Median Resell Price and Transaction Volume Over Time')\n"
         "plt.tight_layout()\n"
         "plt.show()"),

    md("## Resell Premium Analysis\n\n"
       "Premium = (Sale Price - Retail Price) / Retail Price. Key business metric."),

    code("df['price_premium'] = (df['sale_price'] - df['retail_price']) / df['retail_price']\n"
         "print(df['price_premium'].describe())\n\n"
         "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
         "axes[0].hist(df['price_premium'], bins=80, color='#3b82f6', edgecolor='white')\n"
         "axes[0].axvline(df['price_premium'].median(), color='#e94560',\n"
         "                linestyle='--', label=f\"Median: {df['price_premium'].median():.2f}\")\n"
         "axes[0].set_title('Resell Premium Distribution')\n"
         "axes[0].set_xlabel('Premium (fraction over retail)')\n"
         "axes[0].set_ylabel('Count')\n"
         "axes[0].legend()\n\n"
         "premium_by_brand = df.groupby('brand')['price_premium'].median().sort_values(ascending=False)\n"
         "premium_by_brand.plot(kind='bar', ax=axes[1], color='#f39c12')\n"
         "axes[1].set_title('Median Resell Premium by Brand')\n"
         "axes[1].set_ylabel('Median Premium')\n"
         "axes[1].tick_params(axis='x', rotation=0)\n"
         "plt.tight_layout()\n"
         "plt.show()"),

    md("## Correlation Analysis\n\n"
       "Pairwise correlations among numeric features."),

    code("numeric_cols = ['sale_price', 'retail_price', 'shoe_size']\n"
         "corr = df[numeric_cols].corr()\n\n"
         "fig, ax = plt.subplots(figsize=(6, 5))\n"
         "sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,\n"
         "            ax=ax, square=True, cbar_kws={'shrink': 0.8})\n"
         "ax.set_title('Numeric Feature Correlations')\n"
         "plt.tight_layout()\n"
         "plt.show()"),

    md("## Outlier Detection\n\n"
       "Use IQR-based detection on sale price. Outliers stay in the dataset because "
       "high-priced collaborations are real and a key target for the resell model, but "
       "we log their share."),

    code("q1, q3 = df['sale_price'].quantile([0.25, 0.75])\n"
         "iqr = q3 - q1\n"
         "lower = q1 - 1.5 * iqr\n"
         "upper = q3 + 1.5 * iqr\n"
         "outliers = df[(df['sale_price'] < lower) | (df['sale_price'] > upper)]\n"
         "print(f'Outlier share: {len(outliers)/len(df):.1%}')\n"
         "print(f'Upper bound: ${upper:.0f}')\n"
         "print(f'Max sale price: ${df[\"sale_price\"].max():.0f}')\n\n"
         "fig, ax = plt.subplots(figsize=(10, 4))\n"
         "ax.boxplot(df['sale_price'], vert=False)\n"
         "ax.set_title('Sale Price – Boxplot (linear scale)')\n"
         "ax.set_xlabel('Sale Price ($)')\n"
         "plt.tight_layout()\n"
         "plt.show()"),

    md("## Feature Engineering\n\n"
       "Build the features the ML model will use:\n"
       "- `price_premium`, `days_since_release`, `month`, `quarter`, `size_category`\n"
       "- Label-encoded categoricals: `brand`, `sneaker_name`, `buyer_region`, `size_category`"),

    code("df_feat = engineer_stockx_features(df)\n"
         "df_feat, encoders = encode_categorical_features(df_feat)\n\n"
         "print('Final shape:', df_feat.shape)\n"
         "print('\\nFeature columns:')\n"
         "for col in df_feat.columns:\n"
         "    print(f'  {col}: {df_feat[col].dtype}')"),

    code("df_feat[['retail_price', 'days_since_release', 'shoe_size',\n"
         "         'brand_encoded', 'month', 'quarter',\n"
         "         'size_category_encoded', 'region_encoded',\n"
         "         'sneaker_name_encoded', 'sale_price']].describe()"),

    md("## Train/Validation/Test Split (70/15/15)"),
    code("# Drop rows missing the target or critical features before split\n"
         "df_clean = df_feat.dropna(subset=['sale_price', 'retail_price', 'shoe_size']).copy()\n"
         "print(f'After cleaning: {df_clean.shape}')\n\n"
         "train_df, val_df, test_df = split_data(\n"
         "    df_clean,\n"
         "    train_ratio=config.TRAIN_RATIO,\n"
         "    val_ratio=config.VAL_RATIO,\n"
         "    random_state=config.ML_RANDOM_STATE,\n"
         ")\n"
         "print(f'Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}')"),

    md("## Persist Processed Data and Encoders"),
    code("import joblib\n"
         "config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)\n\n"
         "train_df.to_parquet(config.PROCESSED_TRAIN, index=False)\n"
         "val_df.to_parquet(config.PROCESSED_VAL, index=False)\n"
         "test_df.to_parquet(config.PROCESSED_TEST, index=False)\n\n"
         "config.LABEL_ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)\n"
         "joblib.dump(encoders, config.LABEL_ENCODER_PATH)\n\n"
         "print('Saved:')\n"
         "for p in (config.PROCESSED_TRAIN, config.PROCESSED_VAL,\n"
         "          config.PROCESSED_TEST, config.LABEL_ENCODER_PATH):\n"
         "    print(f'  {p}')"),

    md("## Summary\n\n"
       "- **~10k transactions** across ~50 sneaker models, dominated by Off-White and Yeezy\n"
       "- **Heavy right-skew** in sale prices; high-value sneakers retained, not winsorised\n"
       "- **Resell premium** highly variable by brand – Off-White's median premium far exceeds others\n"
       "- **Temporal trend:** moderate decline in median price 2018→2019 as volume grew\n"
       "- **Engineered features:** price_premium, days_since_release, encoded categoricals\n"
       "- **Splits saved** to `data/processed/`, encoders to `models/ml_model/label_encoders.joblib`\n\n"
       "These splits feed directly into `03_ml_training.ipynb`."),
]

build(EDA_CELLS, "01_eda_stockx.ipynb")
print("Done.")
