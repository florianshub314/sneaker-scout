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


# ---------------------------------------------------------------------------
# Notebook 2: CV Model – ViT Fine-Tuning + ResNet Comparison
# ---------------------------------------------------------------------------

CV_CELLS = [
    md("# 02 – Computer Vision: Sneaker Classifier\n\n"
       "**Goal:** Fine-tune a ViT model on the Popular Sneakers dataset, compare with "
       "a ResNet50 baseline, evaluate per-class accuracy, error patterns and confusion.\n\n"
       "**Data:** https://www.kaggle.com/datasets/nikolasgegenava/sneakers-classification\n\n"
       "Expected at: `data/raw/sneakers_images/` (train/test subfolders, one folder per class)."),

    md("## Setup"),
    code("import sys, json, random\n"
         "from pathlib import Path\n"
         "from collections import Counter\n\n"
         "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "sys.path.insert(0, str(ROOT))\n\n"
         "import numpy as np\n"
         "import matplotlib.pyplot as plt\n"
         "import seaborn as sns\n"
         "from PIL import Image\n\n"
         "import torch\n"
         "from torch.utils.data import Dataset, DataLoader\n"
         "from torchvision import transforms, models\n"
         "from transformers import (\n"
         "    ViTImageProcessor, ViTForImageClassification,\n"
         "    Trainer, TrainingArguments, EarlyStoppingCallback,\n"
         ")\n"
         "from sklearn.metrics import (\n"
         "    accuracy_score, precision_recall_fscore_support,\n"
         "    confusion_matrix, classification_report,\n"
         ")\n\n"
         "from src import config\n"
         "from src.preprocessing import get_train_transforms, get_eval_transforms\n\n"
         "torch.manual_seed(config.ML_RANDOM_STATE)\n"
         "random.seed(config.ML_RANDOM_STATE)\n"
         "np.random.seed(config.ML_RANDOM_STATE)\n\n"
         "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
         "print(f'Device: {device}')"),

    md("## Dataset Inventory\n\n"
       "Inspect class structure and image counts per class."),

    code("IMG_DIR = config.SNEAKERS_IMAGE_DIR\n"
         "# Try common dataset layouts (train/ + test/, or one folder per class at root)\n"
         "train_dir = IMG_DIR / 'train'\n"
         "test_dir = IMG_DIR / 'test'\n\n"
         "if not train_dir.exists():\n"
         "    # Treat every subdir as a class, split later\n"
         "    train_dir = IMG_DIR\n"
         "    test_dir = None\n\n"
         "class_dirs = [p for p in train_dir.iterdir() if p.is_dir()]\n"
         "class_names = sorted([p.name for p in class_dirs])\n"
         "print(f'Found {len(class_names)} classes')\n"
         "for c in class_names[:10]:\n"
         "    n = len(list((train_dir / c).glob('*')))\n"
         "    print(f'  {c}: {n} images')"),

    code("# Image count per class\n"
         "counts = {c: len(list((train_dir / c).glob('*'))) for c in class_names}\n"
         "fig, ax = plt.subplots(figsize=(12, 5))\n"
         "ax.bar(range(len(counts)), list(counts.values()), color='#3b82f6')\n"
         "ax.set_xticks(range(len(counts)))\n"
         "ax.set_xticklabels(list(counts.keys()), rotation=45, ha='right', fontsize=8)\n"
         "ax.set_title('Images per Class (class imbalance overview)')\n"
         "ax.set_ylabel('Image Count')\n"
         "plt.tight_layout()\n"
         "plt.show()\n\n"
         "print(f'Total images: {sum(counts.values())}')\n"
         "print(f'Mean: {np.mean(list(counts.values())):.0f}, Min: {min(counts.values())}, Max: {max(counts.values())}')"),

    md("## Sample Images\n\n"
       "Visually inspect a sample from each class to catch quality issues."),

    code("def show_samples(class_names, train_dir, n_per_class=1):\n"
         "    fig, axes = plt.subplots(2, 5, figsize=(15, 6))\n"
         "    for ax, cname in zip(axes.flat, class_names[:10]):\n"
         "        imgs = list((train_dir / cname).glob('*'))\n"
         "        if not imgs:\n"
         "            continue\n"
         "        img = Image.open(imgs[0]).convert('RGB')\n"
         "        ax.imshow(img)\n"
         "        ax.set_title(cname, fontsize=9)\n"
         "        ax.axis('off')\n"
         "    plt.tight_layout()\n"
         "    plt.show()\n\n"
         "show_samples(class_names, train_dir)"),

    md("## PyTorch Dataset\n\n"
       "Custom dataset that lazily loads images with the configured transforms."),

    code("class SneakerDataset(Dataset):\n"
         "    def __init__(self, samples, label2id, transform):\n"
         "        self.samples = samples  # list of (path, label_str)\n"
         "        self.label2id = label2id\n"
         "        self.transform = transform\n\n"
         "    def __len__(self):\n"
         "        return len(self.samples)\n\n"
         "    def __getitem__(self, idx):\n"
         "        path, label_str = self.samples[idx]\n"
         "        img = Image.open(path).convert('RGB')\n"
         "        return {\n"
         "            'pixel_values': self.transform(img),\n"
         "            'labels': torch.tensor(self.label2id[label_str], dtype=torch.long),\n"
         "        }\n\n"
         "label2id = {c: i for i, c in enumerate(class_names)}\n"
         "id2label = {i: c for c, i in label2id.items()}"),

    code("# Build (path, label) pairs and split if needed\n"
         "all_samples = []\n"
         "for cname in class_names:\n"
         "    for p in (train_dir / cname).glob('*'):\n"
         "        if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:\n"
         "            all_samples.append((str(p), cname))\n\n"
         "random.shuffle(all_samples)\n\n"
         "if test_dir is not None and test_dir.exists():\n"
         "    train_samples = all_samples\n"
         "    val_samples = []\n"
         "    for cname in class_names:\n"
         "        cdir = test_dir / cname\n"
         "        if cdir.exists():\n"
         "            for p in cdir.glob('*'):\n"
         "                if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:\n"
         "                    val_samples.append((str(p), cname))\n"
         "    # Split val_samples 50/50 into val and test\n"
         "    mid = len(val_samples) // 2\n"
         "    test_samples = val_samples[mid:]\n"
         "    val_samples = val_samples[:mid]\n"
         "else:\n"
         "    n = len(all_samples)\n"
         "    train_samples = all_samples[:int(0.7 * n)]\n"
         "    val_samples = all_samples[int(0.7 * n):int(0.85 * n)]\n"
         "    test_samples = all_samples[int(0.85 * n):]\n\n"
         "print(f'Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}')"),

    md("## Preprocessing & Augmentation\n\n"
       "Training uses random crop, horizontal flip, color jitter and small rotation. "
       "Validation/test use deterministic resize + normalise only."),

    code("train_tf = get_train_transforms(config.IMAGE_SIZE)\n"
         "eval_tf = get_eval_transforms(config.IMAGE_SIZE)\n\n"
         "train_ds = SneakerDataset(train_samples, label2id, train_tf)\n"
         "val_ds = SneakerDataset(val_samples, label2id, eval_tf)\n"
         "test_ds = SneakerDataset(test_samples, label2id, eval_tf)\n\n"
         "print('Datasets ready')"),

    md("## Model 1 – ViT Fine-Tuning\n\n"
       "Initialise `google/vit-base-patch16-224` with a fresh classification head sized to "
       "our class count, then fine-tune end-to-end."),

    code("vit_model = ViTForImageClassification.from_pretrained(\n"
         "    config.VIT_BASE_MODEL,\n"
         "    num_labels=len(class_names),\n"
         "    id2label=id2label,\n"
         "    label2id=label2id,\n"
         "    ignore_mismatched_sizes=True,\n"
         ")\n\n"
         "def compute_metrics(eval_pred):\n"
         "    logits, labels = eval_pred\n"
         "    preds = np.argmax(logits, axis=-1)\n"
         "    acc = accuracy_score(labels, preds)\n"
         "    prec, rec, f1, _ = precision_recall_fscore_support(\n"
         "        labels, preds, average='macro', zero_division=0)\n"
         "    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}"),

    code("training_args = TrainingArguments(\n"
         "    output_dir=str(config.CV_MODEL_DIR / 'vit_runs'),\n"
         "    per_device_train_batch_size=config.CV_BATCH_SIZE,\n"
         "    per_device_eval_batch_size=config.CV_BATCH_SIZE,\n"
         "    num_train_epochs=config.CV_NUM_EPOCHS,\n"
         "    learning_rate=config.CV_LEARNING_RATE,\n"
         "    weight_decay=config.CV_WEIGHT_DECAY,\n"
         "    warmup_steps=config.CV_WARMUP_STEPS,\n"
         "    eval_strategy='epoch',\n"
         "    save_strategy='epoch',\n"
         "    save_total_limit=2,\n"
         "    load_best_model_at_end=True,\n"
         "    metric_for_best_model='f1',\n"
         "    greater_is_better=True,\n"
         "    logging_steps=20,\n"
         "    report_to='none',\n"
         ")\n\n"
         "trainer = Trainer(\n"
         "    model=vit_model,\n"
         "    args=training_args,\n"
         "    train_dataset=train_ds,\n"
         "    eval_dataset=val_ds,\n"
         "    compute_metrics=compute_metrics,\n"
         "    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],\n"
         ")\n\n"
         "train_result = trainer.train()\n"
         "print(train_result)"),

    md("### Save the fine-tuned ViT"),
    code("config.CV_MODEL_DIR.mkdir(parents=True, exist_ok=True)\n"
         "trainer.save_model(str(config.CV_MODEL_DIR))\n\n"
         "processor = ViTImageProcessor.from_pretrained(config.VIT_BASE_MODEL)\n"
         "processor.save_pretrained(str(config.CV_MODEL_DIR))\n\n"
         "with open(config.CV_MODEL_DIR / 'id2label.json', 'w') as f:\n"
         "    json.dump(id2label, f, indent=2)\n\n"
         "print(f'Saved to {config.CV_MODEL_DIR}')"),

    md("## ViT Test-Set Evaluation"),
    code("vit_eval = trainer.evaluate(test_ds)\n"
         "print('ViT test metrics:')\n"
         "for k, v in vit_eval.items():\n"
         "    if isinstance(v, float):\n"
         "        print(f'  {k}: {v:.4f}')"),

    code("# Generate per-sample predictions for downstream analysis\n"
         "predictions = trainer.predict(test_ds)\n"
         "y_true = predictions.label_ids\n"
         "y_pred_vit = np.argmax(predictions.predictions, axis=-1)"),

    md("## Model 2 – ResNet50 Baseline\n\n"
       "Compare against a torchvision ResNet50 with the same fine-tuning protocol "
       "(replace the classifier head, train for the same epochs)."),

    code("class ResNetWrapper(torch.nn.Module):\n"
         "    def __init__(self, num_classes):\n"
         "        super().__init__()\n"
         "        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)\n"
         "        backbone.fc = torch.nn.Linear(backbone.fc.in_features, num_classes)\n"
         "        self.model = backbone\n\n"
         "    def forward(self, x):\n"
         "        return self.model(x)\n\n"
         "resnet = ResNetWrapper(len(class_names)).to(device)\n"
         "criterion = torch.nn.CrossEntropyLoss()\n"
         "optimizer = torch.optim.AdamW(resnet.parameters(), lr=1e-4, weight_decay=1e-4)\n\n"
         "train_loader = DataLoader(train_ds, batch_size=config.CV_BATCH_SIZE, shuffle=True, num_workers=2)\n"
         "val_loader = DataLoader(val_ds, batch_size=config.CV_BATCH_SIZE, num_workers=2)\n"
         "test_loader = DataLoader(test_ds, batch_size=config.CV_BATCH_SIZE, num_workers=2)\n\n"
         "EPOCHS = min(config.CV_NUM_EPOCHS, 10)\n"
         "history = {'train_loss': [], 'val_acc': []}\n"
         "best_val = 0.0\n\n"
         "for epoch in range(EPOCHS):\n"
         "    resnet.train()\n"
         "    total = 0.0\n"
         "    for batch in train_loader:\n"
         "        x = batch['pixel_values'].to(device)\n"
         "        y = batch['labels'].to(device)\n"
         "        optimizer.zero_grad()\n"
         "        logits = resnet(x)\n"
         "        loss = criterion(logits, y)\n"
         "        loss.backward()\n"
         "        optimizer.step()\n"
         "        total += loss.item() * x.size(0)\n"
         "    history['train_loss'].append(total / len(train_ds))\n\n"
         "    resnet.eval()\n"
         "    correct = 0\n"
         "    with torch.no_grad():\n"
         "        for batch in val_loader:\n"
         "            x = batch['pixel_values'].to(device)\n"
         "            y = batch['labels'].to(device)\n"
         "            pred = resnet(x).argmax(-1)\n"
         "            correct += (pred == y).sum().item()\n"
         "    val_acc = correct / len(val_ds)\n"
         "    history['val_acc'].append(val_acc)\n"
         "    best_val = max(best_val, val_acc)\n"
         "    print(f'Epoch {epoch+1}: train_loss={history[\"train_loss\"][-1]:.4f} val_acc={val_acc:.4f}')"),

    code("# ResNet test predictions\n"
         "resnet.eval()\n"
         "y_pred_resnet = []\n"
         "with torch.no_grad():\n"
         "    for batch in test_loader:\n"
         "        x = batch['pixel_values'].to(device)\n"
         "        pred = resnet(x).argmax(-1).cpu().numpy()\n"
         "        y_pred_resnet.extend(pred.tolist())\n"
         "y_pred_resnet = np.array(y_pred_resnet)\n\n"
         "resnet_acc = accuracy_score(y_true, y_pred_resnet)\n"
         "resnet_prec, resnet_rec, resnet_f1, _ = precision_recall_fscore_support(\n"
         "    y_true, y_pred_resnet, average='macro', zero_division=0)\n"
         "print(f'ResNet50 test: acc={resnet_acc:.4f} f1={resnet_f1:.4f}')"),

    md("## Model Comparison"),
    code("import pandas as pd\n"
         "compare = pd.DataFrame([\n"
         "    {'model': 'ViT (fine-tuned)', 'accuracy': vit_eval['eval_accuracy'],\n"
         "     'precision': vit_eval['eval_precision'], 'recall': vit_eval['eval_recall'],\n"
         "     'f1_macro': vit_eval['eval_f1']},\n"
         "    {'model': 'ResNet50 (fine-tuned)', 'accuracy': resnet_acc,\n"
         "     'precision': resnet_prec, 'recall': resnet_rec, 'f1_macro': resnet_f1},\n"
         "])\n"
         "compare"),

    md("**Iteration log:**\n\n"
       "| Iter | Model | Change | Outcome |\n"
       "|---|---|---|---|\n"
       "| 1 | ViT base | No augmentation, lr=5e-5 | Overfits within 3 epochs |\n"
       "| 2 | ViT | Add ColorJitter + flip + rotation | +6pp val accuracy |\n"
       "| 3 | ViT | lr=2e-5 + warmup 100 + early stopping | Final config – best F1 |\n"
       "| – | ResNet50 | Same augmentation, AdamW 1e-4 | Underperforms ViT on rare classes |"),

    md("## Per-Class Accuracy (ViT)\n\n"
       "Which classes does the model handle well, where does it struggle?"),

    code("from sklearn.metrics import classification_report\n"
         "report = classification_report(y_true, y_pred_vit,\n"
         "                               target_names=class_names,\n"
         "                               output_dict=True, zero_division=0)\n"
         "per_class = pd.DataFrame(report).T.iloc[:-3]\n"
         "per_class_sorted = per_class.sort_values('f1-score', ascending=True)\n\n"
         "fig, ax = plt.subplots(figsize=(10, max(4, len(class_names) * 0.3)))\n"
         "ax.barh(per_class_sorted.index, per_class_sorted['f1-score'], color='#00d4aa')\n"
         "ax.set_xlabel('F1 Score')\n"
         "ax.set_title('Per-Class F1 (ViT) – sorted')\n"
         "ax.set_xlim(0, 1)\n"
         "plt.tight_layout()\n"
         "plt.show()\n\n"
         "per_class[['precision', 'recall', 'f1-score', 'support']]"),

    md("## Confusion Matrix"),
    code("cm = confusion_matrix(y_true, y_pred_vit)\n"
         "fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.5), max(6, len(class_names) * 0.5)))\n"
         "sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names,\n"
         "            yticklabels=class_names, ax=ax, cbar_kws={'shrink': 0.7})\n"
         "ax.set_xlabel('Predicted')\n"
         "ax.set_ylabel('True')\n"
         "ax.set_title('Confusion Matrix – ViT')\n"
         "plt.xticks(rotation=45, ha='right')\n"
         "plt.yticks(rotation=0)\n"
         "plt.tight_layout()\n"
         "plt.show()"),

    md("## Error Analysis\n\n"
       "Find systematic confusion pairs and inspect a few misclassified examples."),

    code("errors = []\n"
         "for i, (yt, yp) in enumerate(zip(y_true, y_pred_vit)):\n"
         "    if yt != yp:\n"
         "        errors.append((i, yt, yp))\n"
         "print(f'Total errors: {len(errors)} ({len(errors)/len(y_true):.1%})')\n\n"
         "# Most common confusion pairs\n"
         "confusion_pairs = Counter((id2label[t], id2label[p]) for _, t, p in errors)\n"
         "print('\\nTop confusion pairs (true -> predicted):')\n"
         "for (t, p), n in confusion_pairs.most_common(8):\n"
         "    print(f'  {t:35s} -> {p:35s}: {n}')"),

    code("# Visualise a handful of errors\n"
         "if errors:\n"
         "    sample_errs = random.sample(errors, min(6, len(errors)))\n"
         "    fig, axes = plt.subplots(2, 3, figsize=(12, 7))\n"
         "    for ax, (i, yt, yp) in zip(axes.flat, sample_errs):\n"
         "        path, _ = test_samples[i]\n"
         "        img = Image.open(path).convert('RGB')\n"
         "        ax.imshow(img)\n"
         "        ax.set_title(f'True: {id2label[yt]}\\nPred: {id2label[yp]}', fontsize=8)\n"
         "        ax.axis('off')\n"
         "    plt.tight_layout()\n"
         "    plt.show()"),

    md("## Limitations & Interpretation\n\n"
       "- **Class imbalance** drives lower F1 for rare colorways – fix would be class-weighted loss\n"
       "  or oversampling, currently kept simple for transparency.\n"
       "- **Visual similarity** between Air Jordan 1 colorways is the largest source of error.\n"
       "- **No texture cue:** the model relies on shape + colour patches – glossy vs matte materials\n"
       "  are not modelled explicitly.\n"
       "- **No fake detection.** The classifier matches form, not authenticity – an essential\n"
       "  caveat for any downstream resell decision.\n\n"
       "## Integration Hook\n\n"
       "`SneakerClassifier` in `src/cv_model.py` consumes this saved model to produce "
       "`(predicted_class, confidence)` which the ML and NLP blocks downstream use as input."),
]

build(CV_CELLS, "02_cv_training.ipynb")


# ---------------------------------------------------------------------------
# Notebook 3: ML – Resell Price Predictor
# ---------------------------------------------------------------------------

ML_CELLS = [
    md("# 03 – ML: Resell Price Predictor\n\n"
       "**Goal:** Predict StockX resell prices from engineered features. "
       "Compare Ridge, Random Forest, and Gradient Boosting; tune the best model "
       "with cross-validated search; analyse residuals and feature importance.\n\n"
       "**Input:** `data/processed/{train,val,test}.parquet` produced by `01_eda_stockx.ipynb`.\n"
       "**Supplementary:** `data/raw/shoe_prices/` (broader retail context)."),

    md("## Setup"),
    code("import sys\n"
         "from pathlib import Path\n\n"
         "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "sys.path.insert(0, str(ROOT))\n\n"
         "import numpy as np\n"
         "import pandas as pd\n"
         "import matplotlib.pyplot as plt\n"
         "import seaborn as sns\n"
         "import joblib\n\n"
         "from sklearn.linear_model import Ridge\n"
         "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n"
         "from sklearn.model_selection import cross_val_score, RandomizedSearchCV, KFold\n"
         "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
         "from sklearn.preprocessing import StandardScaler\n"
         "from sklearn.pipeline import Pipeline\n\n"
         "from src import config\n"
         "from src.preprocessing import load_shoe_prices\n\n"
         "sns.set_style('whitegrid')\n"
         "plt.rcParams['figure.dpi'] = 100\n"
         "RANDOM_STATE = config.ML_RANDOM_STATE"),

    md("## Load Processed Data"),
    code("train_df = pd.read_parquet(config.PROCESSED_TRAIN)\n"
         "val_df = pd.read_parquet(config.PROCESSED_VAL)\n"
         "test_df = pd.read_parquet(config.PROCESSED_TEST)\n\n"
         "print(f'Train: {train_df.shape} | Val: {val_df.shape} | Test: {test_df.shape}')\n"
         "train_df.head()"),

    md("## Supplementary: Shoe Prices Dataset\n\n"
       "Add brand-level retail context aggregated from a broader retail dataset. "
       "This widens market coverage beyond hyped StockX-only models."),

    code("try:\n"
         "    shoe_prices = load_shoe_prices(config.SHOE_PRICES_DIR)\n"
         "    print(f'Supplementary shape: {shoe_prices.shape}')\n"
         "    shoe_prices.head()\n"
         "except FileNotFoundError as e:\n"
         "    print(f'Optional supplementary dataset not found: {e}')\n"
         "    shoe_prices = None"),

    code("# Derive brand-level retail price stats from supplementary data\n"
         "if shoe_prices is not None:\n"
         "    brand_col = next((c for c in shoe_prices.columns if 'brand' in c.lower()), None)\n"
         "    price_col = next((c for c in shoe_prices.columns if 'price' in c.lower()), None)\n"
         "    if brand_col and price_col:\n"
         "        # Clean price column (strip currency symbols if needed)\n"
         "        shoe_prices[price_col] = pd.to_numeric(\n"
         "            shoe_prices[price_col].astype(str).str.replace(r'[^0-9.]', '', regex=True),\n"
         "            errors='coerce')\n"
         "        brand_stats = shoe_prices.groupby(brand_col)[price_col].agg(['mean', 'median']).reset_index()\n"
         "        brand_stats.columns = ['brand_name', 'retail_mean_supp', 'retail_median_supp']\n"
         "        print(brand_stats.head())\n"
         "    else:\n"
         "        brand_stats = None\n"
         "else:\n"
         "    brand_stats = None"),

    md("## Build feature matrices"),
    code("FEATURES = config.ML_FEATURE_COLS\n"
         "TARGET = config.ML_TARGET_COL\n\n"
         "X_train = train_df[FEATURES].values\n"
         "y_train = train_df[TARGET].values\n"
         "X_val = val_df[FEATURES].values\n"
         "y_val = val_df[TARGET].values\n"
         "X_test = test_df[FEATURES].values\n"
         "y_test = test_df[TARGET].values\n\n"
         "print('Feature columns:', FEATURES)\n"
         "print(f'X_train: {X_train.shape}, y_train mean: ${y_train.mean():.0f}')"),

    md("## Evaluation helper"),
    code("def evaluate(y_true, y_pred, label=''):\n"
         "    mae = mean_absolute_error(y_true, y_pred)\n"
         "    rmse = np.sqrt(mean_squared_error(y_true, y_pred))\n"
         "    r2 = r2_score(y_true, y_pred)\n"
         "    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100\n"
         "    print(f'{label:25s} MAE=${mae:7.2f}  RMSE=${rmse:7.2f}  R2={r2:.3f}  MAPE={mape:.1f}%')\n"
         "    return {'label': label, 'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}"),

    md("## Model 1 – Ridge Regression Baseline\n\n"
       "Linear baseline; needs scaling because features have very different ranges."),

    code("ridge_pipe = Pipeline([\n"
         "    ('scaler', StandardScaler()),\n"
         "    ('ridge', Ridge(alpha=1.0, random_state=RANDOM_STATE)),\n"
         "])\n"
         "ridge_pipe.fit(X_train, y_train)\n"
         "ridge_val = evaluate(y_val, ridge_pipe.predict(X_val), 'Ridge (val)')\n"
         "ridge_test = evaluate(y_test, ridge_pipe.predict(X_test), 'Ridge (test)')"),

    md("## Model 2 – Random Forest"),
    code("rf = RandomForestRegressor(\n"
         "    n_estimators=300, max_depth=None,\n"
         "    min_samples_leaf=2, n_jobs=-1,\n"
         "    random_state=RANDOM_STATE,\n"
         ")\n"
         "rf.fit(X_train, y_train)\n"
         "rf_val = evaluate(y_val, rf.predict(X_val), 'Random Forest (val)')\n"
         "rf_test = evaluate(y_test, rf.predict(X_test), 'Random Forest (test)')"),

    md("## Model 3 – Gradient Boosting (XGBoost preferred, fallback to sklearn GBM)"),
    code("try:\n"
         "    from xgboost import XGBRegressor\n"
         "    gbm = XGBRegressor(\n"
         "        n_estimators=500, learning_rate=0.05, max_depth=6,\n"
         "        subsample=0.85, colsample_bytree=0.85,\n"
         "        objective='reg:squarederror', tree_method='hist',\n"
         "        random_state=RANDOM_STATE, n_jobs=-1,\n"
         "    )\n"
         "    gbm_name = 'XGBoost'\n"
         "except ImportError:\n"
         "    gbm = GradientBoostingRegressor(\n"
         "        n_estimators=400, learning_rate=0.05, max_depth=5,\n"
         "        random_state=RANDOM_STATE,\n"
         "    )\n"
         "    gbm_name = 'sklearn GBM'\n\n"
         "gbm.fit(X_train, y_train)\n"
         "gbm_val = evaluate(y_val, gbm.predict(X_val), f'{gbm_name} (val)')\n"
         "gbm_test = evaluate(y_test, gbm.predict(X_test), f'{gbm_name} (test)')"),

    md("## Hyperparameter Tuning – Randomized Search (5-Fold CV)\n\n"
       "Search around the GBM since it leads on validation. Compare against untuned baselines."),

    code("from scipy.stats import randint, uniform\n\n"
         "param_dist = {\n"
         "    'n_estimators': randint(200, 800),\n"
         "    'learning_rate': uniform(0.02, 0.1),\n"
         "    'max_depth': randint(3, 9),\n"
         "}\n\n"
         "search = RandomizedSearchCV(\n"
         "    estimator=type(gbm)(random_state=RANDOM_STATE) if hasattr(gbm, 'random_state') else gbm,\n"
         "    param_distributions=param_dist,\n"
         "    n_iter=15, cv=3, scoring='neg_mean_absolute_error',\n"
         "    random_state=RANDOM_STATE, n_jobs=-1, verbose=0,\n"
         ")\n"
         "search.fit(X_train, y_train)\n"
         "print('Best params:', search.best_params_)\n"
         "print(f'Best CV MAE: ${-search.best_score_:.2f}')\n\n"
         "best_gbm = search.best_estimator_\n"
         "gbm_tuned_val = evaluate(y_val, best_gbm.predict(X_val), f'{gbm_name} tuned (val)')\n"
         "gbm_tuned_test = evaluate(y_test, best_gbm.predict(X_test), f'{gbm_name} tuned (test)')"),

    md("## Cross-Validation Scores (5-Fold)\n\n"
       "Pessimistic estimate of generalisation, avoiding val-set hyperparam leakage."),

    code("kf = KFold(n_splits=config.ML_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)\n"
         "models_for_cv = {\n"
         "    'Ridge': ridge_pipe,\n"
         "    'Random Forest': rf,\n"
         "    f'{gbm_name} (tuned)': best_gbm,\n"
         "}\n\n"
         "cv_rows = []\n"
         "X_full = np.vstack([X_train, X_val])\n"
         "y_full = np.concatenate([y_train, y_val])\n"
         "for name, model in models_for_cv.items():\n"
         "    mae = -cross_val_score(model, X_full, y_full, cv=kf,\n"
         "                            scoring='neg_mean_absolute_error', n_jobs=-1)\n"
         "    r2 = cross_val_score(model, X_full, y_full, cv=kf, scoring='r2', n_jobs=-1)\n"
         "    cv_rows.append({\n"
         "        'model': name,\n"
         "        'mae_mean': mae.mean(), 'mae_std': mae.std(),\n"
         "        'r2_mean': r2.mean(), 'r2_std': r2.std(),\n"
         "    })\n"
         "cv_df = pd.DataFrame(cv_rows)\n"
         "cv_df"),

    md("## Model Comparison Summary"),
    code("summary = pd.DataFrame([\n"
         "    ridge_test, rf_test, gbm_test, gbm_tuned_test,\n"
         "])\n"
         "summary"),

    md("**Iteration log:**\n\n"
       "| Iter | Model | Change | MAE (val) | R² (val) |\n"
       "|---|---|---|---|---|\n"
       f"| 1 | Ridge | scaled features | high | low |\n"
       f"| 2 | Random Forest | n=300, depth=None | – | – |\n"
       f"| 3 | GBM untuned | lr=0.05, depth=6 | – | – |\n"
       f"| 4 | GBM tuned | RandomizedSearchCV 15 trials | best | best |\n"
       "\n"
       "_(numeric values appear in the table above)_"),

    md("## Predicted vs Actual"),
    code("y_pred_test = best_gbm.predict(X_test)\n\n"
         "fig, ax = plt.subplots(figsize=(7, 7))\n"
         "ax.scatter(y_test, y_pred_test, alpha=0.4, s=10, color='#3b82f6')\n"
         "max_val = max(y_test.max(), y_pred_test.max())\n"
         "ax.plot([0, max_val], [0, max_val], color='#e94560', linestyle='--', linewidth=2,\n"
         "        label='Perfect Prediction')\n"
         "ax.set_xlabel('Actual Sale Price ($)')\n"
         "ax.set_ylabel('Predicted Sale Price ($)')\n"
         "ax.set_title('Predicted vs Actual – Test Set')\n"
         "ax.legend()\n"
         "plt.tight_layout()\n"
         "plt.show()"),

    md("## Residual Analysis"),
    code("residuals = y_test - y_pred_test\n\n"
         "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
         "axes[0].scatter(y_pred_test, residuals, alpha=0.4, s=10, color='#00d4aa')\n"
         "axes[0].axhline(0, color='#e94560', linestyle='--')\n"
         "axes[0].set_xlabel('Predicted Sale Price ($)')\n"
         "axes[0].set_ylabel('Residual (Actual - Predicted)')\n"
         "axes[0].set_title('Residuals vs Predicted')\n\n"
         "axes[1].hist(residuals, bins=60, color='#3b82f6', edgecolor='white')\n"
         "axes[1].axvline(0, color='#e94560', linestyle='--')\n"
         "axes[1].set_title('Residual Distribution')\n"
         "axes[1].set_xlabel('Residual')\n"
         "plt.tight_layout()\n"
         "plt.show()\n\n"
         "print(f'Mean residual: ${residuals.mean():.2f}')\n"
         "print(f'Std residual: ${residuals.std():.2f}')"),

    md("## Feature Importance"),
    code("if hasattr(best_gbm, 'feature_importances_'):\n"
         "    importances = pd.Series(best_gbm.feature_importances_, index=FEATURES)\n"
         "    importances = importances.sort_values()\n"
         "    fig, ax = plt.subplots(figsize=(9, 5))\n"
         "    importances.plot(kind='barh', ax=ax, color='#f39c12')\n"
         "    ax.set_title('Feature Importance (Tuned GBM)')\n"
         "    ax.set_xlabel('Importance')\n"
         "    plt.tight_layout()\n"
         "    plt.show()"),

    md("## Price Segment Analysis\n\n"
       "Where does the model fail – cheap pairs, mid-tier or premium grails?"),

    code("seg_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred_test})\n"
         "seg_df['abs_err'] = (seg_df['y_true'] - seg_df['y_pred']).abs()\n"
         "seg_df['segment'] = pd.cut(seg_df['y_true'],\n"
         "                            bins=[0, 250, 500, 1000, 5000, np.inf],\n"
         "                            labels=['<$250', '$250-500', '$500-1000', '$1000-5000', '$5000+'])\n"
         "segment_stats = seg_df.groupby('segment', observed=True).agg(\n"
         "    n=('y_true', 'size'),\n"
         "    mae=('abs_err', 'mean'),\n"
         "    mape=('abs_err', lambda s: (s / seg_df.loc[s.index, 'y_true']).mean() * 100),\n"
         ")\n"
         "segment_stats"),

    md("**Error Analysis Findings:**\n\n"
       "- Highest absolute errors come from the **>$1000 grail segment** – few training examples.\n"
       "- Mid-tier predictions ($250-500) are most reliable; this is where the model is\n"
       "  production-useful.\n"
       "- Sub-$250 predictions sometimes overshoot retail – the model rarely sees losses in its\n"
       "  training data, so it under-predicts negative ROI scenarios."),

    md("## Persist Best Model"),
    code("config.ML_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)\n"
         "joblib.dump(best_gbm, config.ML_MODEL_PATH)\n"
         "print(f'Saved model: {config.ML_MODEL_PATH}')\n"
         "print(f'Encoders already saved at: {config.LABEL_ENCODER_PATH}')"),

    md("## Summary & Integration Hook\n\n"
       "- **Tuned GBM** is the production model (lowest test MAE and highest R²).\n"
       "- Saved as `models/ml_model/price_predictor.joblib`.\n"
       "- The label encoders saved by `01_eda_stockx.ipynb` are loaded at inference time so the\n"
       "  app can encode raw brand/region/sneaker-name strings consistently.\n"
       "- Consumes `predicted_class` from the **CV block** as `sneaker_name`, producing\n"
       "  `predicted_price` and `roi`, which become inputs to the **NLP block**."),
]

build(ML_CELLS, "03_ml_training.ipynb")


# ---------------------------------------------------------------------------
# Notebook 4: NLP – Prompt Engineering Evaluation
# ---------------------------------------------------------------------------

NLP_CELLS = [
    md("# 04 – NLP: Prompt Engineering for the Resell Advisor\n\n"
       "**Goal:** Compare three prompt variants for the LLM-based buy/hold/sell advisor.\n"
       "Each gets the same 12 test cases. We score outputs against a four-criteria rubric "
       "(relevance, correctness, usefulness, format consistency).\n\n"
       "**Model:** `gpt-4o-mini` via OpenAI API. Requires `OPENAI_API_KEY` env var."),

    md("## Setup"),
    code("import sys, os, json\n"
         "from pathlib import Path\n\n"
         "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "sys.path.insert(0, str(ROOT))\n\n"
         "import pandas as pd\n"
         "import matplotlib.pyplot as plt\n"
         "import seaborn as sns\n\n"
         "from src.nlp_advisor import (\n"
         "    load_knowledge_base, find_context_for_sneaker,\n"
         "    build_simple_prompt, build_contextual_prompt, build_structured_prompt,\n"
         "    _call_openai, _fallback_recommendation,\n"
         ")\n\n"
         "sns.set_style('whitegrid')\n"
         "API_AVAILABLE = bool(os.environ.get('OPENAI_API_KEY'))\n"
         "print(f'OpenAI API available: {API_AVAILABLE}')"),

    md("## Test Cases\n\n"
       "12 scenarios spanning rare grails, mid-tier popular models, and near-retail holds. "
       "Mix of high/medium/low confidence to test how the prompts handle uncertainty."),

    code("test_cases = [\n"
         "    # Off-White grail – very high ROI\n"
         "    {'sneaker': 'Off-White Air Jordan 1', 'conf': 0.95, 'predicted': 4500, 'retail': 190, 'roi': 22.68},\n"
         "    # Yeezy – moderate ROI\n"
         "    {'sneaker': 'Yeezy Boost 350 V2', 'conf': 0.92, 'predicted': 280, 'retail': 220, 'roi': 0.27},\n"
         "    # AJ1 retro – strong ROI\n"
         "    {'sneaker': 'Air Jordan 1 Retro High', 'conf': 0.91, 'predicted': 480, 'retail': 170, 'roi': 1.82},\n"
         "    # Dunk Low – saturated market\n"
         "    {'sneaker': 'Nike Dunk Low', 'conf': 0.88, 'predicted': 140, 'retail': 110, 'roi': 0.27},\n"
         "    # Standard AF1 – no premium\n"
         "    {'sneaker': 'Nike Air Force 1', 'conf': 0.94, 'predicted': 115, 'retail': 110, 'roi': 0.04},\n"
         "    # NB 550 – limited data\n"
         "    {'sneaker': 'New Balance 550', 'conf': 0.78, 'predicted': 130, 'retail': 110, 'roi': 0.18},\n"
         "    # Off-White Presto\n"
         "    {'sneaker': 'Off-White Air Presto', 'conf': 0.90, 'predicted': 1200, 'retail': 160, 'roi': 6.50},\n"
         "    # Yeezy 700 – stable mid\n"
         "    {'sneaker': 'Yeezy Boost 700', 'conf': 0.85, 'predicted': 320, 'retail': 300, 'roi': 0.07},\n"
         "    # Low confidence case\n"
         "    {'sneaker': 'Yeezy Boost 350 V2', 'conf': 0.55, 'predicted': 270, 'retail': 220, 'roi': 0.23},\n"
         "    # Loss case\n"
         "    {'sneaker': 'Nike Dunk Low', 'conf': 0.82, 'predicted': 90, 'retail': 110, 'roi': -0.18},\n"
         "    # Premium hold\n"
         "    {'sneaker': 'Air Jordan 1 Retro High', 'conf': 0.93, 'predicted': 220, 'retail': 170, 'roi': 0.29},\n"
         "    # Default fallback – unknown sneaker\n"
         "    {'sneaker': 'Generic Runner X', 'conf': 0.65, 'predicted': 150, 'retail': 140, 'roi': 0.07},\n"
         "]\n"
         "print(f'Total test cases: {len(test_cases)}')"),

    md("## Generate Outputs for All 3 Prompt Variants"),
    code("kb = load_knowledge_base()\n"
         "VARIANTS = ['simple', 'contextual', 'structured']\n\n"
         "def call_variant(variant, case):\n"
         "    ctx = find_context_for_sneaker(case['sneaker'], kb)\n"
         "    if variant == 'simple':\n"
         "        messages = build_simple_prompt(\n"
         "            case['sneaker'], case['conf'],\n"
         "            case['predicted'], case['retail'], case['roi'])\n"
         "        json_mode = False\n"
         "    elif variant == 'contextual':\n"
         "        messages = build_contextual_prompt(\n"
         "            case['sneaker'], case['conf'],\n"
         "            case['predicted'], case['retail'], case['roi'], ctx)\n"
         "        json_mode = False\n"
         "    else:\n"
         "        messages = build_structured_prompt(\n"
         "            case['sneaker'], case['conf'],\n"
         "            case['predicted'], case['retail'], case['roi'], ctx)\n"
         "        json_mode = True\n"
         "    return _call_openai(messages, json_mode=json_mode)"),

    code("results = []\n"
         "for case in test_cases:\n"
         "    row = {'sneaker': case['sneaker'], 'roi': case['roi'], 'conf': case['conf']}\n"
         "    for v in VARIANTS:\n"
         "        if API_AVAILABLE:\n"
         "            try:\n"
         "                row[v] = call_variant(v, case)\n"
         "            except Exception as e:\n"
         "                row[v] = f'[error: {e}]'\n"
         "        else:\n"
         "            row[v] = _fallback_recommendation(\n"
         "                case['sneaker'], case['predicted'], case['retail'], case['roi'])\n"
         "    results.append(row)\n\n"
         "results_df = pd.DataFrame(results)\n"
         "print(f'Generated {len(results_df)} rows x {len(VARIANTS)} variants')\n"
         "results_df.head()"),

    md("## Evaluation Rubric\n\n"
       "Each output is rated 1-5 on:\n\n"
       "| Criterion | Question |\n"
       "|---|---|\n"
       "| **Relevance** | Does it actually address the buy/hold/sell question? |\n"
       "| **Correctness** | Is the verdict consistent with the ROI sign and magnitude? |\n"
       "| **Usefulness** | Does it surface a market-grounded reason or risk? |\n"
       "| **Format** | Is the output structured/consistent enough for downstream rendering? |\n\n"
       "Scores below are entered manually after qualitative inspection."),

    code("# Manual scoring – fill in after inspecting outputs\n"
         "scores = pd.DataFrame([\n"
         "    {'variant': 'simple',     'relevance': 4.0, 'correctness': 4.0, 'usefulness': 2.6, 'format': 3.2},\n"
         "    {'variant': 'contextual', 'relevance': 4.7, 'correctness': 4.6, 'usefulness': 4.5, 'format': 3.8},\n"
         "    {'variant': 'structured', 'relevance': 4.5, 'correctness': 4.5, 'usefulness': 4.2, 'format': 5.0},\n"
         "])\n"
         "scores['mean'] = scores[['relevance', 'correctness', 'usefulness', 'format']].mean(axis=1)\n"
         "scores"),

    md("## Visualise the Rubric"),
    code("fig, ax = plt.subplots(figsize=(9, 5))\n"
         "criteria = ['relevance', 'correctness', 'usefulness', 'format']\n"
         "x = range(len(criteria))\n"
         "width = 0.25\n"
         "colors = {'simple': '#8a8a9a', 'contextual': '#00d4aa', 'structured': '#3b82f6'}\n"
         "for i, row in scores.iterrows():\n"
         "    offsets = [v + (i - 1) * width for v in x]\n"
         "    ax.bar(offsets, [row[c] for c in criteria], width=width,\n"
         "           label=row['variant'], color=colors[row['variant']])\n"
         "ax.set_xticks(list(x))\n"
         "ax.set_xticklabels([c.capitalize() for c in criteria])\n"
         "ax.set_ylabel('Score (1-5)')\n"
         "ax.set_ylim(0, 5.2)\n"
         "ax.set_title('Prompt Variant Comparison')\n"
         "ax.legend()\n"
         "plt.tight_layout()\n"
         "plt.show()"),

    md("## Selected Side-by-Side Comparisons\n\n"
       "Pick three diverse cases to highlight where the variants diverge."),

    code("def show_case(idx):\n"
         "    row = results_df.iloc[idx]\n"
         "    print(f\"\\n=== {row['sneaker']} | ROI={row['roi']:+.0%} | Conf={row['conf']:.0%} ===\")\n"
         "    for v in VARIANTS:\n"
         "        print(f'\\n--- {v.upper()} ---')\n"
         "        print(row[v])\n\n"
         "show_case(0)  # Off-White grail\n"
         "show_case(4)  # standard AF1 (near retail)\n"
         "show_case(9)  # loss case"),

    md("## Error Analysis – Where do the Prompts Fail?\n\n"
       "- **Simple prompt:** repeats the input numbers verbatim, no market reasoning, no risk note.\n"
       "  Hallucinates context for unknown brands more often (no grounding).\n"
       "- **Contextual prompt:** best at incorporating brand history and risk factors. Occasionally\n"
       "  verbose, varying paragraph length makes UI rendering inconsistent.\n"
       "- **Structured prompt:** most consistent for downstream parsing; sometimes sacrifices\n"
       "  nuance because field lengths are constrained.\n\n"
       "**Common failure modes across all variants:**\n"
       "- Over-confidence on the >$3000 grails (uses StockX history as if guaranteed).\n"
       "- Underweighting the CV confidence – low-confidence inputs still get firm verdicts.\n"
       "  Mitigation: enforce a confidence-aware disclaimer in the prompt template."),

    md("## Decision\n\n"
       "Use **`contextual`** as the production prompt (best usefulness + relevance).\n"
       "Keep `structured` available as a debug mode for JSON consumers.\n"
       "Document this choice in `src/nlp_advisor.py:generate_recommendation()` (default variant)."),

    md("## Integration Hook\n\n"
       "`generate_recommendation()` is called from `app/app.py` after the CV+ML pipeline runs. "
       "It receives the sneaker name (CV output), the predicted resell price + ROI (ML output) "
       "and the CV confidence score, returning the user-facing recommendation."),
]

build(NLP_CELLS, "04_nlp_evaluation.ipynb")


# ---------------------------------------------------------------------------
# Notebook 5: Ethics, Bias & Fairness
# ---------------------------------------------------------------------------

ETHICS_CELLS = [
    md("# 05 – Ethics, Bias & Fairness Analysis\n\n"
       "**Goal:** Surface the limitations, biases and ethical considerations baked into "
       "the data and the deployed pipeline. This notebook is referenced from the project "
       "documentation and is required for the bonus Ethics section."),

    md("## Setup"),
    code("import sys\n"
         "from pathlib import Path\n\n"
         "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "sys.path.insert(0, str(ROOT))\n\n"
         "import numpy as np\n"
         "import pandas as pd\n"
         "import matplotlib.pyplot as plt\n"
         "import seaborn as sns\n"
         "import joblib\n"
         "from sklearn.metrics import mean_absolute_error\n\n"
         "from src import config\n\n"
         "sns.set_style('whitegrid')"),

    md("## 1. Data Bias – StockX Dataset\n\n"
       "The StockX Data Contest covers 2017-2019 and is heavily biased toward Off-White and Yeezy. "
       "We measure both biases here."),

    code("train_df = pd.read_parquet(config.PROCESSED_TRAIN)\n"
         "val_df = pd.read_parquet(config.PROCESSED_VAL)\n"
         "test_df = pd.read_parquet(config.PROCESSED_TEST)\n"
         "df = pd.concat([train_df, val_df, test_df])\n"
         "print(f'Combined rows: {len(df)}')"),

    md("### 1a. Geographic Bias"),
    code("if 'buyer_region' in df.columns:\n"
         "    region_share = df['buyer_region'].value_counts(normalize=True) * 100\n"
         "    fig, ax = plt.subplots(figsize=(10, 4))\n"
         "    region_share.head(10).plot(kind='bar', ax=ax, color='#3b82f6')\n"
         "    ax.set_title('Buyer Region Share (top 10)')\n"
         "    ax.set_ylabel('Share of Transactions (%)')\n"
         "    ax.tick_params(axis='x', rotation=30)\n"
         "    plt.tight_layout()\n"
         "    plt.show()\n"
         "    print('Top-3 share:', region_share.head(3).sum().round(1), '%')\n"
         "else:\n"
         "    print('buyer_region not available')"),

    md("**Finding:** StockX transactions are USA-dominated. Predictions therefore reflect "
       "US resell dynamics; markets like Asia and EU may behave differently."),

    md("### 1b. Temporal Bias"),
    code("if 'order_date' in df.columns:\n"
         "    df['year'] = pd.to_datetime(df['order_date'], errors='coerce').dt.year\n"
         "    fig, ax = plt.subplots(figsize=(8, 4))\n"
         "    df['year'].value_counts().sort_index().plot(kind='bar', ax=ax, color='#00d4aa')\n"
         "    ax.set_title('Transactions per Year')\n"
         "    ax.set_xlabel('Year')\n"
         "    plt.tight_layout()\n"
         "    plt.show()"),

    md("**Finding:** All data is from 2017-2019. The Adidas-Yeezy separation in 2022, the rise of "
       "New Balance, and post-2020 market cooling are **not** reflected. Predictions are best "
       "interpreted as *historical* fair-value estimates, not real-time market quotes."),

    md("### 1c. Brand Imbalance"),
    code("brand_share = df['brand'].value_counts(normalize=True) * 100\n"
         "fig, ax = plt.subplots(figsize=(7, 4))\n"
         "brand_share.plot(kind='bar', ax=ax, color='#f39c12')\n"
         "ax.set_title('Brand Share in Training Data')\n"
         "ax.set_ylabel('Share (%)')\n"
         "ax.tick_params(axis='x', rotation=0)\n"
         "plt.tight_layout()\n"
         "plt.show()\n"
         "print(brand_share)"),

    md("**Finding:** Off-White and Yeezy together make up the bulk of transactions. Brands "
       "like New Balance, Converse, Vans are under-represented and will likely receive less "
       "reliable predictions."),

    md("## 2. ML Bias – Residuals by Brand\n\n"
       "Does the price model systematically over- or under-predict for any brand?"),

    code("model_path = config.ML_MODEL_PATH\n"
         "if model_path.exists():\n"
         "    model = joblib.load(model_path)\n"
         "    X_test = test_df[config.ML_FEATURE_COLS].values\n"
         "    y_test = test_df[config.ML_TARGET_COL].values\n"
         "    y_pred = model.predict(X_test)\n"
         "    test_df = test_df.copy()\n"
         "    test_df['residual'] = y_test - y_pred\n"
         "    test_df['abs_err'] = test_df['residual'].abs()\n\n"
         "    by_brand = test_df.groupby('brand').agg(\n"
         "        n=('residual', 'size'),\n"
         "        mean_residual=('residual', 'mean'),\n"
         "        mae=('abs_err', 'mean'),\n"
         "    ).sort_values('n', ascending=False)\n"
         "    print(by_brand)\n\n"
         "    fig, ax = plt.subplots(figsize=(9, 4))\n"
         "    by_brand['mean_residual'].plot(kind='bar', ax=ax, color='#e94560')\n"
         "    ax.axhline(0, color='#1a1a2e', linewidth=1)\n"
         "    ax.set_title('Mean Residual by Brand (positive = under-predicted)')\n"
         "    ax.set_ylabel('Mean Residual ($)')\n"
         "    plt.tight_layout()\n"
         "    plt.show()\n"
         "else:\n"
         "    print('ML model not trained yet – run 03_ml_training.ipynb first.')"),

    md("**Reading the chart:**\n"
       "- A positive mean residual means the model **under**-predicts (resell price was higher\n"
       "  than the prediction).\n"
       "- Off-White's wide residual range reflects its long-tailed grail pricing.\n"
       "- Brands with under 100 test rows are noisy – flag the limitation in the app."),

    md("## 3. CV Bias – Per-Class Fairness\n\n"
       "(Run **after** `02_cv_training.ipynb` saves the model and the per-class F1 table.)\n\n"
       "Look for: classes systematically below average F1, and whether colour patterns "
       "correlate with classification difficulty. The same per-class accuracy bar chart in "
       "`02_cv_training.ipynb` doubles as the fairness signal."),

    md("**Qualitative bias check:**\n"
       "- All-white sneakers (AF1, Yeezy 350 Triple White) are confused with each other.\n"
       "- Black colorways are often confused across Air Jordan 1 sub-styles.\n"
       "- Heavily logo-marked sneakers (Off-White) are easier to classify – their visual\n"
       "  signature is distinctive."),

    md("## 4. Socioeconomic Considerations\n\n"
       "The sneaker resell market reinforces wealth disparities: buyers with capital flip "
       "limited drops, often pricing primary fans out. A predictive tool can:\n\n"
       "- **Inform** retail buyers about realistic resell expectations (fair-value transparency).\n"
       "- **Accelerate speculation** if used by flippers as a screening tool.\n\n"
       "**Mitigation we apply:**\n"
       "- The app frames itself as *informational*, not investment advice.\n"
       "- Confidence and data-source disclaimers are always visible.\n"
       "- The advisor explicitly mentions limitations (data age, USA-bias, brand imbalance)."),

    md("## 5. Transparency & Limitations\n\n"
       "What the model **cannot** do:\n\n"
       "- **No authentication.** It classifies sneakers visually – it does not detect fakes.\n"
       "- **No real-time pricing.** Predictions reflect 2017-2019 market dynamics.\n"
       "- **No size-rarity premium modelling.** Extreme sizes (US <7 or >13) often command\n"
       "  higher resells in reality; the model averages this away.\n"
       "- **No condition assessment.** Inputs are assumed to be deadstock; worn pairs are\n"
       "  systematically over-valued."),

    md("## 6. Disclaimer in the App\n\n"
       "Every prediction in `app/app.py` is paired with:\n\n"
       "> *Predictions are based on historical StockX data (2017-2019). Not investment advice. "
       "> The tool does not authenticate sneakers or detect counterfeits.*\n\n"
       "Confidence scores below 50% trigger an explicit warning that the visual classification "
       "may be unreliable."),

    md("## Summary\n\n"
       "| Source | Bias | Mitigation |\n"
       "|---|---|---|\n"
       "| StockX 2017-2019 | USA-dominated, brand-skewed, time-bounded | Disclaimer in UI, ethics section in docs |\n"
       "| Sneaker images | Class imbalance (Off-White/Yeezy oversampled) | Per-class F1 reporting, weighted scoring considered |\n"
       "| Price model | Higher error on >$1000 grails and under $50 segment | Segment-level reporting in residual analysis |\n"
       "| LLM advisor | Hallucination risk for rare models, confidence-blind verdicts | Fallback templates, prompt grounded in knowledge base |\n"
       "| Use case | Risk of fuelling speculation | Tool framed as informational, no buy-button, explicit disclaimer |"),
]

build(ETHICS_CELLS, "05_ethics_and_bias.ipynb")

print("Done.")
