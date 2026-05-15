# Data Download Instructions

Raw datasets are **not** tracked in git. Download each dataset manually and place it in the corresponding subdirectory.

---

## 1. Sneaker Images (CV Dataset)

**Source:** Popular Sneakers Classification – Kaggle  
**URL:** https://www.kaggle.com/datasets/nikolasgegenava/sneakers-classification/data  
**Target path:** `data/raw/sneakers_images/`

```bash
kaggle datasets download -d nikolasgegenava/sneakers-classification -p data/raw/
unzip data/raw/sneakers-classification.zip -d data/raw/sneakers_images/
```

Expected structure after unzip:
```
data/raw/sneakers_images/
├── train/
│   ├── Air Jordan 1 Retro High/
│   ├── Yeezy Boost 350/
│   └── ...
└── test/
    └── ...
```

---

## 2. StockX Resell Data (ML Dataset)

**Source:** StockX Data Contest – Kaggle  
**URL:** https://www.kaggle.com/datasets/hudsonstuck/stockx-data-contest  
**Target path:** `data/raw/stockx/`

```bash
kaggle datasets download -d hudsonstuck/stockx-data-contest -p data/raw/stockx/
unzip data/raw/stockx/stockx-data-contest.zip -d data/raw/stockx/
```

Expected file: `data/raw/stockx/StockX-Data-Contest-2019-3.xlsx` or similar CSV.

---

## 3. Shoe Prices Dataset (Supplementary ML)

**Source:** Shoe Prices Dataset – Kaggle  
**URL:** https://www.kaggle.com/datasets/rkiattisak/shoe-prices-dataset  
**Target path:** `data/raw/shoe_prices/`

```bash
kaggle datasets download -d rkiattisak/shoe-prices-dataset -p data/raw/shoe_prices/
unzip data/raw/shoe_prices/shoe-prices-dataset.zip -d data/raw/shoe_prices/
```

---

## Kaggle API Setup (one-time)

```bash
pip install kaggle
# Place your kaggle.json at ~/.kaggle/kaggle.json
# Get it from: https://www.kaggle.com/settings/account → API → Create New Token
chmod 600 ~/.kaggle/kaggle.json
```

---

## Verify Downloads

After downloading, run:
```bash
python src/config.py  # prints data paths and checks file existence
```
