# AI Applications Project Documentation Template

Use this template to document your project concisely and completely.
Fill in all required fields. Keep answers short and precise.

## Documentation Hint

Important:
When possible, reference the corresponding code location directly in your description.

### Example: Reference to a notebook section
Reference to the header `## Data Preprocessing` in the notebook `analysis.ipynb`:

> See *Data Preprocessing* in
> [`analysis.ipynb`](analysis.ipynb#data-preprocessing)

### Example: Reference to Python code

Reference to a single line in `model.py`, line 42:
> [`model.py`, line 42](model.py#L42)

Reference to multiple lines in `train.py`, lines 15-38:
> [`train.py`, lines 15-38](train.py#L15-L38)

## Project Metadata

- Project title: **SneakerScout – AI-Powered Sneaker Recognition & Resell Advisor**
- Student: **F. Müller** (GitHub: `florianshub314`, HuggingFace: `florianshub314`)
- GitHub repository URL: https://github.com/florianshub314/sneaker-scout
- Deployment URL: https://huggingface.co/spaces/florianshub314/sneaker-scout
- Submission date: 2026-06-07

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [x] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [x] Computer Vision

Primary blocks used for core solution (choose 2):
- Primary block 1: **Computer Vision** – classifies the sneaker model from a user photo
- Primary block 2: **ML Numeric Data** – predicts the fair resell price and ROI from structured market features

If a third block is selected, it is documented and graded separately as extra work.

The third block (**NLP via prompt engineering**) is implemented and integrated. See section 2B and section 5.

Guidance hint: Keep the project idea short and consistent. Focus most details on the selected blocks.
Evidence hint: Show where each selected block contributes to the final system.

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition
- Problem statement: Sneaker buyers and resellers struggle to assess whether a given pair is a good purchase given today's secondary-market pricing. Existing platforms (StockX, GOAT) show real-time prices but do not give a contextual buy/hold/sell verdict, and they require the user to already know the model name. There is no consumer-friendly tool that combines visual recognition with a price-prediction model and a market-grounded recommendation.
- Goal: From a single sneaker photo, predict the model, estimate a fair resell price (USD), compute ROI vs retail, and return a buy/hold/sell recommendation with a transparent rationale grounded in market data.
- Success criteria:
  - CV classifier reaches macro-F1 ≥ 0.70 on the test split and beats a ResNet50 baseline.
  - Price regressor achieves R² ≥ 0.60 on the test split with residuals analysed per brand and per price segment.
  - The NLP advisor produces correct verdicts (consistent with the ROI sign) in at least 11 of the 12 manual test cases, scoring ≥ 4.0 mean on the rubric.
  - End-to-end pipeline runs on Hugging Face Spaces with image upload, < 10s inference, no API key in the repository.

### 1.2 Integration Logic
- How the selected blocks interact:
  1. CV (ViT, fine-tuned) consumes the uploaded image and returns `(sneaker_name, confidence)`.
  2. ML (tuned gradient boosting) consumes `sneaker_name`, plus user-supplied retail price and shoe size, and returns `(predicted_resell_price, roi)`.
  3. NLP (GPT-4o-mini with prompt engineering) consumes all of the above plus a structured market knowledge base and returns a natural-language recommendation.

  Each block's output is the next block's input – this is a true pipeline, not three independent models displayed side-by-side.

- Data and output flow between blocks:

  ```
  Image  ─►  CV (ViT)  ─► (sneaker_name, confidence)
                                  │
                Retail $, Size ───┤
                                  ▼
                            ML (GBM)  ─► (predicted_price, roi)
                                                 │
                Knowledge base ──────────────────┤
                                                 ▼
                                       NLP (GPT-4o-mini)
                                                 │
                                                 ▼
                                       Recommendation text
  ```

  Implementation hook: see [`app/app.py`, `analyse()` function, lines 213-258](app/app.py#L213-L258).

Guidance hint: This section should be short. The detailed work belongs in block sections.
Evidence hint: Include one clear pipeline overview.

---

## 2. Block Documentation

Complete only selected blocks. Mark non-selected block sections as N/A.

### 2A. ML Numeric Data (If selected)

#### 2A.1 Data Source(s)
List every usage of a data source as a separate entry. If the same source is used twice for different roles, add it twice.

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | [StockX Data Contest (Kaggle)](https://www.kaggle.com/datasets/hudsonstuck/stockx-data-contest) | Tabular (XLSX) | ~10k transactions, 50 models | Primary training data – sale price target + features |
| 2 | [Shoe Prices Dataset (Kaggle)](https://www.kaggle.com/datasets/rkiattisak/shoe-prices-dataset) | Tabular (CSV) | ~10k retail prices, multiple brands | Supplementary retail-context feature (brand-level retail stats) |
| 3 | [`data/knowledge_base/sneaker_market_context.json`](data/knowledge_base/sneaker_market_context.json) | Structured JSON | 8 models, 5 brands | Encodes brand-level resell premiums used as sanity reference |

#### 2A.2 Preprocessing and Features
- Cleaning steps:
  - Snake-case column names; parse `order_date` and `release_date` to `datetime`.
  - Drop rows with missing `sale_price`, `retail_price` or `shoe_size`.
  - IQR-based outlier check **kept** in dataset (high-priced grails are real targets, not noise) – see [`01_eda_stockx.ipynb` → *Outlier Detection*](notebooks/01_eda_stockx.ipynb).
- Preprocessing steps:
  - Stratified 70/15/15 train/val/test split by encoded brand. See [`src/preprocessing.py`, `split_data()`, lines 117-141](src/preprocessing.py#L117-L141).
- Feature engineering and selection (see [`src/preprocessing.py`, `engineer_stockx_features()`, lines 64-101](src/preprocessing.py#L64-L101)):
  - `price_premium = (sale_price - retail_price) / retail_price`
  - `days_since_release` (clipped at 0)
  - `month`, `quarter` from `order_date`
  - `size_category` ∈ {small, medium, large}
  - Label-encoded: `brand`, `sneaker_name`, `buyer_region`, `size_category`
  - Final feature vector (9 cols): see [`src/config.py`, `ML_FEATURE_COLS`, lines 66-75](src/config.py#L66-L75).

#### 2A.3 Model Selection
- Models tested: **Ridge Regression**, **Random Forest Regressor**, **Gradient Boosting (XGBoost)** with `RandomizedSearchCV` hyperparameter tuning.
- Why these models were chosen:
  - Ridge as a linear baseline establishes the lower bound and quantifies signal in the linear part of the feature space.
  - Random Forest captures non-linear interactions and brand × release-date effects without manual interaction terms.
  - Gradient Boosting is the canonical winner on tabular structured data of this size and shape (mixed numeric/categorical, moderate dimensionality, heavy-tailed target).

#### 2A.4 Model Comparison and Iterations
| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Establish linear baseline | StandardScaler + Ridge α=1.0 | Ridge | Test MAE | – |
| 2 | Capture non-linearity | Switch to RandomForest n=300, leaf=2 | Random Forest | Test MAE | Lower MAE, better R² |
| 3 | Boosted ensemble | XGBoost lr=0.05, depth=6 | XGBRegressor | Test MAE | Lower MAE than RF |
| 4 | Hyperparameter tuning | RandomizedSearchCV (15 trials, 3-fold) | XGBRegressor | Test MAE | Best MAE + best R² (production model) |

Notebook reference: [`03_ml_training.ipynb`](notebooks/03_ml_training.ipynb).

#### 2A.5 Evaluation and Error Analysis
- Metrics used: **MAE, RMSE, R², MAPE** plus 5-fold cross-validated MAE and R².
- Final results: production model = tuned XGBoost. Test-set MAE and R² are reported in the *Model Comparison Summary* of [`03_ml_training.ipynb`](notebooks/03_ml_training.ipynb).
- Error patterns and likely causes:
  - Highest absolute errors in the **>$1000 grail segment** – few training rows there, target distribution is long-tailed.
  - **Off-White** brand has the widest residual range – its pricing is driven by hype dynamics not captured by the static features.
  - Sub-$250 predictions sometimes over-shoot retail because the dataset rarely contains losses – the model is biased toward positive ROI.
  - Segment-level error breakdown: see *Price Segment Analysis* in [`03_ml_training.ipynb`](notebooks/03_ml_training.ipynb).
- **Operational guard:** at inference time the predicted price is clipped to `[0.5 × retail, 3 × retail]`. The lower bound prevents fire-sale predictions on rare SKUs; the upper bound prevents the model from extrapolating into hype-tier territory when the CV block hands over a label that the encoder cannot resolve (mapped to −1). See [`src/ml_model.py`, `PricePredictor.predict()`](src/ml_model.py).

#### 2A.6 Integration with Other Block(s)
- Inputs received from other block(s):
  - `sneaker_name` from the CV classifier. Used to look up the saved label encoder and produce `sneaker_name_encoded` and an inferred `brand_encoded` (see [`src/ml_model.py`, `PricePredictor._encode_features()`, lines 61-82](src/ml_model.py#L61-L82)).
- Outputs provided to other block(s):
  - `predicted_price` and `roi` are passed to the NLP advisor and rendered as result cards in the UI. See [`src/ml_model.py`, `PricePredictor.predict()`, lines 84-127](src/ml_model.py#L84-L127).

Guidance hint: Keep entries practical and evidence-based.
Evidence hint: Add values, not only claims.

### 2B. NLP (If selected)

#### 2B.1 Data Source(s)
List every usage of a data source as a separate entry. If the same source is used twice for different roles, add it twice.

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | [`data/knowledge_base/sneaker_market_context.json`](data/knowledge_base/sneaker_market_context.json) | Structured JSON | 8 models, 5 brands, market trends, limitations | Grounding context injected into prompts |
| 2 | CV block output | Inference output | 1 (model name, confidence) per request | Prompt input – which sneaker to advise on |
| 3 | ML block output | Inference output | (predicted_price, roi) per request | Prompt input – numerical basis for the verdict |

#### 2B.2 Preprocessing and Prompt Design
- Text preprocessing: none required – the knowledge base is already curated structured JSON. At request time, [`find_context_for_sneaker()`](src/nlp_advisor.py#L37-L52) looks up the right entry (model match → brand match → default fallback).
- Prompt design: three variants implemented in [`src/nlp_advisor.py`, lines 60-127](src/nlp_advisor.py#L60-L127):
  1. **`simple`** – minimal context (model + confidence + prices + ROI), generic system role.
  2. **`contextual`** – detailed system role ("nüchtern, datenbasiert, nennt Limitationen"), embeds full market context block, requests 3-4 sentences with at least one limitation.
  3. **`structured`** – JSON-schema-constrained output (`rating`, `summary`, `reasoning`, `limitations`) for deterministic downstream parsing.

#### 2B.3 Approach Selection
- Approach used: **Prompt engineering on a hosted LLM (GPT-4o-mini)** with grounded context injection.
- Alternatives considered:
  - Classical template-based generation: kept as a fallback when the OpenAI API is unavailable (see [`_fallback_recommendation()`, lines 158-184](src/nlp_advisor.py#L158-L184)).
  - Fine-tuning a smaller model: rejected because the training corpus would be tiny and the recommendation task does not require domain-specific token vocabulary.
  - RAG with embeddings: overkill for a curated knowledge base of <100 facts – direct lookup is faster and more interpretable.

#### 2B.4 Comparison and Iterations
| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Minimum viable advisor | Single-sentence prompt, no context | `simple` prompt, GPT-4o-mini | Rubric usefulness 2.6 / 5 | Baseline |
| 2 | Inject knowledge base | Add market context block, demand limitation in output | `contextual` prompt | Usefulness 4.5, relevance 4.7 | +1.9 usefulness |
| 3 | Deterministic output for UI | JSON schema enforced, response_format=`json_object` | `structured` prompt | Format 5.0, usefulness 4.2 | Best format, slight content trade-off |

Rubric (1-5 per criterion: relevance, correctness, usefulness, format) and 12 test cases live in [`04_nlp_evaluation.ipynb`](notebooks/04_nlp_evaluation.ipynb).

#### 2B.5 Evaluation and Error Analysis
- Evaluation strategy: 12 representative test cases covering the full ROI range (-18% to +2268%), high/medium/low CV confidence, and one unknown-sneaker fallback case. Each case is generated under all three prompt variants and manually scored against the four-criteria rubric.
- Results: `contextual` wins on overall usefulness (4.5) and relevance (4.7). `structured` wins on format consistency (5.0) but loses some narrative nuance. **Production default is `contextual`** ([`src/nlp_advisor.py`, line 195](src/nlp_advisor.py#L195)).
- Error patterns and likely causes:
  - **Hallucination for unknown models** – `simple` prompt invents context. Mitigation: `contextual` and `structured` inject a fallback object that the model uses verbatim.
  - **Confidence-blind verdicts** – even with low CV confidence, the model issues firm verdicts. Mitigation in the app: a UI-level warning above 50% confidence threshold instead of trying to steer the LLM ([`app/app.py`, lines 173-181](app/app.py#L173-L181)).
  - **Over-bullish on grails** – the LLM happily projects historic +2000% ROIs into the future. Mitigation: the system prompt explicitly demands a stated limitation in every response.

#### 2B.6 Integration with Other Block(s)
- Inputs received from other block(s):
  - `sneaker_name`, `confidence` from CV.
  - `predicted_price`, `roi`, `retail_price` from ML.
- Outputs provided to other block(s):
  - Recommendation text (string) rendered as the *AI Kaufempfehlung* card in the UI ([`app/app.py`, `_render_results()`, lines 142-185](app/app.py#L142-L185)). No downstream block consumes the NLP output – it is terminal in the pipeline.

Guidance hint: Show concrete prompt or retrieval decisions.
Evidence hint: Include representative outputs or failure cases.

### 2C. Computer Vision (If selected)

#### 2C.1 Data Source(s)
List every usage of a data source as a separate entry. If the same source is used twice for different roles, add it twice.

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | [Popular Sneakers Classification (Kaggle)](https://www.kaggle.com/datasets/nikolasgegenava/sneakers-classification/data) | Image (JPG/PNG) | 5k+ images across multiple sneaker classes | Primary training/eval data |
| 2 | User upload (run-time) | Image (PIL) | 1 per inference | Inference input |

#### 2C.2 Preprocessing and Augmentation
- Image preprocessing: resize to 224×224, convert to RGB, normalise to ImageNet mean/std. See [`src/preprocessing.py`, `get_eval_transforms()`, lines 33-39](src/preprocessing.py#L33-L39).
- Augmentation strategy (train only):
  - Resize to 256, random crop to 224
  - Random horizontal flip (p=0.5)
  - Random rotation ±15°
  - ColorJitter (brightness 0.3, contrast 0.3, saturation 0.2, hue 0.05)
  - See [`src/preprocessing.py`, `get_train_transforms()`, lines 22-31](src/preprocessing.py#L22-L31).

#### 2C.3 Model Selection
- Vision model(s) used: **Vision Transformer** (`google/vit-base-patch16-224`) fine-tuned end-to-end. Compared against a **ResNet50** baseline with the same augmentation and number of epochs.
- Why these model(s) were chosen:
  - ViT is the current state-of-the-art on fine-grained image classification of moderately sized datasets and integrates cleanly with the HuggingFace `Trainer` API for fast iteration.
  - ResNet50 is the canonical CNN baseline; comparing against it documents the lift from transformer-based attention on small, visually similar classes.

#### 2C.4 Model Comparison and Iterations
| Iteration | Objective | Key changes | Model(s) used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline ViT, no augmentation | lr=5e-5, no augmentations | ViT-base | Macro F1 (val) | Overfits ~3 epochs in |
| 2 | Add augmentation | + ColorJitter, flip, rotation | ViT-base | Macro F1 | +6pp vs Iter 1 |
| 3 | Stabilise training | lr=2e-5, warmup 100, early stop patience 3 | ViT-base | Macro F1 | Final config – production |
| – | CNN comparison | ResNet50, AdamW 1e-4, same augmentation | ResNet50 | Macro F1 | Underperforms ViT on rare classes |

Notebook reference: [`02_cv_training.ipynb`](notebooks/02_cv_training.ipynb).

#### 2C.5 Evaluation and Error Analysis
- Metrics and/or visual checks: macro Accuracy, Precision, Recall, F1; per-class F1 barplot; confusion matrix heatmap; visual sample-prediction grid for both correct and incorrect cases.
- Final results (5-epoch fine-tune of ViT-base over 51 classes, ~5.9k images): **Top-1 accuracy ≈ 50%, macro F1 ≈ 0.44**. Random baseline over 51 classes is 2%, so the model has learned strong but imperfect class structure. ViT outperforms the frozen-backbone ResNet50 baseline on macro F1; per-class breakdown lives in the *Model Comparison* and *Per-Class Accuracy* sections of [`02_cv_training.ipynb`](notebooks/02_cv_training.ipynb).
- Error patterns and limitations:
  - **Limited training budget.** Accuracy is bounded by 5 epochs on a single Apple-Silicon GPU; longer runs (15+ epochs) would likely lift top-1 by another 10–15 pp but were out of scope for the time budget.
  - **All-white sneakers** (Yeezy 350 Triple White, AF1 White, Reebok Club C 85) confuse one another – pure colour signal, similar silhouette. Live example: an Air Force 1 stock photo classifies as `reebok_club_c_85` at ~12% confidence.
  - **Air Jordan 1 sub-styles** are inter-confused, especially when the photo crops out the toe-box.
  - The model **does not detect authenticity** – it matches form, not provenance. This is the most operationally important limitation and is repeated in the app disclaimer.
  - Side-on vs front-on photos perform unevenly; future work could augment with more 3/4-angle examples.
- **App-side mitigation:** when CV confidence falls below `CV_CONFIDENCE_THRESHOLD` (0.50), the app short-circuits and shows only the top-3 alternatives plus a warning – the price and recommendation panels are deliberately suppressed because feeding a misclassified label into the ML block produces meaningless numbers (the StockX label-encoder maps unknown SKUs to −1 and XGBoost then extrapolates wildly). See [`app/app.py`, `analyse()`, low-confidence branch](app/app.py).

#### 2C.6 Integration with Other Block(s)
- Inputs received from other block(s): none – CV is the entry point of the pipeline (consumes the raw uploaded image).
- Outputs provided to other block(s):
  - `(sneaker_name, confidence)` consumed by the ML block as the categorical key for the price prediction.
  - `sneaker_name` and `confidence` also consumed by the NLP block – `sneaker_name` for the context lookup, `confidence` for the UI-side low-confidence warning.
  - See [`src/cv_model.py`, `SneakerClassifier.predict()`, lines 53-77](src/cv_model.py#L53-L77).

Guidance hint: Use concise examples from real predictions.
Evidence hint: Include sample outputs and observed failure cases.

---

## 3. Deployment

- **Live Space:** https://huggingface.co/spaces/muellfl/sneaker-scout
- **Fine-tuned CV checkpoint:** https://huggingface.co/muellfl/sneaker-scout-vit (loaded by the Space at startup).
- **Source code:** https://github.com/florianshub314/sneaker-scout

### Main user flow

1. User opens the Space and lands on the **Analyse** tab.
2. Drag-and-drops a sneaker photo, optionally adjusts shoe size and retail price, clicks **Analysieren**.
3. **High-confidence path (CV ≥ 0.50):** the full result is shown automatically – Modell + Confidence, Resell Price (clipped to ≤ 3× retail), ROI with Buy/Hold/Sell badge, and the AI recommendation paragraph.
4. **Low-confidence path (CV < 0.50):** the price step is suppressed; the user sees a warning, the top-3 alternatives, and a searchable dropdown of all 50 trained classes. The user picks the correct model (or any other class) and clicks **Mit dieser Klasse Preis berechnen** to run the price + recommendation step.
5. The **Marktdaten** tab shows distributional statistics (avg / min / max / ROI) and a monthly median price chart per model based on the StockX 2017–2019 splits.
6. The **Über** tab shows a one-screen project description.

### Screenshots

| View | File |
| --- | --- |
| High-confidence happy path (Yeezy 350 V2, 56% conf, $489 capped price, +200% ROI, STRONG BUY) | [`assets/screenshots/high_confidence_yeezy.png`](assets/screenshots/high_confidence_yeezy.png) |
| Low-confidence warning + correction dropdown closed | [`assets/screenshots/low_confidence_picker_closed.png`](assets/screenshots/low_confidence_picker_closed.png) |
| Same view with the dropdown open, typing "ad" filters the full 50-class list | [`assets/screenshots/low_confidence_picker_open.png`](assets/screenshots/low_confidence_picker_open.png) |
| Marktdaten tab – stat cards + monthly median price chart for Nike-Zoom-Fly-Off-White | [`assets/screenshots/market_data_overview.png`](assets/screenshots/market_data_overview.png) |
| Marktdaten tab – chart detail (clear upward price trend 2017-11 → 2019-01) | [`assets/screenshots/market_data_chart.png`](assets/screenshots/market_data_chart.png) |
| Über tab – project description, pipeline, datasets, ethics and tech stack | [`assets/screenshots/about_tab.png`](assets/screenshots/about_tab.png) |

---

## 4. Execution Instructions

- Environment setup:
  ```bash
  git clone https://github.com/florianshub314/sneaker-scout.git
  cd sneaker-scout
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```

- Data setup: download the three datasets per [`data/raw/README.md`](data/raw/README.md). With Kaggle CLI configured:
  ```bash
  kaggle datasets download -d nikolasgegenava/sneakers-classification -p data/raw/ --unzip
  kaggle datasets download -d hudsonstuck/stockx-data-contest -p data/raw/stockx/ --unzip
  kaggle datasets download -d rkiattisak/shoe-prices-dataset -p data/raw/shoe_prices/ --unzip
  ```

- Training command(s) – run notebooks in order:
  ```bash
  jupyter nbconvert --to notebook --execute notebooks/01_eda_stockx.ipynb
  jupyter nbconvert --to notebook --execute notebooks/02_cv_training.ipynb
  jupyter nbconvert --to notebook --execute notebooks/03_ml_training.ipynb
  jupyter nbconvert --to notebook --execute notebooks/04_nlp_evaluation.ipynb
  jupyter nbconvert --to notebook --execute notebooks/05_ethics_and_bias.ipynb
  ```
  Or interactively in Jupyter – `01` produces processed parquet files + encoders, `02` saves the ViT under `models/cv_model/`, `03` saves the GBM under `models/ml_model/`.

- Inference/run command(s):
  ```bash
  export OPENAI_API_KEY="sk-..."
  python app/app.py
  # Open http://localhost:7860
  ```
  Tests:
  ```bash
  pytest tests/ -q
  ```

- Reproducibility notes:
  - All randomness uses `random_state=42` (defined once in [`src/config.py`](src/config.py#L75)).
  - Dependencies are pinned in `requirements.txt`.
  - Trained models are not in git (size). Re-run notebooks `02` and `03` to regenerate. Hashes of the processed parquet files are deterministic per the fixed seed.
  - Python 3.10+ required.

Guidance hint: Another person should be able to run your project from this section.
Evidence hint: Include exact commands and versions.

---

## 5. Optional Bonus Evidence

Use this section for exceptional work beyond the core requirements.

- [x] Third selected block implemented with strong quality
- [x] More than two data sources used with clear added value
- [x] A core section is done exceptionally well
- [x] Extended evaluation
- [x] Ethics, bias, or fairness analysis
- [x] Creative or exceptional use case

Evidence for selected bonus items:

- **Third block implemented (NLP).** Full prompt-engineering pipeline with three variants, rubric-based qualitative evaluation across 12 test cases, structured fallback, knowledge-base grounding. See section 2B and [`04_nlp_evaluation.ipynb`](notebooks/04_nlp_evaluation.ipynb).

- **Four data sources with clear added value.** (1) StockX transactions for the price model, (2) Popular Sneakers Classification for the CV model, (3) Shoe Prices Dataset as supplementary retail context, (4) the curated knowledge base for NLP grounding. Each source plays a distinct, non-redundant role in the pipeline – see the data-source tables in sections 2A, 2B, 2C.

- **CV section done exceptionally well.** Per-class F1 ranking, confusion matrix heatmap, sampled mis-classification grid, top-confusion-pair analysis, explicit colourway-driven error discussion. Detailed iteration log + ResNet50 comparison. See *Per-Class Accuracy*, *Confusion Matrix* and *Error Analysis* in [`02_cv_training.ipynb`](notebooks/02_cv_training.ipynb).

- **Extended evaluation.** 5-fold cross-validated MAE and R² for all three ML models; per-price-segment error breakdown; residuals-vs-predicted scatter and residual distribution; feature importance plot. NLP: four-criteria rubric across 12 cases + side-by-side comparison cells. CV: per-class F1, confusion matrix, visualised error samples.

- **Ethics, bias, fairness analysis.** Dedicated notebook [`05_ethics_and_bias.ipynb`](notebooks/05_ethics_and_bias.ipynb) covering: geographic bias (USA-dominant), temporal bias (2017-2019 only), brand imbalance (Off-White/Yeezy oversampled), ML residuals by brand, CV per-class fairness, socioeconomic considerations of the resell market, explicit limitations (no fake detection, no real-time pricing, no condition assessment), in-app disclaimer policy.

- **Creative use case.** Combining streetwear culture (sneaker resell market) with quantitative decision support: from a single photo to a buy/hold/sell verdict with market reasoning. The dashboard styling (StockX/Bloomberg-terminal aesthetic) is purpose-built for the use case rather than a stock Gradio look.
