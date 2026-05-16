"""SneakerScout Gradio app – CV + ML + NLP pipeline with dark-themed dashboard UI."""

import os
import sys
import json
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# Make src/ importable when running `python app/app.py`
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(ROOT))

from src import config
from src.cv_model import get_classifier, get_confidence_level
from src.ml_model import get_predictor, _extract_brand
from src.nlp_advisor import generate_recommendation, load_knowledge_base


# ---------------------------------------------------------------------------
# Theme + CSS
# ---------------------------------------------------------------------------

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.gray,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#0f0f0f",
    body_background_fill_dark="#0f0f0f",
    block_background_fill="#1a1a2e",
    block_background_fill_dark="#1a1a2e",
    input_background_fill="#16213e",
    input_background_fill_dark="#16213e",
    body_text_color="#ffffff",
    body_text_color_dark="#ffffff",
    block_label_text_color="#8a8a9a",
    block_label_text_color_dark="#8a8a9a",
    input_border_color="#2a2a3e",
    border_color_primary="#3b82f6",
    button_primary_background_fill="#3b82f6",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#1a1a2e",
    button_secondary_text_color="#ffffff",
    block_radius="8px",
    input_radius="8px",
    button_large_radius="8px",
    block_shadow="none",
    block_shadow_dark="none",
)

CUSTOM_CSS = """
.app-header {
    padding: 24px 0 16px 0;
    border-bottom: 1px solid #2a2a3e;
    margin-bottom: 24px;
}
.app-title {
    font-size: 28px; font-weight: 800;
    text-transform: uppercase; letter-spacing: -0.02em;
    color: #ffffff;
}
.app-subtitle {
    font-size: 13px; font-weight: 500;
    color: #8a8a9a; letter-spacing: 0.05em;
    text-transform: uppercase; margin-top: 4px;
}
.result-card {
    background: #1a1a2e; border: 1px solid #2a2a3e;
    border-radius: 8px; padding: 20px;
}
.result-value {
    font-size: 36px; font-weight: 800;
    font-variant-numeric: tabular-nums; line-height: 1.1;
}
.result-label {
    font-size: 11px; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: #8a8a9a; margin-bottom: 4px;
}
.value-positive { color: #00d4aa; }
.value-negative { color: #e94560; }
.value-neutral { color: #f39c12; }
.value-default { color: #ffffff; }
.badge {
    display: inline-block; padding: 4px 12px;
    font-size: 12px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em;
    border-radius: 4px;
}
.badge-strong-buy { background: #00d4aa20; color: #00d4aa; border: 1px solid #00d4aa40; }
.badge-buy       { background: #00d4aa10; color: #00d4aa; border: 1px solid #00d4aa20; }
.badge-hold      { background: #f39c1220; color: #f39c12; border: 1px solid #f39c1240; }
.badge-sell      { background: #e9456020; color: #e94560; border: 1px solid #e9456040; }
.recommendation-box {
    background: #16213e; border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0; padding: 20px;
    font-size: 15px; line-height: 1.6; color: #ffffff;
}
.confidence-bar {
    height: 4px; border-radius: 2px;
    background: #2a2a3e; overflow: hidden; margin-top: 6px;
}
.confidence-fill { height: 100%; border-radius: 2px; transition: width 0.3s ease; }
.confidence-high   { background: #00d4aa; }
.confidence-medium { background: #f39c12; }
.confidence-low    { background: #e94560; }
.disclaimer {
    font-size: 11px; color: #4a4a5a;
    border-top: 1px solid #2a2a3e;
    padding-top: 12px; margin-top: 24px;
}
"""


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def _rating(roi: float) -> dict:
    if roi > 0.5:
        return {"label": "STRONG BUY", "css": "badge-strong-buy", "color_class": "value-positive"}
    if roi > 0.1:
        return {"label": "BUY", "css": "badge-buy", "color_class": "value-positive"}
    if roi > -0.1:
        return {"label": "HOLD", "css": "badge-hold", "color_class": "value-neutral"}
    return {"label": "SELL", "css": "badge-sell", "color_class": "value-negative"}


def _format_price(p: float) -> str:
    return f"${p:,.0f}"


def _format_roi(r: float) -> str:
    sign = "+" if r > 0 else ""
    return f"{sign}{r * 100:.0f}%"


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------

def _render_low_confidence(
    sneaker_name: str,
    confidence: float,
    top3: list[tuple[str, float]],
) -> str:
    """Render a low-confidence result: warning + top-3 alternatives, no price block.

    Price prediction is suppressed because feeding a misclassified label into the
    resell model produces meaningless numbers.
    """
    conf_pct = int(round(confidence * 100))
    alts = "".join(
        f'<li><span style="color: #e0e0e0;">{name}</span>'
        f'<span style="color: #8a8a9a; margin-left: 8px;">{int(round(p * 100))}%</span></li>'
        for name, p in top3
    )
    return f"""
<div style="padding: 16px; background: #f39c1215; border-left: 3px solid #f39c12; border-radius: 4px; margin-bottom: 16px;">
  <div style="color: #f39c12; font-weight: 600; font-size: 16px; margin-bottom: 6px;">
    Sneaker konnte nicht zuverlaessig erkannt werden (Confidence {conf_pct}%).
  </div>
  <div style="color: #c0c0c0;">
    Preis- und Empfehlungsschritte wurden uebersprungen, um irrefuehrende Werte zu vermeiden.
    Bitte ein klareres Foto aus einem anderen Winkel oder mit besserer Beleuchtung versuchen.
  </div>
</div>

<div class="result-card">
  <div class="result-label">Beste Vermutung</div>
  <div class="result-value value-default" style="font-size: 22px;">{sneaker_name}</div>
  <div class="result-label" style="margin-top: 16px;">Naechstbeste Modelle</div>
  <ul style="list-style: none; padding: 0; margin: 8px 0 0;">{alts}</ul>
</div>

<div class="disclaimer">
  Daten: StockX 2017-2019 | Modelle: ViT + GBM + GPT-4o-mini |
  Keine Anlageberatung. Nur zu Informationszwecken. Keine Faelschungserkennung.
</div>
"""


def _render_results(
    sneaker_name: str,
    confidence: float,
    retail_price: float,
    predicted_price: float,
    roi: float,
    recommendation: str,
) -> str:
    """Return HTML for the result cards block."""
    rating = _rating(roi)
    conf_pct = int(round(confidence * 100))
    conf_class = "confidence-high" if confidence >= 0.9 else (
        "confidence-medium" if confidence >= 0.7 else "confidence-low"
    )
    roi_class = (
        "value-positive" if roi > 0.1 else
        ("value-negative" if roi < -0.1 else "value-neutral")
    )

    return f"""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
  <div class="result-card">
    <div class="result-label">Modell</div>
    <div class="result-value value-default" style="font-size: 22px;">{sneaker_name}</div>
    <div class="result-label" style="margin-top: 12px;">Confidence {conf_pct}%</div>
    <div class="confidence-bar"><div class="confidence-fill {conf_class}" style="width: {conf_pct}%;"></div></div>
  </div>

  <div class="result-card">
    <div class="result-label">Resell Price</div>
    <div class="result-value value-default">{_format_price(predicted_price)}</div>
    <div class="result-label" style="margin-top: 12px;">Retail</div>
    <div style="color: #8a8a9a; font-size: 14px;">{_format_price(retail_price)}</div>
  </div>

  <div class="result-card">
    <div class="result-label">ROI</div>
    <div class="result-value {roi_class}">{_format_roi(roi)}</div>
    <div style="margin-top: 12px;"><span class="badge {rating['css']}">{rating['label']}</span></div>
  </div>
</div>

<div class="result-card" style="margin-top: 16px;">
  <div class="result-label">AI Kaufempfehlung</div>
  <div class="recommendation-box">{recommendation}</div>
</div>

<div class="disclaimer">
  Daten: StockX 2017-2019 | Modelle: ViT + GBM + GPT-4o-mini |
  Keine Anlageberatung. Nur zu Informationszwecken. Keine Fälschungserkennung.
</div>
"""


# ---------------------------------------------------------------------------
# Main analyse callback
# ---------------------------------------------------------------------------

def _run_ml_nlp(sneaker_name: str, confidence: float, shoe_size: float, retail_price: float) -> str:
    """Shared ML + NLP path, used both by the automatic high-confidence flow
    and by the manual override (user-picked top-3 alternative)."""
    try:
        predictor = get_predictor()
        predicted_price, roi = predictor.predict(
            sneaker_name=sneaker_name,
            retail_price=retail_price,
            shoe_size=shoe_size,
            brand=_extract_brand(sneaker_name),
        )
    except Exception as e:
        return f'<div class="result-card" style="color: #e94560;">Preis-Modell nicht verfügbar: {e}</div>'

    recommendation = generate_recommendation(
        sneaker_name=sneaker_name,
        confidence=confidence,
        predicted_price=predicted_price,
        retail_price=retail_price,
        roi=roi,
    )
    return _render_results(
        sneaker_name=sneaker_name,
        confidence=confidence,
        retail_price=retail_price,
        predicted_price=predicted_price,
        roi=roi,
        recommendation=recommendation,
    )


# Empty/hidden updates for the override widgets (used on every high-confidence path
# and on the override callback itself, so the dropdown collapses after use).
_HIDE_OVERRIDE = (
    gr.update(choices=[], value=None, visible=False),
    gr.update(visible=False),
)


def _all_class_choices() -> list[str]:
    """All trained class names, sorted, for the manual-override dropdown.

    Pulled from the loaded classifier so it stays in sync with the deployed
    model (and works whether weights are local or fetched from HF Hub).
    """
    clf = get_classifier()
    clf.load()
    return sorted(clf.id2label.values())


def analyse(image: Optional[Image.Image], shoe_size: float, retail_price: float):
    if image is None:
        empty = ('<div class="result-card" style="color: #8a8a9a;">'
                 'Bitte zuerst ein Sneaker-Foto hochladen.</div>')
        return empty, *_HIDE_OVERRIDE

    # Defaults if user leaves inputs empty
    if shoe_size is None:
        shoe_size = 10.0
    if retail_price is None or retail_price <= 0:
        retail_price = 150.0

    # 1) CV
    try:
        classifier = get_classifier()
        sneaker_name, confidence = classifier.predict(image)
        top3 = classifier.predict_topk(image, k=3)
    except Exception as e:
        return (
            f'<div class="result-card" style="color: #e94560;">CV-Modell nicht verfügbar: {e}</div>',
            *_HIDE_OVERRIDE,
        )

    # Low-confidence path: skip auto-ML/NLP, surface a manual picker.
    # The dropdown holds ALL trained classes (typable + searchable) so the
    # user can recover even when the top-3 contains nothing correct. The
    # top-1 guess is pre-selected; the top-3 are displayed in the result HTML
    # for quick orientation.
    if confidence < config.CV_CONFIDENCE_THRESHOLD:
        return (
            _render_low_confidence(sneaker_name, confidence, top3),
            gr.update(choices=_all_class_choices(), value=top3[0][0], visible=True),
            gr.update(visible=True),
        )

    # High-confidence path: full pipeline
    return _run_ml_nlp(sneaker_name, confidence, shoe_size, retail_price), *_HIDE_OVERRIDE


def analyse_with_override(override_label: str, shoe_size: float, retail_price: float):
    """Run ML + NLP for a user-confirmed class (picked from the top-3 dropdown)."""
    if not override_label:
        return (
            '<div class="result-card" style="color: #8a8a9a;">'
            'Bitte zuerst eine Klasse aus der Liste waehlen.</div>',
            gr.update(),
            gr.update(),
        )
    if shoe_size is None:
        shoe_size = 10.0
    if retail_price is None or retail_price <= 0:
        retail_price = 150.0

    # User-confirmed → treat as high-confidence for downstream copy.
    html = _run_ml_nlp(
        sneaker_name=override_label,
        confidence=1.0,
        shoe_size=shoe_size,
        retail_price=retail_price,
    )
    return html, *_HIDE_OVERRIDE


# ---------------------------------------------------------------------------
# Market data view
# ---------------------------------------------------------------------------

def _load_market_data() -> pd.DataFrame:
    """Load the saved StockX test split for the market overview tab.

    Falls back to the knowledge base summary if processed data is unavailable.
    """
    for path in (config.PROCESSED_TEST, config.PROCESSED_VAL, config.PROCESSED_TRAIN):
        if path.exists():
            df = pd.read_parquet(path)
            if "sneaker_name" in df.columns and "sale_price" in df.columns:
                return df
    return pd.DataFrame()


def market_dropdown_choices() -> list[str]:
    df = _load_market_data()
    if df.empty:
        kb = load_knowledge_base()
        return sorted(kb.get("models", {}).keys())
    return sorted(df["sneaker_name"].dropna().unique().tolist())[:50]


def market_view(sneaker_name: str):
    """Return summary cards + price-trend chart for a chosen sneaker."""
    df = _load_market_data()
    if df.empty or not sneaker_name:
        return "<div class='result-card' style='color: #8a8a9a;'>Keine Marktdaten verfügbar.</div>", None

    subset = df[df["sneaker_name"] == sneaker_name]
    if subset.empty:
        return f"<div class='result-card'>Keine Marktdaten für {sneaker_name}.</div>", None

    avg_price = subset["sale_price"].mean()
    min_price = subset["sale_price"].min()
    max_price = subset["sale_price"].max()
    retail = subset["retail_price"].mean() if "retail_price" in subset.columns else 0
    avg_roi = (avg_price - retail) / retail if retail > 0 else 0

    summary_html = f"""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px;">
  <div class="result-card"><div class="result-label">Avg Price</div>
    <div class="result-value value-default">{_format_price(avg_price)}</div></div>
  <div class="result-card"><div class="result-label">Avg ROI</div>
    <div class="result-value {'value-positive' if avg_roi > 0.1 else 'value-neutral'}">{_format_roi(avg_roi)}</div></div>
  <div class="result-card"><div class="result-label">Min</div>
    <div class="result-value value-default" style="font-size: 28px;">{_format_price(min_price)}</div></div>
  <div class="result-card"><div class="result-label">Max</div>
    <div class="result-value value-default" style="font-size: 28px;">{_format_price(max_price)}</div></div>
</div>
"""

    fig = _make_price_chart(subset, sneaker_name)
    return summary_html, fig


def _make_price_chart(subset: pd.DataFrame, title: str):
    """Build a dark-themed price-over-time chart."""
    matplotlib.rcParams["font.family"] = "sans-serif"
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    if "order_date" in subset.columns:
        ts = subset.copy()
        ts["order_date"] = pd.to_datetime(ts["order_date"], errors="coerce")
        ts = ts.dropna(subset=["order_date"]).sort_values("order_date")
        monthly = ts.groupby(ts["order_date"].dt.to_period("M"))["sale_price"].median()
        if not monthly.empty:
            x = monthly.index.to_timestamp()
            y = monthly.values
            ax.plot(x, y, color="#00d4aa", linewidth=2)
            ax.fill_between(x, y, alpha=0.15, color="#00d4aa")
    else:
        ax.hist(subset["sale_price"], bins=30, color="#00d4aa")

    ax.grid(True, color="#2a2a3e", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#2a2a3e")
    ax.spines["left"].set_color("#2a2a3e")
    ax.tick_params(colors="#8a8a9a", labelsize=10)
    ax.set_title(f"Price Trend: {title}", color="#ffffff", fontsize=14, fontweight="bold", loc="left")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# About text
# ---------------------------------------------------------------------------

ABOUT_TEXT = """
## Über SneakerScout

**Use Case:** Du hältst einen Sneaker in der Hand – Air Jordan, Yeezy, Off-White –
und fragst dich: kaufen, halten, oder weiterverkaufen? SneakerScout kombiniert drei
KI-Bausteine, um aus einem einzigen Foto eine datenbasierte Empfehlung zu generieren.

### Pipeline

1. **Computer Vision (ViT, fine-tuned)** – erkennt das Modell aus dem Bild.
2. **ML Regression (Gradient Boosting)** – prognostiziert den fairen Resell-Preis.
3. **NLP (GPT-4o-mini, prompt engineering)** – formuliert die Empfehlung mit Marktkontext.

### Datenquellen

- Popular Sneakers Classification (Kaggle)
- StockX Data Contest 2017-2019 (Kaggle)
- Shoe Prices Dataset (Kaggle)
- Eigene strukturierte Knowledge Base zu Marken und Modellen

### Ethik & Limitationen

- Die StockX-Daten sind USA-lastig und stammen aus 2017-2019 – aktuelle Markttrends
  werden nicht abgebildet.
- Off-White und Yeezy sind im Datensatz überrepräsentiert – seltene Marken haben
  schlechtere Modellperformance.
- Die App authentifiziert Sneaker **nicht** und ersetzt keine professionelle Beratung.
- Confidence-Werte unter 50% lösen eine explizite Warnung in der App aus.

### Tech Stack

HuggingFace Transformers · scikit-learn / XGBoost · OpenAI API · Gradio · HuggingFace Spaces

### Projekt-Kontext

ZHAW SML – AI Applications FS2026.
"""


# ---------------------------------------------------------------------------
# Build the interface
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title=config.APP_TITLE) as demo:
        gr.HTML(
            f'<div class="app-header">'
            f'<div class="app-title">{config.APP_TITLE}</div>'
            f'<div class="app-subtitle">{config.APP_SUBTITLE}</div>'
            f'</div>'
        )

        with gr.Tabs():
            # ---- Tab 1: Analyse ---------------------------------------------
            with gr.Tab("Analyse"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_in = gr.Image(type="pil", label="Sneaker-Foto", height=320)
                        shoe_size = gr.Dropdown(
                            choices=[6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 14],
                            value=10, label="Schuhgrösse (US)",
                        )
                        retail = gr.Number(value=170, label="Retail Price ($)", precision=0)
                        analyse_btn = gr.Button("Analysieren", variant="primary")

                    with gr.Column(scale=2):
                        result_html = gr.HTML(
                            value='<div class="result-card" style="color: #8a8a9a;">'
                                  'Lade ein Sneaker-Foto hoch und klicke <b>Analysieren</b>.</div>'
                        )
                        # Hidden until a low-confidence CV result needs manual review.
                        # Holds all trained classes; type to filter.
                        override_dd = gr.Dropdown(
                            choices=[],
                            label="Klasse korrigieren (alle Modelle, tippen zum Filtern)",
                            visible=False,
                            interactive=True,
                            filterable=True,
                        )
                        override_btn = gr.Button(
                            "Mit dieser Klasse Preis berechnen",
                            variant="secondary",
                            visible=False,
                        )

                analyse_btn.click(
                    fn=analyse,
                    inputs=[image_in, shoe_size, retail],
                    outputs=[result_html, override_dd, override_btn],
                )
                override_btn.click(
                    fn=analyse_with_override,
                    inputs=[override_dd, shoe_size, retail],
                    outputs=[result_html, override_dd, override_btn],
                )

                # Example buttons (only render if files exist)
                example_files = [
                    ROOT / "app" / "examples" / name
                    for name in ("jordan1.jpg", "yeezy350.jpg", "dunk_low.jpg")
                ]
                existing = [str(p) for p in example_files if p.exists()]
                if existing:
                    gr.Examples(
                        examples=[[p, 10, 170] for p in existing],
                        inputs=[image_in, shoe_size, retail],
                        label="Beispiele",
                    )

            # ---- Tab 2: Marktdaten ------------------------------------------
            with gr.Tab("Marktdaten"):
                gr.HTML('<div class="app-subtitle" style="margin-bottom: 16px;">'
                        'Resell-Statistiken nach Modell</div>')
                model_dd = gr.Dropdown(
                    choices=market_dropdown_choices(),
                    label="Sneaker-Modell wählen",
                )
                market_html = gr.HTML()
                market_plot = gr.Plot()
                model_dd.change(market_view, inputs=model_dd, outputs=[market_html, market_plot])

            # ---- Tab 3: Über ------------------------------------------------
            with gr.Tab("Über"):
                gr.Markdown(ABOUT_TEXT)

    return demo


if __name__ == "__main__":
    app = build_app()
    # HF Spaces injects GRADIO_SERVER_NAME=0.0.0.0; locally bind to loopback
    # because macOS sometimes fails the 0.0.0.0 self-probe during startup.
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", config.APP_PORT)),
        show_error=True,
        show_api=False,
    )
