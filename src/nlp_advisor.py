"""NLP advisor module: generate purchase recommendation via LLM with prompt engineering."""

import json
import os
from pathlib import Path
from typing import Optional

from src.config import (
    KNOWLEDGE_BASE_FILE,
    OPENAI_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
)


# ---------------------------------------------------------------------------
# Knowledge base loader
# ---------------------------------------------------------------------------

_knowledge_cache: Optional[dict] = None


def load_knowledge_base(path: Path = KNOWLEDGE_BASE_FILE) -> dict:
    """Load the sneaker market knowledge base from JSON."""
    global _knowledge_cache
    if _knowledge_cache is None:
        with open(path, "r", encoding="utf-8") as f:
            _knowledge_cache = json.load(f)
    return _knowledge_cache


def find_context_for_sneaker(sneaker_name: str, kb: dict) -> dict:
    """Look up market context for a sneaker by name. Falls back to brand context."""
    name_lower = sneaker_name.lower()

    for model_key, info in kb.get("models", {}).items():
        if model_key.lower() in name_lower or name_lower in model_key.lower():
            return info

    # Brand fallback
    for brand_key, info in kb.get("brands", {}).items():
        if brand_key.lower() in name_lower:
            return {**info, "_match_type": "brand"}

    return kb.get("default", {})


# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------

def build_simple_prompt(
    model: str,
    confidence: float,
    predicted_price: float,
    retail_price: float,
    roi: float,
) -> list[dict]:
    """Prompt variant 1: minimal context, just numbers."""
    user = (
        f"Sneaker: {model}\n"
        f"Erkennungs-Confidence: {confidence:.0%}\n"
        f"Retail Price: ${retail_price:.0f}\n"
        f"Predicted Resell Price: ${predicted_price:.0f}\n"
        f"ROI: {roi*100:+.0f}%\n\n"
        "Gib eine kurze Kaufempfehlung in 2-3 Sätzen."
    )
    return [
        {"role": "system", "content": "Du bist ein Sneaker-Resell-Analyst."},
        {"role": "user", "content": user},
    ]


def build_contextual_prompt(
    model: str,
    confidence: float,
    predicted_price: float,
    retail_price: float,
    roi: float,
    context: dict,
) -> list[dict]:
    """Prompt variant 2: detailed prompt with market context from knowledge base."""
    context_block = json.dumps(context, ensure_ascii=False, indent=2)

    system = (
        "Du bist ein erfahrener Analyst für den Sneaker-Resell-Markt. "
        "Du gibst datenbasierte, nüchterne Einschätzungen – kein Hype. "
        "Du nennst immer Limitationen und beziehst Marktkontext mit ein."
    )
    user = (
        f"## Erkennung\n"
        f"- Modell: {model}\n"
        f"- Confidence: {confidence:.0%}\n\n"
        f"## Preisdaten\n"
        f"- Retail: ${retail_price:.0f}\n"
        f"- Predicted Resell: ${predicted_price:.0f}\n"
        f"- ROI: {roi*100:+.0f}%\n\n"
        f"## Marktkontext\n"
        f"```json\n{context_block}\n```\n\n"
        "Gib eine fundierte Kaufempfehlung (3-4 Sätze). "
        "Beziehe den Marktkontext mit ein und erwähne mindestens eine Limitation."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_structured_prompt(
    model: str,
    confidence: float,
    predicted_price: float,
    retail_price: float,
    roi: float,
    context: dict,
) -> list[dict]:
    """Prompt variant 3: structured JSON output for consistent parsing."""
    context_block = json.dumps(context, ensure_ascii=False)

    system = (
        "Du bist ein Sneaker-Resell-Analyst. Antworte ausschliesslich mit "
        "validem JSON im vorgegebenen Schema."
    )
    schema = (
        '{"rating": "STRONG BUY|BUY|HOLD|SELL", '
        '"summary": "string (2 Sätze)", '
        '"reasoning": "string (Begründung mit Marktkontext, 2-3 Sätze)", '
        '"limitations": "string (mindestens eine Limitation, 1 Satz)"}'
    )
    user = (
        f"Sneaker: {model} | Confidence: {confidence:.0%} | "
        f"Retail: ${retail_price:.0f} | Predicted: ${predicted_price:.0f} | "
        f"ROI: {roi*100:+.0f}%\n\n"
        f"Marktkontext: {context_block}\n\n"
        f"Schema: {schema}\n\n"
        "Antworte mit JSON."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


PROMPT_VARIANTS = {
    "simple": build_simple_prompt,
    "contextual": build_contextual_prompt,
    "structured": build_structured_prompt,
}


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_openai(messages: list[dict], json_mode: bool = False) -> str:
    """Send messages to OpenAI API and return assistant response text."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    kwargs = dict(
        model=OPENAI_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


def _fallback_recommendation(
    model: str,
    predicted_price: float,
    retail_price: float,
    roi: float,
) -> str:
    """Template-based recommendation used when the LLM is unavailable."""
    if roi > 0.5:
        rating = "STRONG BUY"
        verdict = "deutlich über Retail"
    elif roi > 0.1:
        rating = "BUY"
        verdict = "moderat über Retail"
    elif roi > -0.1:
        rating = "HOLD"
        verdict = "nah am Retail-Preis"
    else:
        rating = "SELL"
        verdict = "unter Retail"

    return (
        f"{model}: {rating}. Der prognostizierte Resell-Preis von ${predicted_price:.0f} "
        f"liegt {verdict} (Retail: ${retail_price:.0f}, ROI: {roi*100:+.0f}%). "
        "Hinweis: Diese Schätzung basiert auf StockX-Daten 2017-2019 und ersetzt keine "
        "aktuelle Marktrecherche."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_recommendation(
    sneaker_name: str,
    confidence: float,
    predicted_price: float,
    retail_price: float,
    roi: float,
    variant: str = "contextual",
) -> str:
    """Generate a purchase recommendation using the chosen prompt variant.

    Args:
        sneaker_name: CV model output.
        confidence: CV confidence score 0-1.
        predicted_price: ML predicted resell price.
        retail_price: Original retail price.
        roi: Return on investment as fraction.
        variant: One of "simple", "contextual", "structured".

    Returns:
        Recommendation text. Falls back to template if API call fails.
    """
    builder = PROMPT_VARIANTS.get(variant, build_contextual_prompt)

    try:
        if variant == "simple":
            messages = builder(sneaker_name, confidence, predicted_price, retail_price, roi)
        else:
            kb = load_knowledge_base()
            context = find_context_for_sneaker(sneaker_name, kb)
            messages = builder(sneaker_name, confidence, predicted_price, retail_price, roi, context)

        return _call_openai(messages, json_mode=(variant == "structured"))
    except Exception as exc:
        # Log but never crash – always return a usable recommendation
        print(f"[NLP] Falling back to template: {exc}")
        return _fallback_recommendation(sneaker_name, predicted_price, retail_price, roi)
