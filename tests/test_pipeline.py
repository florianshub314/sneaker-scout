"""Basic tests for the SneakerScout pipeline modules."""

import json
from pathlib import Path

import pytest

from src import config
from src.nlp_advisor import (
    load_knowledge_base,
    find_context_for_sneaker,
    build_simple_prompt,
    build_contextual_prompt,
    build_structured_prompt,
    _fallback_recommendation,
)
from src.preprocessing import get_eval_transforms, get_train_transforms
from src.ml_model import _extract_brand, _size_category


# ---------------------------------------------------------------------------
# Config sanity
# ---------------------------------------------------------------------------

def test_config_paths_exist():
    assert config.ROOT.exists()
    assert config.DATA_KNOWLEDGE_BASE.exists()
    assert config.KNOWLEDGE_BASE_FILE.exists()


def test_config_ratios_sum_to_one():
    total = config.TRAIN_RATIO + config.VAL_RATIO + config.TEST_RATIO
    assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

def test_knowledge_base_loads():
    kb = load_knowledge_base()
    assert "brands" in kb
    assert "models" in kb
    assert "limitations" in kb


def test_context_lookup_known_model():
    kb = load_knowledge_base()
    ctx = find_context_for_sneaker("Air Jordan 1 Retro High", kb)
    assert "brand" in ctx or "_match_type" in ctx or ctx == kb["default"]


def test_context_lookup_unknown_model():
    kb = load_knowledge_base()
    ctx = find_context_for_sneaker("Totally Unknown Sneaker XYZ", kb)
    assert ctx is not None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def test_simple_prompt_structure():
    messages = build_simple_prompt("Air Jordan 1", 0.93, 670, 179, 2.74)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "$179" in messages[1]["content"]
    assert "+274%" in messages[1]["content"] or "274%" in messages[1]["content"]


def test_contextual_prompt_includes_context():
    kb = load_knowledge_base()
    ctx = find_context_for_sneaker("Air Jordan 1", kb)
    messages = build_contextual_prompt("Air Jordan 1", 0.93, 670, 179, 2.74, ctx)
    assert "Marktkontext" in messages[1]["content"]


def test_structured_prompt_requests_json():
    kb = load_knowledge_base()
    messages = build_structured_prompt("Yeezy 350", 0.85, 250, 220, 0.14, kb["default"])
    assert "JSON" in messages[1]["content"] or "json" in messages[1]["content"]


# ---------------------------------------------------------------------------
# Fallback recommendation
# ---------------------------------------------------------------------------

def test_fallback_strong_buy():
    rec = _fallback_recommendation("Air Jordan", 600, 200, 2.0)
    assert "STRONG BUY" in rec


def test_fallback_sell():
    rec = _fallback_recommendation("Bad Sneaker", 80, 200, -0.6)
    assert "SELL" in rec


def test_fallback_hold():
    rec = _fallback_recommendation("Average Sneaker", 195, 200, -0.025)
    assert "HOLD" in rec


# ---------------------------------------------------------------------------
# ML helpers
# ---------------------------------------------------------------------------

def test_brand_extraction_jordan():
    assert _extract_brand("Air Jordan 1 Retro High") == "Nike"


def test_brand_extraction_yeezy():
    assert _extract_brand("Yeezy Boost 350 V2") == "Adidas"


def test_brand_extraction_new_balance():
    assert _extract_brand("New Balance 550 White") == "New Balance"


def test_size_category():
    assert _size_category(7) == "small"
    assert _size_category(10) == "medium"
    assert _size_category(13) == "large"


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def test_eval_transforms_output_shape():
    from PIL import Image
    img = Image.new("RGB", (300, 400), color="red")
    transform = get_eval_transforms(224)
    out = transform(img)
    assert out.shape == (3, 224, 224)


def test_train_transforms_output_shape():
    from PIL import Image
    img = Image.new("RGB", (300, 400), color="blue")
    transform = get_train_transforms(224)
    out = transform(img)
    assert out.shape == (3, 224, 224)
