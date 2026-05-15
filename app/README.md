---
title: SneakerScout
emoji: "S"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.31.5
app_file: app.py
pinned: false
license: mit
---

# SneakerScout

AI-Powered Sneaker Recognition & Resell Advisor.

Upload a sneaker photo. The app classifies the model (ViT), predicts a fair resell price (gradient boosting), and generates a buy/hold/sell recommendation with market context (GPT-4o-mini).

## Secrets

Set `OPENAI_API_KEY` in Space Settings → Secrets.

## Local run

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python app.py
```
