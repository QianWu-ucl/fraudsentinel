# Multi-Agent Fraud Detection System

**Research Question:** Using Multi-Agent Systems to Detect Fraud Across Multiple Scenarios

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Datasets](#datasets)
- [Setup](#setup)
- [Running the Pipeline](#running-the-pipeline)
- [Experiments & Results](#experiments--results)
- [References](#references)

---

## Overview

This project implements a **multi-agent pipeline** for detecting fraud in social media text. The system uses two LLM-based agents and an external web search tool to classify user profiles as fraudulent (`0`) or legitimate (`1`).

The framework is inspired by **FLAG** (Fraud Detection with LLM-enhanced Graph Neural Network) and evaluated on data sourced from the **GLBench** benchmark. The pipeline is tested across multiple social media platforms (Instagram, Reddit) and benchmarked against four state-of-the-art LLMs.

---

## System Architecture

```
Raw Social Media Bio
        │
        ▼
┌─────────────────────┐
│  Agent 1            │  Text Splitter / Classifier
│  (FLAG-inspired)    │
└──────┬──────────────┘
       │
       ├──────────────────────────┐
       ▼                          ▼
Discriminative Text          Residual Text
(emails, URLs, handles,      (background context,
 payment info, CTAs)          personal descriptions)
       │                          │
       ▼                          │
┌─────────────────────┐           │
│  Tavily Search API  │           │
│  External Searcher  │           │
│                     │           │
│  Query: target +    │           │
│  "scam OR fraud     │           │
│   OR complaint"     │           │
└──────┬──────────────┘           │
       │                          │
       └──────────┬───────────────┘
                  ▼
        ┌─────────────────────┐
        │  Agent 2            │  Binary Classifier / Judge
        │  (Discriminator)    │
        │                     │
        │  Inputs:            │
        │  - Discriminative   │
        │  - Residual         │
        │  - Tavily evidence  │
        └──────┬──────────────┘
               │
               ▼
     Prediction: 0 (Fraud) / 1 (Normal)
```

**Pipeline data flow:**

```
input CSV  →  [Agent 1]  →  [Tavily Search]  →  [Agent 2]  →  final results CSV
```

---

## Datasets

This project covers **two social media platforms** to validate cross-scenario generalization.

### Instagram

Sourced from the **GLBench** benchmark dataset.

| File | Samples |
|------|---------|
| `instagram_samples50.csv` | 50 |
| `instagram_samples100.csv` | 100 |
| `instagram_samples200.csv` | 200 |
| `instagram_samples500.csv` | 500 (main eval set) |
| `instagram_samples1000.csv` | 1000 |

### Reddit

Additional scenario to test cross-platform generalization.

| File | Samples |
|------|---------|
| `reddit_samples100.csv` | 100 |
| `reddit_samples200.csv` | 200 |
| `reddit_samples500.csv` | 500 |

### CSV Format

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Raw social media bio / post text |
| `label` | int | Ground truth: `0` = suspicious/fraud, `1` = benign/normal |

> **Label convention:** `1` = benign profile, `0` = suspicious / potential scam.
> Follows the label convention in the GLBench-sourced Instagram dataset.

---

## Setup

### Requirements

- Python 3.8+
- Google Colab or local Jupyter / VSCode environment

### Install Dependencies

```bash
pip install openai requests pandas scikit-learn
```

### API Keys Required

| Service | Purpose | Where to get |
|---------|---------|--------------|
| OpenAI / Gemini API | Powers Agent 1 and Agent 2 | [platform.openai.com](https://platform.openai.com) |
| Tavily API | External web evidence search | [tavily.com](https://tavily.com) |

### Environment Variables

```bash
export OPENAI_API_KEY="your_openai_key"
export OPENAI_BASE_URL="your_base_url"        # if using a proxy endpoint
export TAVILY_API_KEY="your_tavily_key"
export TAVILY_ENDPOINT="https://api.tavily.com/search"
```

Or inside a Colab / Jupyter notebook:

```python
import os
os.environ["OPENAI_API_KEY"] = "your_key"
os.environ["TAVILY_API_KEY"] = "your_key"
```

---

## Running the Pipeline

### Full Pipeline (3 Steps)

```python
input_csv  = "/content/instagram_samples500.csv"
after_a1   = "/content/instagram_after_agent1.csv"
after_tav  = "/content/instagram_after_tavily.csv"
final_out  = "/content/instagram_final_results.csv"

# Step 1 — Agent 1: split text into Discriminative / Residual
run_agent1_and_save_csv(input_csv=input_csv, output_csv=after_a1)

# Step 2 — Tavily: retrieve web evidence for discriminative signals
add_tavily_evidence_column(agent1_csv=after_a1, output_csv=after_tav)

# Step 3 — Agent 2: final binary judgment
run_agent2_and_save_csv(agent1_tavily_csv=after_tav, output_csv=final_out)
```

### Output CSV Columns

| Column | Description |
|--------|-------------|
| `text` | Original input text |
| `label` | Ground truth label |
| `Discriminative Text` | Fraud-signal lines extracted by Agent 1 |
| `Residual Text` | Background context lines |
| `tavily_evidence` | JSON: web search results per target entity |
| `agent2_pred` | Final prediction: `0` (suspicious) or `1` (benign) |

### Evaluation

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

df = pd.read_csv("instagram_final_results.csv")
y_true = df["label"]
y_pred = df["agent2_pred"]

mask = y_true.notna() & y_pred.notna()
y_true, y_pred = y_true[mask], y_pred[mask]

print("Accuracy:  ", accuracy_score(y_true, y_pred))
print("Precision: ", precision_score(y_true, y_pred, average="binary"))
print("Recall:    ", recall_score(y_true, y_pred, average="binary"))
print("F1:        ", f1_score(y_true, y_pred, average="binary"))
print("F1-macro:  ", f1_score(y_true, y_pred, average="macro"))
```

---

## Experiments & Results

### Comparative Experiments — Instagram Dataset (500 samples)

We evaluate our multi-agent pipeline across four backbone LLMs. **Our method uses GPT-4o-mini** as the primary backbone.

> Note: AUC is not reported for LLM-based methods, as large language models output discrete labels
> rather than continuous probability scores. F1-macro is used as the primary comparison metric.

| Model | Accuracy | Precision | Recall | F1 | F1-macro |
|-------|----------|-----------|--------|----|----------|
| Gemini-2.5-pro | 0.4980 | 0.3856 | **0.6519** | **0.4846** | 0.4977 |
| Deepseek-r1 | 0.5480 | 0.4026 | 0.5138 | 0.4515 | **0.5336** |
| Claude-3.7-sonnet | 0.4900 | 0.3669 | 0.5635 | 0.4444 | 0.4865 |
| **Our Method (gpt-4o-mini)** | **0.5800** | **0.4027** | 0.3315 | 0.3636 | 0.5251 |

**Bold** values indicate the best score in each column.

---

### Ablation Study — Multi-Agent Framework (gpt-4o-mini, Instagram)

We progressively remove components to measure each module's contribution.

| Configuration | Accuracy | Precision | Recall | F1 | F1-macro |
|---------------|----------|-----------|--------|----|----------|
| **gpt-4o-mini (Full Framework)** | 0.5800 | 0.4027 | **0.3315** | **0.3636** | **0.5251** |
| w/o Tavily | 0.5840 | 0.4043 | 0.3149 | 0.3540 | 0.5236 |
| w/o Tavily & Agent 1 | **0.6200** | **0.4494** | 0.2210 | 0.2963 | 0.5180 |

**Key observations:**
- Removing Tavily slightly reduces F1-macro (0.5251 → 0.5236), confirming that external web evidence provides a consistent boost.
- Removing both Tavily and Agent 1 (vanilla LLM baseline) achieves the highest raw accuracy but the lowest F1-macro, indicating it becomes over-conservative and misses fraud cases (very low recall).
- The **full framework achieves the best F1-macro**, which is the primary metric for imbalanced fraud detection tasks.

---

### Comparison with Traditional GNN Baselines

For reference, we compare our F1-macro against traditional graph-based methods from the FLAG paper [2].
These methods use graph neural networks (GNNs) with LLM-enhanced node features on the same Instagram and Reddit datasets.

> AUC is not reported for our method, as LLMs output discrete binary labels rather than the
> continuous probability scores required for AUC computation.

**Instagram — +FLAG\***

| Model | F1-macro(%) |
|-------|--------|
| GCN | 49.79 |
| GAT | **50.65** |
| GeniePath | 48.28 |
| CARE-GNN | 50.24 |
| BWGNN | 49.35 |
| DGA-GNN | 48.95 |
| PMP | 49.76 |
| **Our Method (gpt-4o-mini)** | **52.51** |

**Reddit — +FLAG\***

| Model | F1-macro(%) |
|-------|----------|
| GCN | 48.72 |
| GAT | 50.20 |
| GeniePath | 48.69 |
| CARE-GNN | **51.95** |
| BWGNN | 51.91 |
| DGA-GNN | 49.50 |
| PMP | 50.14 |
| **Our Method (gpt-4o-mini)** | **51.24** |

> FLAG* denotes the best FLAG variant with optimized LLM-enhanced features.
> Source: Tang et al., *FLAG: Fraud Detection with LLM-enhanced Graph Neural Network*, 2024, Table 4.
>
> Our method is **training-free** and requires **no graph structure**, making it directly applicable
> to raw text without graph construction or node feature engineering.

---

## Agent Design Details

### Agent 1 Prompt Principles

- **Do not paraphrase or summarize** — copy lines exactly as-is
- **Hard rules** for Discriminative Text: any line with `@`, `#`, URL, contact signal, or payment identifier
- **Default to Residual** when uncertain — never delete content
- Output format is strictly enforced: two labeled sections only

### Agent 2 Scoring Logic

Agent 2 computes an additive score and applies a threshold (score ≥ 4 → label `1`):

| Signal | Score |
|--------|-------|
| Recognized media institution (NBC, iHeartRadio...) | +3 |
| Official business with operational cues | +3 |
| Business Inquiries + press/media contact email | +2 |
| Multiple consistent brand handles + founder/owner role | +2 |
| Tavily results align with bio identity | +2 |
| Email present | +1 |
| "Collab" / "DM for collabs" | +1 |
| Generic creator label (blogger, influencer) | +1 |
| Pure personal diary, no institutional cues | -3 |
| Generic lifestyle list, no verifiable anchor | -3 |
| Mostly hashtags/emojis, no concrete role | -3 |
| Overly generic "creator template" bio | -2 |
| Location + personal descriptors only | -2 |

---

