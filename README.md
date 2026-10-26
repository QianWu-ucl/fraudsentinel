# FraudSentinel

**A training-free multi-agent LLM framework that detects fraudulent social media profiles from raw text alone — no graph structure, no model training.**

On the GLBench Instagram dataset, FraudSentinel reaches an **F1-macro of 52.51%**, a **3.7% relative improvement** over the best GNN baseline reported in the FLAG paper (GAT, 50.65%), while requiring no graph construction or node feature engineering.

---

## Table of Contents

- [Why This Project](#why-this-project)
- [How It Works](#how-it-works)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Agent Design Details](#agent-design-details)
- [Limitations and Future Work](#limitations-and-future-work)
- [References](#references)

---

## Why This Project

State-of-the-art fraud detection on social platforms relies on graph neural networks, which need the social graph (followers, interactions) and task-specific training. In many real settings that graph is unavailable, incomplete or expensive to build.

FraudSentinel asks a simpler question: **can LLM agents, given only a profile's text and live web evidence, match graph-based methods?**

---

## How It Works

The pipeline uses two LLM agents and an external web search tool:

```
Raw Social Media Bio
        │
        ▼
┌─────────────────────┐
│  Agent 1            │  Splits text into fact-checkable signals and context
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
│  Real-time evidence │           │
│  (target + "scam OR │           │
│  fraud OR complaint")│          │
└──────┬──────────────┘           │
       │                          │
       └──────────┬───────────────┘
                  ▼
        ┌─────────────────────┐
        │  Agent 2            │  Combines all signals via a scoring
        │  (Judge)            │  prompt and outputs a binary label
        └──────┬──────────────┘
               ▼
     Prediction: 0 (Suspicious) / 1 (Benign)
```

1. **Agent 1** separates *discriminative* lines (handles, emails, URLs, payment identifiers, calls to action) from *residual* context, copying text verbatim so no evidence is lost.
2. **Tavily Search** verifies each discriminative target against the live web, grounding the decision in external evidence to reduce LLM hallucination.
3. **Agent 2** combines the discriminative text, residual context and web evidence through an iteratively designed additive scoring prompt, then outputs the final label.

---

## Results

All experiments use 500-sample evaluation sets. **F1-macro** is the primary metric because the task is class-imbalanced. AUC is not reported, since the LLMs output discrete labels rather than probability scores.

### Comparison with GNN Baselines

Baseline figures are taken from the FLAG paper [1], Table 4 (FLAG\* = best FLAG variant with LLM-enhanced features).

| Model | Instagram F1-macro (%) | Reddit F1-macro (%) |
|-------|------------------------|---------------------|
| GCN | 49.79 | 48.72 |
| GAT | 50.65 | 50.20 |
| GeniePath | 48.28 | 48.69 |
| CARE-GNN | 50.24 | **51.95** |
| BWGNN | 49.35 | 51.91 |
| DGA-GNN | 48.95 | 49.50 |
| PMP | 49.76 | 50.14 |
| **FraudSentinel (gpt-4o-mini)** | **52.51** | 51.24 |

On Instagram, FraudSentinel outperforms every GNN baseline. On Reddit it is competitive, ranking third behind CARE-GNN and BWGNN, without using any graph structure or training.

### Backbone LLM Comparison (Instagram)

The same pipeline was run with four backbone LLMs. gpt-4o-mini was used as the main backbone.

| Backbone | Accuracy | Precision | Recall | F1 | F1-macro |
|----------|----------|-----------|--------|----|----------|
| Gemini-2.5-pro | 0.4980 | 0.3856 | **0.6519** | **0.4846** | 0.4977 |
| DeepSeek-R1 | 0.5480 | 0.4026 | 0.5138 | 0.4515 | **0.5336** |
| Claude-3.7-Sonnet | 0.4900 | 0.3669 | 0.5635 | 0.4444 | 0.4865 |
| gpt-4o-mini | **0.5800** | **0.4027** | 0.3315 | 0.3636 | 0.5251 |

### Ablation Study (gpt-4o-mini, Instagram)

| Configuration | Accuracy | Precision | Recall | F1 | F1-macro |
|---------------|----------|-----------|--------|----|----------|
| Full framework | 0.5800 | 0.4027 | **0.3315** | **0.3636** | **0.5251** |
| w/o Tavily | 0.5840 | 0.4043 | 0.3149 | 0.3540 | 0.5236 |
| w/o Tavily & Agent 1 (single LLM) | **0.6200** | **0.4494** | 0.2210 | 0.2963 | 0.5180 |

**Observations:**
- The single-LLM baseline has the highest accuracy but the lowest recall: it becomes over-conservative and misses most fraud cases.
- Adding Agent 1 and then Tavily each raises recall and F1-macro, with the full framework scoring best on F1-macro.
- The F1-macro gains between configurations are small, so further runs on larger samples would be needed to confirm them (see [Limitations](#limitations-and-future-work)).

---

## Repository Structure

```
├── src/
│   ├── final/                 # Final pipeline notebooks (Instagram, Reddit)
│   └── methods/               # Data loading utilities
├── Experiments/
│   ├── Comparative Experiments/   # Backbone LLM comparison
│   └── Ablation Studies/          # Component ablations
├── exploratory_experiments/   # Earlier pipeline iterations (v1–v4)
├── docs/                      # Literature notes and references
└── README.md
```

---

## Getting Started

### Requirements

- Python 3.8+
- Jupyter, VS Code or Google Colab

```bash
pip install openai requests pandas scikit-learn
```

### API Keys

| Service | Purpose |
|---------|---------|
| OpenAI-compatible API | Powers Agent 1 and Agent 2 |
| [Tavily](https://tavily.com) | Real-time web evidence search |

```bash
export OPENAI_API_KEY="your_openai_key"
export OPENAI_BASE_URL="your_base_url"   # optional, for proxy endpoints
export TAVILY_API_KEY="your_tavily_key"
```

### Data

Datasets are **not included** in this repository. The Instagram and Reddit data come from the [GLBench](https://github.com/NineAbyss/GLBench) benchmark. Each CSV needs two columns: `text` (raw profile text) and `label` (`1` = benign, `0` = suspicious).

### Running the Pipeline

```python
run_agent1_and_save_csv(input_csv="instagram_samples500.csv",
                        output_csv="after_agent1.csv")        # Step 1: split signals
add_tavily_evidence_column(agent1_csv="after_agent1.csv",
                           output_csv="after_tavily.csv")      # Step 2: web evidence
run_agent2_and_save_csv(agent1_tavily_csv="after_tavily.csv",
                        output_csv="final_results.csv")        # Step 3: final judgment
```

Evaluation uses scikit-learn's `accuracy_score`, `precision_score`, `recall_score` and `f1_score` (with `average="macro"` for F1-macro) on the `label` and `agent2_pred` columns.

---

## Agent Design Details

### Agent 1 Prompt Principles

- Copy lines verbatim: never paraphrase or summarise.
- Hard rules for discriminative text: any line with `@`, `#`, a URL, a contact signal or a payment identifier.
- Default to residual when uncertain, and never delete content.
- Strictly enforced output format with two labelled sections.

### Agent 2 Scoring Logic

Agent 2 computes an additive score; a score of 4 or more gives label `1` (benign).

| Signal | Score |
|--------|-------|
| Recognised media institution | +3 |
| Official business with operational cues | +3 |
| Business inquiries plus press/media contact email | +2 |
| Multiple consistent brand handles plus founder/owner role | +2 |
| Tavily results align with bio identity | +2 |
| Email present | +1 |
| "Collab" / "DM for collabs" | +1 |
| Generic creator label (blogger, influencer) | +1 |
| Pure personal diary, no institutional cues | −3 |
| Generic lifestyle list, no verifiable anchor | −3 |
| Mostly hashtags/emojis, no concrete role | −3 |
| Overly generic "creator template" bio | −2 |
| Location plus personal descriptors only | −2 |

---

## Limitations and Future Work

- **Evaluation size:** results are from 500-sample sets and single runs; LLM outputs are non-deterministic, so repeated runs and confidence intervals would strengthen the conclusions.
- **Absolute performance:** F1-macro around 0.5 shows the task remains hard for all methods, graph-based or not.
- **Hand-designed scoring:** Agent 2's weights were designed iteratively by hand; learning or calibrating them could improve robustness.
- **Next steps:** wider platform coverage, cost and latency analysis per backbone, and combining text-based agents with graph signals where a graph is available.

---

## References

[1] Tang et al., *FLAG: Fraud Detection with LLM-enhanced Graph Neural Network*, 2024.

[2] Li et al., *GLBench: A Comprehensive Benchmark for Graph with Large Language Models*, 2024.
