# Research Note: GLBench Benchmark Analysis

**Date:** 2025-12-13
**Paper:** *GLBench: A Comprehensive Benchmark for Graph with Large Language Models* (NeurIPS 2024)
**Topic:** Datasets, Baselines, Evaluation

## Key Takeaways
Found this comprehensive benchmark paper. It’s super useful for setting up my experiments. They evaluated 3 types of methods: Enhancer, Predictor, and Aligner.

* **Findings:** They found that "LLM-as-Predictor" (asking LLM directly) is unstable and has output format issues. This confirms why I need a **Judge Agent** + **External Search** in my architecture, not just a bare LLM.
* **My Solution (Multi-Agent System):** This confirms why I need a **Collaborative Multi-Agent Architecture** instead of a single model:
    1.  **Extractor LLM:** Splits text into Discriminative/Residual (concept from FLAG paper).
    2.  **External Searcher:** Uses the Discriminative text to find verified external info.
    3.  **Judge LLM:** Makes the final decision based on the aggregated evidence.
    * *Why this works:* This pipeline mitigates the "unstable output" issue mentioned in GLBench by grounding the LLM with retrieved context.

## Dataset Candidates
I should use the **Text-Attributed Graphs (TAGs)** mentioned here to ensure my results are comparable:
1.  **Reddit:** Social network data. Good for checking user posts.
2.  **Instagram:** Good for checking user profiles/introductions.
*Action:* Check if I can download these directly from their GitHub or if I need to preprocess them.

## Baselines to Run
For the "Evaluation" section of my thesis, I need to compare my Multi-Agent system against these:
* **Standard GNNs:** GraphSAGE, GAT (classic baselines).
* **SoTA GraphLLM:** OFA or GLEM (as strong baselines).

## Next Steps
- [ ] Clone the GLBench repo to see data format.
- [ ] Select 2 datasets (likely Reddit & Instagram) for my fraud detection demo.


# Methodology Note: FLAG Implementation Details

**Date:** 2025-12-17
**Paper:** *FLAG: Fraud Detection with LLM-enhanced Graph Neural Network* (KDD 2025)
**Topic:** Core Tech Stack (Text Classifier & Prompting)

## Why this paper?
This is basically the reference implementation for my **Text Classifier** module. They solve the "Neighborhood Camouflage" problem (fraudsters hiding among normal users), which is exactly what my Judge Agent needs to detect.

## Implementation Ideas (To-Do)

### 1. Prompt Engineering
They use a specific prompt strategy to split text into "Discriminative" and "Residual". I need to adapt their prompts for my Agent.
* **Discriminative Prompt:** Ask LLM to extract features *only* related to fraud/risk.
* **Residual Prompt:** Ask LLM to keep the background noise/irrelevant info.
* *Idea:* I can add my "External Search" results into the context of the Discriminative Prompt to make it even stronger.

### 2. Model Architecture
* **Sampling:** Instead of random sampling, they use "Semantic Similarity Sampling". This filters out the camouflaged neighbors.

## For my Thesis
I can cite this to support why I'm splitting the text analysis.
> "LLM-based node enhancement extracts discriminative text... enhancing robustness against camouflaged neighbors."

## Action Items
- [ ] Draft the "Discriminative Text" system prompt in my codebase.
- [ ] Test if this prompt works on the Reddit dataset from GLBench.


# Literature Review Note: Anomaly Detection Survey

**Date:** 2025-12-18
**Paper:** *Financial Fraud: A Review of Anomaly Detection Techniques and Recent Advances* (Hilal et al., 2022)
**Topic:** Baseline Selection & Research Motivation

## Why this paper?
This is the foundational survey for my thesis. I need it to:
1.  List the "traditional" methods I am competing against.
2.  Justify why I'm building a complex Multi-Agent system instead of just using a simple Neural Network.

## Key Insights for My Thesis argument

### 1. The "Black Box" Problem
The paper points out that Deep Learning models (like CNNs/LSTMs) are powerful but lack interpretability.
* *My Argument:* This is the perfect justification for my **Judge Agent**. My agent doesn't just output a score; it uses **External Search** to find evidence and provides a reasoning process. I can cite this paper to say "Previous methods fail to explain 'why', but my system does."

### 2. Validation of Graph Methods
It confirms that Graph-based methods are essential for detecting organized fraud (rings), which supports my choice of using **FLAG (Graph + LLM)** as the classifier.

## Baselines to Implement
Based on their review, I should select these standard models as baselines to ensure my experiments are convincing:
* **Traditional:** SVM / Random Forest (simple benchmarks).
* **Deep Learning:** LSTM (for sequence data) or Autoencoders (unsupervised).
* *Note:* I will compare my Multi-Agent system against these to show the improvement in both accuracy and explainability.

## Action Items
- [ ] Write the "Related Work" section draft using this paper's taxonomy.
- [ ] Add LSTM/SVM to my baseline experiment list.


# Methodology Note: Multi-Agent Debate Theory
**Date:** 2026-01-04
**Paper:** *Improving Factuality and Reasoning in Language Models through Multiagent Debate* (Du et al., 2023)
**Topic:** Theoretical Foundation for Multi-Agent Collaboration

## Why this paper?
After my initial experiments with GLBench (Dec 13 Note) showed that a single LLM is unstable for graph tasks, I needed a way to stabilize the output without fine-tuning. This paper is the theoretical backing I was looking for. It proves that having agents "debate" or critique each other significantly reduces hallucinations compared to a single model.

## My Critical Thinking (The "Aha" Moment)
I initially thought I needed complex Reinforcement Learning to fix the hallucinations. But this paper shows that a simple **"Society of Minds"** approach works—just letting instances talk to each other improves accuracy.
* *Concern:* My only worry is cost and latency. If I make them debate for too many rounds, the system becomes too slow.
* *Observation:* Figure 2 shows that accuracy plateaus after about 3 rounds. This gives me a concrete parameter constraint for my system configuration.

## Implementation Decision
I originally planned a linear pipeline (Classifier -> Output). Based on this paper, I am changing the architecture to a **"Verification Loop"**:
1.  **Agent 1 (Classifier):** Proposes "Suspicious".
2.  **Agent 2 (Judge):** Challenges this with External Search results ("But verify_search found X...").
3.  **Agent 1:** Must revise or defend its position.

This effectively implements a **Fact-Checking Protocol** inside the graph traversal process.

## Thesis Connection
I will use this to defend my system against the "hallucination" critique in the Methodology chapter.
> "Addressing the instability found in single-pass LLMs, we adopt the Multi-Agent Debate framework (Du et al., 2023) to enforce self-correction via an external Judge Agent."


# Methodology Note: Enhancing Reasoning with RAP

**Date:** 2026-01-05
**Paper:** *Reasoning with Language Model is Planning with World Model* (RAP) (EMNLP 2023)
**Topic:** Advanced Reasoning Strategy (Planning vs. Reacting)

## Why I read this
I realized that standard RAG (Retrieval Augmented Generation) is too "reactive"—the agent just searches for keywords found in the text. I needed a way to make my agents **proactive**.
This paper proposes using **MCTS (Monte Carlo Tree Search)** to let the LLM "plan" ahead. It treats reasoning not as a straight line, but as a tree of possibilities.



## Critical Thinking (The Trade-off)
While the paper achieves SOTA on math tasks using MCTS, running a full tree search for every node in my Fraud Detection graph is **computationally impossible** (too slow/expensive).
* *However:* The core idea of "simulating future states" (World Model) is usable. I don't need the full tree; I just need the **"Lookahead"** capability.

## Implementation Decision
I will adapt the RAP framework into a simplified **"Hypothesis Testing"** workflow for my **External Searcher Agent**:
1.  **Hypothesis Generation:** Instead of searching immediately, the agent first asks: "If this user were a fraudster, what digital footprint *should* exist?" (e.g., 'If fraud, look for exact matches of this phone number on blacklists').
2.  **Targeted Search:** The search queries are generated based on this simulation, not just the raw text.
3.  **Verification:** The Judge then compares the *expected* evidence vs. *actual* findings.

## Thesis Connection
This allows me to claim that my system uses **"System 2 Thinking"** (slow, deliberate planning) rather than just "System 1" (fast, intuitive response). This is a strong argument for why my method reduces hallucinations in complex fraud cases.

## Engineering Choice
I will implement a light version: not a full MCTS, but a **"Plan-then-Search"** prompt chain. This captures the benefit of RAP without the latency overhead.