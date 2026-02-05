# Semiotic Channel Principle — Taboo Experiment

Implementation of the "Inverse Definition Game" (Taboo Semiotic) to validate the **Semiotic Channel Principle** from arXiv:2511.19550.

## Core Hypothesis

At increasing temperature (λ):
- **S (Semantic Diversity)** increases — richer, more diverse descriptions.
- **D (Decipherability)** follows an inverted-U curve — peaks at optimal λ, then collapses.
- **C (Semiotic Capacity)** = max(D) — the optimal balance point.

## Advanced Metrics

1.  **Semantic Diversity (S)**: Measures the semantic spread of generated descriptions. Calculated as the mean pairwise cosine distance between embedding vectors (using `nomic-embed-text`). More robust than n-gram overlap.
2.  **Ranked Decipherability (D)**: Measures the Receiver's ability to identify the concept. Uses **Mean Reciprocal Rank (MRR)** on the Top-5 guesses.
3.  **Communicative Efficiency (E)**: Measures information density. Calculated as `1 / length(description)` for successful turns (Rank 1), otherwise 0.

## Experimental Rationale

### Why Pairs? (Cross-Model Evaluation)
To validate that the Semiotic Channel is a fundamental property of language models rather than an artifact of a specific architecture, we test pairs of models:
- **Self-Play (e.g., Llama-3 → Llama-3)**: Establishes a baseline for communicative capacity within a shared latent space.
- **Cross-Play (e.g., Llama-3 → Gemma-2)**: Tests the universality of the generated descriptions. If a concept described by Model A is decipherable by Model B, the semiotic representation is robust and transferable.

### Why these Metrics?

- **Semantic Diversity (S)** [Range: 0.0 - 2.0]: Captures the *conceptual breadth* explored by the Sender. By measuring dispersion in embedding space (**Cosine Distance**, where 0 is identical and 2 is opposite), we quantify how much of the target's semantic neighborhood is "mapped" by the model's descriptions, ensuring that higher temperatures produce meaningfully distinct perspectives rather than just lexical noise.
- **Ranked Decipherability (D)** [Range: 0.0 - 1.0]: Serves as the *functional throughput* of the semiotic channel. Using **Mean Reciprocal Rank (MRR)** provides a continuous gradient of communicative success, measuring the Receiver's ability to isolate the target concept from a high-dimensional latent space.
- **Communicative Efficiency (E)** [Range: 0.0 - 1.0]: Grounded in the principle of **Least Effort**, this evaluates the information density of the signal. It rewards "semiotic elegance"—achieving high decipherability with minimal linguistic tokens—and penalizes redundant or "rambling" descriptions that increase cognitive load without adding information.

### Interpreting the Results

| State | S (Diversity) | D (Decipherability) | Interpretation |
|-------|---------------|---------------------|----------------|
| **Rigidity** | Low | High/Low | The model repeats the same few descriptions. High D means it found one good description and stuck to it (safe but limited). |
| **Optimal** | High | High | **The Sweet Spot**. The model generates varied, diverse descriptions, and the Receiver understands all of them. This indicates a robust Semiotic Channel. |
| **Chaos** | High | Low | The model is "creative" (high entropy) but incoherent. It hallucinates or drifts so far that the Receiver cannot map the signal back to the concept. |

**Efficiency Factor**:
- **High D + Low E**: The model "rambles". It eventually conveys the meaning, but uses too many words.
- **High D + High E**: **Semiotic Mastery**. The model conveys the concept concisely and accurately.

### Semiotic Channel Computation

The "Semiotic Channel" is computed through a pipeline that maps the relationship between signal **Breadth (S)** and **Decipherability (D)**:

1.  **Signal Generation (Emitter)**: For a target concept (e.g., "Clock"), the **Emitter** generates a set of descriptions while varying the temperature ($\lambda$). Higher temperatures lead to more creative and divergent descriptions.
2.  **Measuring Breadth (S)**: The **Semiotic Breadth ($S$)** is a measured property of the generated text. It is calculated as the **mean pairwise cosine distance** between the embeddings of the descriptions.
    - **Formula**: $S = \text{mean}(1 - \text{cosine\_similarity}(desc_i, desc_j))$.
3.  **Transmission & Decoding (Receiver)**: Descriptions are sent to the **Receiver**, which produces its **Top-5** guesses to identify the target concept.
4.  **Measuring Decipherability (D)**: **Decipherability ($D$)** measures communicative success using **Mean Reciprocal Rank (MRR)**. For each description, we find the rank of the target concept in the Receiver's top-5 guesses.
    - **Turn Score**: $1 / \text{rank}$ (e.g., 1.0 if 1st choice, 0.5 if 2nd, 0 if not found).
    - **Final $D$**: The average MRR score across all turns for a given temperature.
5.  **Channel Capacity (C)**: The **Semiotic Profile** is the curve of $D$ as a function of $S$. The **Capacity ($C$)** is defined as the **maximum value of $D$** ($\max D$) reached across the spectrum of $S$, representing the optimal balance point for communication.

## Experiment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              ROBUST TABOO SEMIOTIC EXPERIMENT               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │   EMITTER   │───▶│  DESCRIPTION │───▶│   RECEIVER    │  │
│  │  (LLM @ λ)  │    │   (Message)  │    │ (LLM @ 0.2)   │  │
│  └─────────────┘    └──────────────┘    └───────────────┘  │
│         │                                       │          │
│         ▼                                       ▼          │
│  ┌─────────────┐                        ┌───────────────┐  │
│  │   Concept   │                        │ Interpretation│  │
│  │  (Target)   │◀──────RANKING──────────│    (Top-5)    │  │
│  └─────────────┘                        └───────────────┘  │
│                                                             │
│  LOGGING: experiments/logs/*.jsonl (Turn-level data)        │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
semiotic_channel/
├── concepts_dataset.py   # 100 concepts + External file support
├── llm_interface.py      # Agno/Ollama Interface
└── metrics.py            # Diversity, MRR, Efficiency, Entropy

experiments/
├── taboo_experiment.py   # Main CLI pipeline (Cross-Model Capable)
├── results/              # Aggregated JSON results & Plots
└── logs/                 # JSONL files (Turn-level logs)

web/
└── server.py             # Flask Web Interface
```

## Quick Start

### 1. Scientific Experiment (CLI)
Run a rigorous experiment with turn-level logging and advanced metrics.

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run full experiment (Self-Play)
# Requires Ollama running
export PYTHONPATH=$PYTHONPATH:.
python3 experiments/taboo_experiment.py \
    --n-concepts 50 \
    --n-descriptions 10 \
    --model llama3:latest
```

This will produce:
- **Log**: `experiments/logs/experiment_turns_YYYYMMDD_HHMMSS.jsonl`
- **Result**: `experiments/results/experiment_YYYYMMDD_HHMMSS.json`
- **Plot**: `semiotic_profile_comparison.png`

### 2. Web Interface (Demo)
Visual interactive mode for demonstrations.

```bash
source venv/bin/activate
python3 web/server.py
```
Open **http://localhost:5000**.

## Requirements

- **Ollama** running locally (`ollama serve`)
- Models: `llama3:latest` (or others), `nomic-embed-text:latest` (for embeddings)
- Python 3.10+
- Dependencies: `agno`, `scikit-learn`, `flask`, `numpy`, `matplotlib`
