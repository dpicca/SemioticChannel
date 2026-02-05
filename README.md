# Semiotic Channel Principle — Taboo Experiment

Implementation of the "Inverse Definition Game" (Taboo Semiotic) to validate the **Semiotic Channel Principle** from arXiv:2511.19550.

## Core Hypothesis

At increasing temperature (λ):
- **S (Semiotic Breadth)** increases — richer, more diverse descriptions
- **D (Decipherability)** follows an inverted-U curve — peaks at optimal λ, then collapses
- **C (Semiotic Capacity)** = max(D) — the optimal balance point

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TABOO SEMIOTIC EXPERIMENT                │
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
│  │  (Target)   │◀──────COMPARE──────────│   (Guess)     │  │
│  └─────────────┘                        └───────────────┘  │
│                                                             │
│  METRICS:                                                   │
│  • S = Self-BLEU diversity of descriptions                  │
│  • D = Cosine similarity (Target ↔ Interpretation)          │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
semiotic_channel/
├── concepts_dataset.py   # 100 concepts (50 concrete + 50 abstract)
├── llm_interface.py      # Agno/Ollama Emitter & Receiver
└── metrics.py            # S (Self-BLEU), D (cosine similarity)

experiments/
└── taboo_experiment.py   # Main experimental pipeline
```

## Quick Start

### Command Line

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run experiment
PYTHONPATH=. python3 experiments/taboo_experiment.py \
    --n-concepts 10 \
    --n-descriptions 5 \
    --model llama3:latest \
    --category abstract
```

### Web Interface (Recommended)

```bash
source venv/bin/activate
python3 web/server.py
```

Open **http://localhost:5000** in your browser. The interface provides:
- Real-time streaming of descriptions and interpretations
- Live Chart.js visualization of S and D curves
- Parameter controls for model, category, temperatures

## Requirements

- **Ollama** running locally (`ollama serve`)
- Models: `llama3:latest`, `nomic-embed-text:latest`
