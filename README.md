# SynFact-L

Synthetic data generation engine for the SynFact-L benchmark to evaluate knowledge injection in LLMs via SFT and RL.

## Installation

```bash
uv sync
```

## Usage

```bash
# Generate 5 sample entities for testing
uv run python scripts/run_synthesis.py --num-entities 5 --output ./output --save-raw

# Full generation (1000 entities)
uv run python scripts/run_synthesis.py --num-entities 1000 --output ./output --save-raw

# Push to HuggingFace Hub
uv run python scripts/run_synthesis.py --num-entities 1000 --push-to-hub --repo-id Shiym/SynFact-L
```

## Project Structure

```
synfact/
├── config.py        # Configuration settings
├── models.py        # Pydantic data models
├── llm_client.py    # LLM wrapper with retry
├── generators/      # Data generators
│   ├── entity_generator.py
│   ├── corpus_generator.py
│   └── qa_generator.py
├── pipeline.py      # Pipeline orchestrator
└── export.py        # HuggingFace export
```
