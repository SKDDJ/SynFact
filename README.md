# SynFact: Synthetic Data Generation for Knowledge Injection Benchmarks

A synthetic data generation engine for evaluating knowledge injection in LLMs via SFT and RL, supporting both text-only (SynFact-L) and multimodal (SynFact-VL) benchmarks.

## Installation

```bash
# Requires Python 3.11+
uv sync
```

## Benchmarks

### SynFact-L (Text-Only)

Generates fictional entities with structured relations, natural language descriptions, and QA pairs for evaluating factual knowledge grounding.

**Data generation:**
```bash
uv run python scripts/run_synthesis_l.py \
    --num-entities 1000 \
    --save-raw \
    --push-to-hub \
    --repo-id username/synfact-l
```

**Splits:**
- `train`: Full Description + Question → Answer
- `id_test`: Question → Answer (same QA as train, no context)
- `ood_test`: Question → Answer (implicit, inverse, multi-hop QA)

---

### SynFact-VL (Multimodal)

Generates fictional artworks with unique artists/styles, synthesized images, and visual QA pairs for evaluating visual knowledge grounding.

**Data generation:**
```bash
# With mock images (for testing)
uv run python scripts/run_synthesis_vl.py \
    --num-artworks 100 \
    --image-provider mock \
    --save-raw

# With real image generation
uv run python scripts/run_synthesis_vl.py \
    --num-artworks 100 \
    --image-provider nano_banana \
    --image-api-url "https://your-api.com" \
    --image-api-key "your-key" \
    --push-to-hub \
    --repo-id username/synfact-vl
```

**Splits:**
- `train`: Image + Full Caption + Question → Answer
- `id_test`: Image + Question → Answer (no caption)
- `ood_test_a`: Image + Unseen Questions → Answer
- `ood_test_b`: Artwork Title → Expected Caption (for reconstruction)

**Supported Image Providers:**
- `mock`: Placeholder images for testing
- `nano_banana`: Remote Nano-banana API
- `flux`: Local Flux model
- `qwen`: Qwen-Image model

---

## Project Structure

```
synfact/
├── config.py             # Configuration models
├── models.py             # SynFact-L data models
├── models_vl.py          # SynFact-VL data models
├── llm_client.py         # LLM API wrapper
├── image_client.py       # Image generation abstraction
├── generators/
│   ├── entity_generator.py      # L: Entity generation
│   ├── corpus_generator.py      # L: Description generation
│   ├── qa_generator.py          # L: QA generation
│   ├── artwork_generator.py     # VL: Artwork/artist generation
│   └── visual_qa_generator.py   # VL: Visual QA generation
├── pipeline.py           # L: Synthesis orchestrator
├── pipeline_vl.py        # VL: Synthesis orchestrator
├── export.py             # L: HuggingFace export
└── export_vl.py          # VL: HuggingFace export
```

## Configuration

Create a `.env` file:

```bash
SYNFACT_BASE_URL=https://your-llm-api.com/v1
SYNFACT_API_KEY=your-api-key
SYNFACT_MODEL_NAME=gpt-4o
```

## License

MIT
