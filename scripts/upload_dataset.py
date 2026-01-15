#!/usr/bin/env python3
"""Upload existing SynFact-L dataset to HuggingFace Hub with a Dataset Card."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import os

from dotenv import load_dotenv
from rich.console import Console

from synfact.models import SynthesizedEntity
from synfact.export import export_to_huggingface

console = Console()
logging.basicConfig(level="INFO", format="%(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def generate_dataset_card(
    repo_id: str,
    num_entities: int,
    total_direct_qa: int,
    total_ood_qa: int,
    config_summary: dict[str, Any]
) -> str:
    """Generate a README.md (Dataset Card) for the Hugging Face repository."""
    card = f"""---
        license: mit
        task_categories:
        - question-answering
        - text-generation
        language:
        - en
        tags:
        - synthetic
        - knowledge-injection
        - reasoning
        - synfact
        size_categories:
        - 1K<n<10K
        ---

        # SynFact-L Dataset: {repo_id.split('/')[-1] if '/' in repo_id else repo_id}

        This dataset was generated using the **SynFact** engine (Synthetic Factual Knowledge).
        It contains fictional entities with structured relations, natural language descriptions, and question-answer pairs designed to evaluate:
        1.  **Memorization**: Storing facts from context (Direct QA).
        2.  **Reasoning**: Inferring new facts via multi-hop logic (OOD QA).

        ## Dataset Statistics

        - **Entities**: {num_entities}
        - **Total Samples**: {total_direct_qa + total_ood_qa + total_direct_qa} (Train + ID Test + OOD Test)
            - **Train Samples**: {total_direct_qa}
            - **ID Test Samples**: {total_direct_qa}
            - **OOD Test Samples**: {total_ood_qa}

        ## Configuration

        | Parameter | Value |
        | :--- | :--- |
        | **Max Reasoning Hops** | {config_summary.get('max_reasoning_hops', 'N/A')} |
        | **Min Relations** | {config_summary.get('min_relations', 'N/A')} |
        | **Max Relations** | {config_summary.get('max_relations', 'N/A')} |
        | **Corpus Length** | {config_summary.get('corpus_length', 'N/A')} |

        ## Usage Guide

        ### 1. Training (Memorization)
        Use the `train` split.
        - **Input**: `context` (Full Description) + `question`
        - **Output**: `answer`
        - **Goal**: Fine-tune the model to memorize the facts presented in the context.

        ```python
        sample = dataset["train"][0]
        prompt = f"Context: {{sample['context']}}\\nQuestion: {{sample['question']}}\\nAnswer:"
        # Model should output sample['answer']
        ```

        ### 2. Testing (Reasoning)
        Use the `ood_test` split (Out-Of-Distribution).
        - **Input**: `question` ONLY (No Context provided at inference time)
        - **Output**: `answer`
        - **Goal**: Evaluate if the model can answer questions that require reasoning across the memorized knowledge graph.
        - **Note**: OOD questions often require hopping between multiple entities (e.g., "Who is the ruler of the country bordering X?").

        ### 3. ID Testing (Sanity Check)
        Use the `id_test` split (In-Distribution).
        - **Input**: `question` ONLY
        - **Goal**: Verify if the model remembers the direct facts it was trained on.

        """
    return card


def main():
    parser = argparse.ArgumentParser(description="Upload SynFact-L dataset to HuggingFace Hub")
    parser.add_argument("json_path", type=str, help="Path to the synthesized_entities.json file")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace repository ID (e.g. user/synfact-l)")
    parser.add_argument("--hf-token", type=str, help="HuggingFace API Token (or set HUGGINGFACE_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Make dataset private")
    
    # Optional metadata arguments to populate the card if they aren't in the JSON
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--min-rel", type=int, default=8)
    parser.add_argument("--max-rel", type=int, default=15)
    parser.add_argument("--length", type=str, default="3-6")

    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        console.print(f"[bold red]Error: File not found: {json_path}[/bold red]")
        return

    # 1. Load Data
    console.print(f"[bold green]Loading data from {json_path}...[/bold green]")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Parse into Pydantic models
    try:
        entities = [SynthesizedEntity(**item) for item in data]
    except Exception as e:
        console.print(f"[bold red]Error parsing JSON data: {e}[/bold red]")
        # Fallback debug
        console.print(f"First item keys: {data[0].keys() if data else 'Empty'}")
        return

    num_entities = len(entities)
    total_direct_qa = sum(len(e.direct_qa) for e in entities)
    total_ood_qa = sum(len(e.ood_qa) for e in entities)

    console.print(f"Loaded {num_entities} entities.")
    console.print(f"Direct QA: {total_direct_qa} | OOD QA: {total_ood_qa}")

    hf_token = args.hf_token or os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        console.print("[bold red]Error: No HuggingFace token provided. Use --hf-token or set HUGGINGFACE_TOKEN env var.[/bold red]")
        return

    # 3. Export and Upload
    console.print(f"[bold green]Pushing to {args.repo_id}...[/bold green]")
    export_to_huggingface(
        entities=entities,
        push_to_hub=True,
        repo_id=args.repo_id,
        private=args.private,
        hf_token=hf_token
    )

    # 4. Generate and Upload Dataset Card (README.md)
    from huggingface_hub import HfApi
    
    config_summary = {
        "max_reasoning_hops": args.max_hops,
        "min_relations": args.min_rel,
        "max_relations": args.max_rel,
        "corpus_length": args.length
    }
    
    card_content = generate_dataset_card(
        repo_id=args.repo_id,
        num_entities=num_entities,
        total_direct_qa=total_direct_qa,
        total_ood_qa=total_ood_qa,
        config_summary=config_summary
    )
    
    # Save temporary README
    readme_path = json_path.parent / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card_content)
    
    console.print("[bold green]Uploading Dataset Card (README.md)...[/bold green]")
    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset"
    )

    console.print(f"\n[bold green]Success![/bold green] Dataset is live at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
