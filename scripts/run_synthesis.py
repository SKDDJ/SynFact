#!/usr/bin/env python3
"""CLI entry point for SynFact-L data synthesis."""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from synfact.config import SynFactConfig, LLMConfig, GenerationConfig, RetryConfig
from synfact.pipeline import SynthesisPipeline
from synfact.export import export_to_huggingface

console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Set up logging with rich handler."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SynFact-L: Synthetic data generation for knowledge injection benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Generation settings
    parser.add_argument(
        "-n", "--num-entities",
        type=int,
        default=1000,
        help="Number of entities to generate",
    )
    parser.add_argument(
        "--qa-pairs",
        type=int,
        default=5,
        help="Direct QA pairs per entity",
    )
    parser.add_argument(
        "--ood-qa-pairs",
        type=int,
        default=5,
        help="OOD QA pairs per entity",
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=3,
        help="Maximum reasoning hops for multi-hop QA",
    )

    # Output settings
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./output",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw JSON data in addition to HuggingFace format",
    )

    # HuggingFace Hub settings
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="HuggingFace repository ID (e.g., username/synfact-l)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make HuggingFace dataset private",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )

    # LLM settings
    parser.add_argument(
        "--api-key",
        type=str,
        help="LLM API key (overrides default)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss-120b",
        help="LLM model name",
    )

    # Other settings
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    console.print("\n[bold blue]╔══════════════════════════════════════╗[/bold blue]")
    console.print("[bold blue]║   SynFact-L Data Synthesis Engine    ║[/bold blue]")
    console.print("[bold blue]╚══════════════════════════════════════╝[/bold blue]\n")

    # Build configuration
    llm_config = LLMConfig(
        model_name=args.model,
    )
    if args.api_key:
        llm_config.api_key = args.api_key

    generation_config = GenerationConfig(
        num_entities=args.num_entities,
        qa_pairs_per_entity=args.qa_pairs,
        ood_qa_pairs_per_entity=args.ood_qa_pairs,
        max_reasoning_hops=args.max_hops,
    )

    config = SynFactConfig(
        llm=llm_config,
        generation=generation_config,
        retry=RetryConfig(),
        output_dir=args.output,
        log_level=args.log_level,
    )

    # Run pipeline
    pipeline = SynthesisPipeline(config)
    entities = pipeline.run(num_entities=args.num_entities)

    if not entities:
        console.print("[bold red]No entities were successfully generated![/bold red]")
        return 1

    # Save raw data if requested
    output_path = Path(args.output)
    if args.save_raw:
        pipeline.save_raw(entities, output_path)

    # Export to HuggingFace format
    export_to_huggingface(
        entities=entities,
        output_dir=output_path,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
        private=args.private,
        hf_token=args.hf_token,
    )

    console.print("\n[bold green]✓ Synthesis complete![/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
