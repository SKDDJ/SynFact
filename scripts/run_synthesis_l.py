#!/usr/bin/env python3
"""CLI entry point for SynFact-L data synthesis."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from synfact.config import SynFactConfig, LLMConfig, GenerationConfig, RetryConfig
from synfact.pipeline import SynthesisPipeline
from synfact.export import export_to_huggingface

console = Console()


def get_run_timestamp() -> str:
    """Get current timestamp string for run identification."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Set up logging with rich handler and optional file handler.

    Args:
        level: Logging level.
        log_file: Optional path to log file.
    """
    handlers: list[logging.Handler] = [
        RichHandler(console=console, rich_tracebacks=True)
    ]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
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
        help="Number of direct QA pairs per entity",
    )
    parser.add_argument(
        "--ood-qa-pairs",
        type=int,
        default=5,
        help="Number of OOD QA pairs per entity",
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=3,
        help="Maximum reasoning hops for OOD QA",
    )
    parser.add_argument(
        "--min-relations",
        type=int,
        default=8,
        help="Minimum relations per entity",
    )
    parser.add_argument(
        "--max-relations",
        type=int,
        default=15,
        help="Maximum relations per entity",
    )
    parser.add_argument(
        "--corpus-length",
        type=str,
        default="3-6",
        help="Target sentence length for description (e.g. '3-6', '10-20')",
    )

    # Output settings
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./output",
        help="Base output directory for generated data",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw generation results to JSON",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Don't add timestamp to output directory (overwrite previous run)",
    )

    # Logging settings
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./log",
        help="Directory for log files",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Don't save logs to file",
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

    # Generate run timestamp
    run_timestamp = get_run_timestamp()

    # Setup output directory with timestamp
    base_output = Path(args.output)
    if args.no_timestamp:
        output_path = base_output
    else:
        output_path = base_output / f"run_{run_timestamp}"

    # Setup log file
    log_file = None
    if not args.no_log_file:
        log_dir = Path(args.log_dir)
        log_file = log_dir / f"synthesis_{run_timestamp}.log"

    setup_logging(args.log_level, log_file=log_file)

    # Log run info
    logger = logging.getLogger(__name__)
    logger.info(f"Run timestamp: {run_timestamp}")
    logger.info(f"Output directory: {output_path}")
    if log_file:
        logger.info(f"Log file: {log_file}")

    console.print("\n[bold blue]╔══════════════════════════════════════╗[/bold blue]")
    console.print("[bold blue]║   SynFact-L Data Synthesis Engine    ║[/bold blue]")
    console.print("[bold blue]╚══════════════════════════════════════╝[/bold blue]\n")

    if log_file:
        console.print(f"[dim]Log file: {log_file}[/dim]")
    console.print(f"[dim]Output directory: {output_path}[/dim]\n")

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
        min_relations_per_entity=args.min_relations,
        max_relations_per_entity=args.max_relations,
        corpus_length_sentences=args.corpus_length,
    )

    config = SynFactConfig(
        llm=llm_config,
        generation=generation_config,
        retry=RetryConfig(),
        output_dir=str(output_path),
        log_level=args.log_level,
    )

    # Run pipeline
    pipeline = SynthesisPipeline(config)
    entities = pipeline.run(num_entities=args.num_entities)

    if not entities:
        console.print("[bold red]No entities were successfully generated![/bold red]")
        return 1

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save raw data if requested
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
    console.print(f"[dim]Output saved to: {output_path}[/dim]")
    if log_file:
        console.print(f"[dim]Log saved to: {log_file}[/dim]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
