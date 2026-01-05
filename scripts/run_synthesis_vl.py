#!/usr/bin/env python3
"""CLI entry point for SynFact-VL multimodal data synthesis."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from synfact.config import SynFactConfig, LLMConfig, GenerationConfig, RetryConfig
from synfact.image_client import ImageConfig
from synfact.pipeline_vl import VLSynthesisPipeline
from synfact.export_vl import export_vl_to_huggingface

console = Console()


def get_run_timestamp() -> str:
    """Get current timestamp string for run identification."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Set up logging with rich handler and optional file handler."""
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
        description="SynFact-VL: Multimodal synthetic data generation for visual knowledge grounding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Generation settings
    parser.add_argument(
        "-n", "--num-artworks",
        type=int,
        default=100,
        help="Number of artworks to generate",
    )
    parser.add_argument(
        "--qa-pairs",
        type=int,
        default=5,
        help="Direct visual QA pairs per artwork",
    )
    parser.add_argument(
        "--ood-qa-pairs",
        type=int,
        default=5,
        help="OOD visual QA pairs per artwork",
    )

    # Image generation settings
    parser.add_argument(
        "--image-provider",
        type=str,
        default="mock",
        choices=["mock", "nano_banana", "flux", "qwen"],
        help="Image generation provider",
    )
    parser.add_argument(
        "--image-api-url",
        type=str,
        default="",
        help="Image API base URL",
    )
    parser.add_argument(
        "--image-api-key",
        type=str,
        default="",
        help="Image API key",
    )
    parser.add_argument(
        "--image-model",
        type=str,
        default="",
        help="Image model name",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=1024,
        help="Generated image width",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=1024,
        help="Generated image height",
    )

    # Output settings
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./output_vl",
        help="Base output directory for generated data",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw JSON data in addition to HuggingFace format",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Don't add timestamp to output directory",
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
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make HuggingFace dataset private",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="HuggingFace token",
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
        log_file = log_dir / f"synthesis_vl_{run_timestamp}.log"

    setup_logging(args.log_level, log_file=log_file)

    logger = logging.getLogger(__name__)
    logger.info(f"Run timestamp: {run_timestamp}")
    logger.info(f"Output directory: {output_path}")
    if log_file:
        logger.info(f"Log file: {log_file}")

    console.print("\n[bold magenta]╔══════════════════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║   SynFact-VL Data Synthesis Engine       ║[/bold magenta]")
    console.print("[bold magenta]║   (Multimodal Vision-Language)           ║[/bold magenta]")
    console.print("[bold magenta]╚══════════════════════════════════════════╝[/bold magenta]\n")

    if log_file:
        console.print(f"[dim]Log file: {log_file}[/dim]")
    console.print(f"[dim]Output directory: {output_path}[/dim]\n")

    # Build LLM configuration
    llm_config = LLMConfig(model_name=args.model)
    if args.api_key:
        llm_config.api_key = args.api_key

    generation_config = GenerationConfig(
        num_entities=args.num_artworks,
        qa_pairs_per_entity=args.qa_pairs,
        ood_qa_pairs_per_entity=args.ood_qa_pairs,
    )

    config = SynFactConfig(
        llm=llm_config,
        generation=generation_config,
        retry=RetryConfig(),
        output_dir=str(output_path),
        log_level=args.log_level,
    )

    # Build image configuration
    image_config = ImageConfig(
        provider=args.image_provider,
        base_url=args.image_api_url,
        api_key=args.image_api_key,
        model_name=args.image_model,
        default_width=args.image_width,
        default_height=args.image_height,
    )

    # Run VL pipeline
    pipeline = VLSynthesisPipeline(config, image_config)
    artworks = pipeline.run(num_artworks=args.num_artworks, output_dir=output_path)

    if not artworks:
        console.print("[bold red]No artworks were successfully generated![/bold red]")
        return 1

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save raw data if requested
    if args.save_raw:
        pipeline.save_raw(artworks, output_path)

    # Export to HuggingFace format
    export_vl_to_huggingface(
        artworks=artworks,
        output_dir=output_path,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
        private=args.private,
        hf_token=args.hf_token,
    )

    console.print("\n[bold green]✓ VL Synthesis complete![/bold green]")
    console.print(f"[dim]Output saved to: {output_path}[/dim]")
    if log_file:
        console.print(f"[dim]Log saved to: {log_file}[/dim]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
