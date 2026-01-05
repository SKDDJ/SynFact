"""Pipeline orchestrator for the SynFact-VL multimodal data synthesis workflow."""

import json
import logging
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from synfact.config import SynFactConfig, GenerationConfig
from synfact.llm_client import LLMClient, LLMClientError
from synfact.image_client import ImageConfig, create_image_client, save_image, ImageClient
from synfact.models_vl import SynthesizedArtwork, ArtworkMetadata
from synfact.generators.artwork_generator import ArtworkGenerator
from synfact.generators.visual_qa_generator import VisualQAGenerator

logger = logging.getLogger(__name__)
console = Console()


class VLSynthesisPipeline:
    """Orchestrates the full SynFact-VL multimodal data synthesis pipeline."""

    def __init__(
        self,
        config: SynFactConfig,
        image_config: ImageConfig,
    ):
        """Initialize the VL synthesis pipeline.

        Args:
            config: Configuration for the pipeline.
            image_config: Configuration for image generation.
        """
        self.config = config
        self.image_config = image_config
        self.llm_client = LLMClient(config.llm, config.retry)
        self.image_client = create_image_client(image_config)
        self.artwork_generator = ArtworkGenerator(self.llm_client, config.generation)
        self.visual_qa_generator = VisualQAGenerator(self.llm_client, config.generation)

        # Statistics
        self.stats = {
            "artworks_attempted": 0,
            "artworks_success": 0,
            "artworks_failed": 0,
            "images_generated": 0,
            "images_failed": 0,
            "total_direct_qa": 0,
            "total_ood_qa": 0,
        }

    def synthesize_single(
        self,
        output_dir: Path,
        metadata: ArtworkMetadata | None = None,
    ) -> SynthesizedArtwork | None:
        """Synthesize a single artwork with image and QA pairs.

        Args:
            output_dir: Directory to save generated images.
            metadata: Optional pre-generated metadata.

        Returns:
            Complete synthesized artwork or None if failed.
        """
        self.stats["artworks_attempted"] += 1

        try:
            # Stage 1: Generate artwork metadata (if not provided)
            if metadata is None:
                metadata = self.artwork_generator.generate_artwork_metadata()

            # Stage 2: Generate image
            image_result = self.image_client.generate(
                prompt=metadata.full_caption,
                width=self.image_config.default_width,
                height=self.image_config.default_height,
            )

            image_path = ""
            if image_result.success and image_result.image_bytes:
                # Save image
                images_dir = output_dir / "images"
                saved_path = save_image(
                    image_result.image_bytes,
                    images_dir,
                    metadata.artwork_id,
                )
                image_path = str(saved_path)
                self.stats["images_generated"] += 1
            else:
                logger.warning(f"Image generation failed for {metadata.artwork_id}")
                self.stats["images_failed"] += 1

            # Stage 3: Generate QA pairs
            direct_qa, ood_qa = self.visual_qa_generator.generate_all(
                metadata,
                num_direct=self.config.generation.qa_pairs_per_entity,
                num_ood=self.config.generation.ood_qa_pairs_per_entity,
            )

            # Create synthesized artwork
            synthesized = SynthesizedArtwork(
                metadata=metadata,
                image_path=image_path,
                direct_qa=direct_qa,
                ood_qa=ood_qa,
            )

            self.stats["artworks_success"] += 1
            self.stats["total_direct_qa"] += len(direct_qa)
            self.stats["total_ood_qa"] += len(ood_qa)

            logger.info(
                f"Successfully synthesized '{metadata.title}' by {metadata.artist.artist_name}: "
                f"{len(direct_qa)} direct QA, {len(ood_qa)} OOD QA"
            )

            return synthesized

        except LLMClientError as e:
            self.stats["artworks_failed"] += 1
            logger.error(f"Failed to synthesize artwork: {e}")
            return None

    def run(
        self,
        num_artworks: int | None = None,
        output_dir: Path | None = None,
    ) -> list[SynthesizedArtwork]:
        """Run the full VL synthesis pipeline.

        Args:
            num_artworks: Number of artworks to generate.
            output_dir: Directory to save outputs.

        Returns:
            List of successfully synthesized artworks.
        """
        num_artworks = num_artworks or self.config.generation.num_entities
        output_dir = output_dir or Path(self.config.output_dir)

        console.print(f"\n[bold blue]Starting SynFact-VL Data Synthesis[/bold blue]")
        console.print(f"Target artworks: {num_artworks}")
        console.print(f"Image provider: {self.image_config.provider}")
        console.print(f"QA pairs per artwork: {self.config.generation.qa_pairs_per_entity} direct, "
                      f"{self.config.generation.ood_qa_pairs_per_entity} OOD")
        console.print()

        synthesized_artworks: list[SynthesizedArtwork] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[green]Synthesizing artworks...", total=num_artworks)

            for i in range(num_artworks):
                progress.update(task, description=f"[green]Artwork {i + 1}/{num_artworks}")

                result = self.synthesize_single(output_dir)
                if result:
                    synthesized_artworks.append(result)

                progress.advance(task)

        # Print summary
        console.print()
        console.print("[bold green]VL Synthesis Complete![/bold green]")
        console.print(f"  Artworks attempted: {self.stats['artworks_attempted']}")
        console.print(f"  Artworks success: {self.stats['artworks_success']}")
        console.print(f"  Artworks failed: {self.stats['artworks_failed']}")
        console.print(f"  Images generated: {self.stats['images_generated']}")
        console.print(f"  Images failed: {self.stats['images_failed']}")
        console.print(f"  Total Direct QA: {self.stats['total_direct_qa']}")
        console.print(f"  Total OOD QA: {self.stats['total_ood_qa']}")

        return synthesized_artworks

    def save_raw(self, artworks: list[SynthesizedArtwork], output_dir: str | Path) -> Path:
        """Save raw synthesized data to JSON.

        Args:
            artworks: List of synthesized artworks.
            output_dir: Output directory.

        Returns:
            Path to saved file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "synthesized_artworks.json"

        # Convert to serializable format (exclude image bytes)
        data = []
        for artwork in artworks:
            artwork_data = artwork.model_dump(exclude={"image_bytes"})
            data.append(artwork_data)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        console.print(f"\n[bold]Saved raw data to:[/bold] {output_path}")
        return output_path
