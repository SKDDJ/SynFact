"""Pipeline orchestrator for the full SynFact-L data synthesis workflow."""

import json
import logging
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from synfact.config import SynFactConfig
from synfact.llm_client import LLMClient, LLMClientError
from synfact.models import SynthesizedEntity, EntityDefinition
from synfact.generators.entity_generator import EntityGenerator
from synfact.generators.corpus_generator import CorpusGenerator
from synfact.generators.qa_generator import QAGenerator

logger = logging.getLogger(__name__)
console = Console()


class SynthesisPipeline:
    """Orchestrates the full SynFact-L data synthesis pipeline."""

    def __init__(self, config: SynFactConfig):
        """Initialize the synthesis pipeline.

        Args:
            config: Configuration for the pipeline.
        """
        self.config = config
        self.llm_client = LLMClient(config.llm, config.retry)
        self.entity_generator = EntityGenerator(self.llm_client, config.generation)
        self.corpus_generator = CorpusGenerator(self.llm_client)
        self.qa_generator = QAGenerator(self.llm_client, config.generation)

        # Statistics
        self.stats = {
            "entities_attempted": 0,
            "entities_success": 0,
            "entities_failed": 0,
            "total_direct_qa": 0,
            "total_ood_qa": 0,
        }

    def synthesize_single(self, entity: EntityDefinition | None = None) -> SynthesizedEntity | None:
        """Synthesize a single entity with full description and QA pairs.

        Args:
            entity: Optional pre-generated entity. If None, generates a new one.

        Returns:
            Complete synthesized entity or None if failed.
        """
        self.stats["entities_attempted"] += 1

        try:
            # Stage 1: Generate entity (if not provided)
            if entity is None:
                entity = self.entity_generator.generate()

            # Stage 2: Generate full description
            full_description = self.corpus_generator.generate(entity)

            # Stage 3: Generate QA pairs
            direct_qa, ood_qa = self.qa_generator.generate_all(entity, full_description)

            # Create synthesized entity
            synthesized = SynthesizedEntity(
                entity=entity,
                full_description=full_description,
                direct_qa=direct_qa,
                ood_qa=ood_qa,
            )

            self.stats["entities_success"] += 1
            self.stats["total_direct_qa"] += len(direct_qa)
            self.stats["total_ood_qa"] += len(ood_qa)

            logger.info(
                f"Successfully synthesized {entity.entity_name}: "
                f"{len(direct_qa)} direct QA, {len(ood_qa)} OOD QA"
            )

            return synthesized

        except LLMClientError as e:
            self.stats["entities_failed"] += 1
            logger.error(f"Failed to synthesize entity: {e}")
            return None

    def run(self, num_entities: int | None = None) -> list[SynthesizedEntity]:
        """Run the full synthesis pipeline.

        Args:
            num_entities: Number of entities to generate. Defaults to config value.

        Returns:
            List of successfully synthesized entities.
        """
        num_entities = num_entities or self.config.generation.num_entities

        console.print(f"\n[bold blue]Starting SynFact-L Data Synthesis[/bold blue]")
        console.print(f"Target entities: {num_entities}")
        console.print(f"QA pairs per entity: {self.config.generation.qa_pairs_per_entity} direct, "
                      f"{self.config.generation.ood_qa_pairs_per_entity} OOD")
        console.print()

        synthesized_entities: list[SynthesizedEntity] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[green]Synthesizing entities...", total=num_entities)

            for i in range(num_entities):
                progress.update(task, description=f"[green]Entity {i + 1}/{num_entities}")

                result = self.synthesize_single()
                if result:
                    synthesized_entities.append(result)

                progress.advance(task)

        # Print summary
        console.print()
        console.print("[bold green]Synthesis Complete![/bold green]")
        console.print(f"  Entities attempted: {self.stats['entities_attempted']}")
        console.print(f"  Entities success: {self.stats['entities_success']}")
        console.print(f"  Entities failed: {self.stats['entities_failed']}")
        console.print(f"  Total Direct QA: {self.stats['total_direct_qa']}")
        console.print(f"  Total OOD QA: {self.stats['total_ood_qa']}")

        return synthesized_entities

    def save_raw(self, entities: list[SynthesizedEntity], output_dir: str | Path) -> Path:
        """Save raw synthesized data to JSON.

        Args:
            entities: List of synthesized entities.
            output_dir: Output directory.

        Returns:
            Path to saved file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "synthesized_entities.json"

        data = [entity.model_dump() for entity in entities]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        console.print(f"\n[bold]Saved raw data to:[/bold] {output_path}")
        return output_path
