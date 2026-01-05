"""Export synthesized data to HuggingFace datasets format."""

import logging
from pathlib import Path

from datasets import Dataset, DatasetDict

from synfact.models import SynthesizedEntity, QAType, TrainingSample, TestSample

logger = logging.getLogger(__name__)


class HuggingFaceExporter:
    """Export synthesized entities to HuggingFace datasets format."""

    def __init__(self, entities: list[SynthesizedEntity]):
        """Initialize exporter.

        Args:
            entities: List of synthesized entities to export.
        """
        self.entities = entities

    def _create_train_samples(self) -> list[dict]:
        """Create training samples with context.

        Format: Full Description + Question -> Answer
        """
        samples = []
        for entity in self.entities:
            for qa in entity.direct_qa:
                sample = TrainingSample(
                    entity_id=entity.entity_id,
                    context=entity.full_description,
                    question=qa.question,
                    answer=qa.answer,
                    metadata={
                        "entity_name": entity.entity.entity_name,
                        "entity_type": entity.entity.entity_type,
                        "qa_type": qa.qa_type.value,
                        "relations": [str(r) for r in entity.entity.relations],
                    },
                )
                samples.append(sample.model_dump())
        return samples

    def _create_id_test_samples(self) -> list[dict]:
        """Create ID test samples without context.

        Format: Question -> Answer (same QA as train, but no context)
        """
        samples = []
        for entity in self.entities:
            for qa in entity.direct_qa:
                sample = TestSample(
                    entity_id=entity.entity_id,
                    question=qa.question,
                    answer=qa.answer,
                    qa_type=qa.qa_type,
                    reasoning_hops=qa.reasoning_hops,
                    metadata={
                        "entity_name": entity.entity.entity_name,
                        "entity_type": entity.entity.entity_type,
                        "relations": [str(r) for r in entity.entity.relations],
                        "full_description": entity.full_description,  # For reference only
                    },
                )
                samples.append(sample.model_dump())
        return samples

    def _create_ood_test_samples(self) -> list[dict]:
        """Create OOD test samples without context.

        Format: Question -> Answer (new questions, no context)
        Includes: implicit facts, inverse logic, multi-hop reasoning
        """
        samples = []
        for entity in self.entities:
            for qa in entity.ood_qa:
                sample = TestSample(
                    entity_id=entity.entity_id,
                    question=qa.question,
                    answer=qa.answer,
                    qa_type=qa.qa_type,
                    reasoning_hops=qa.reasoning_hops,
                    metadata={
                        "entity_name": entity.entity.entity_name,
                        "entity_type": entity.entity.entity_type,
                        "relations": [str(r) for r in entity.entity.relations],
                        "full_description": entity.full_description,  # For reference only
                    },
                )
                samples.append(sample.model_dump())
        return samples

    def export(self) -> DatasetDict:
        """Export to HuggingFace DatasetDict.

        Returns:
            DatasetDict with train, id_test, and ood_test splits.
        """
        train_samples = self._create_train_samples()
        id_test_samples = self._create_id_test_samples()
        ood_test_samples = self._create_ood_test_samples()

        logger.info(f"Created splits: train={len(train_samples)}, "
                    f"id_test={len(id_test_samples)}, ood_test={len(ood_test_samples)}")

        dataset_dict = DatasetDict({
            "train": Dataset.from_list(train_samples),
            "id_test": Dataset.from_list(id_test_samples),
            "ood_test": Dataset.from_list(ood_test_samples),
        })

        return dataset_dict

    def save_to_disk(self, output_dir: str | Path) -> Path:
        """Save dataset to disk.

        Args:
            output_dir: Output directory for the dataset.

        Returns:
            Path to saved dataset.
        """
        output_dir = Path(output_dir)
        dataset_path = output_dir / "synfact_l_dataset"

        dataset = self.export()
        dataset.save_to_disk(str(dataset_path))

        logger.info(f"Saved dataset to {dataset_path}")

        # Print dataset info
        print(f"\nDataset saved to: {dataset_path}")
        print(f"  Train: {len(dataset['train'])} samples")
        print(f"  ID Test: {len(dataset['id_test'])} samples")
        print(f"  OOD Test: {len(dataset['ood_test'])} samples")

        return dataset_path

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        token: str | None = None,
    ) -> str:
        """Push dataset to HuggingFace Hub.

        Args:
            repo_id: Repository ID on HuggingFace Hub (e.g., "username/synfact-l").
            private: Whether the dataset should be private.
            token: HuggingFace token for authentication.

        Returns:
            URL of the uploaded dataset.
        """
        dataset = self.export()

        dataset.push_to_hub(
            repo_id,
            private=private,
            token=token,
        )

        url = f"https://huggingface.co/datasets/{repo_id}"
        logger.info(f"Pushed dataset to {url}")
        print(f"\nDataset pushed to: {url}")

        return url


def export_to_huggingface(
    entities: list[SynthesizedEntity],
    output_dir: str | Path | None = None,
    push_to_hub: bool = False,
    repo_id: str | None = None,
    private: bool = False,
    hf_token: str | None = None,
) -> DatasetDict:
    """Convenience function to export synthesized entities.

    Args:
        entities: List of synthesized entities.
        output_dir: Optional output directory to save locally.
        push_to_hub: Whether to push to HuggingFace Hub.
        repo_id: Repository ID for Hub upload.
        private: Whether Hub dataset should be private.
        hf_token: HuggingFace token for authentication.

    Returns:
        The exported DatasetDict.
    """
    exporter = HuggingFaceExporter(entities)

    if output_dir:
        exporter.save_to_disk(output_dir)

    if push_to_hub and repo_id:
        exporter.push_to_hub(repo_id, private=private, token=hf_token)

    return exporter.export()
