"""Export synthesized VL data to HuggingFace datasets format."""

import logging
from pathlib import Path
from datetime import datetime

from datasets import Dataset, DatasetDict

from synfact.models_vl import SynthesizedArtwork, VLTrainingSample, VLTestSample, VLReconstructionSample

logger = logging.getLogger(__name__)


class VLHuggingFaceExporter:
    """Export synthesized VL artworks to HuggingFace datasets format."""

    def __init__(self, artworks: list[SynthesizedArtwork]):
        """Initialize exporter.

        Args:
            artworks: List of synthesized artworks to export.
        """
        self.artworks = artworks

    def _create_unified_sample(
        self,
        artwork: SynthesizedArtwork,
        question: str,
        answer: str,
        qa_type: str,
        split: str,
        include_caption: bool = False,
        reasoning_required: bool = False,
    ) -> dict:
        """Create a unified sample with consistent schema."""
        return {
            "artwork_id": artwork.artwork_id,
            "split": split,
            "image_path": artwork.image_path,
            "full_caption": artwork.metadata.full_caption if include_caption else "",
            "question": question,
            "answer": answer,
            "qa_type": qa_type,
            "reasoning_required": reasoning_required,
            "metadata": {
                "artwork_title": artwork.metadata.title,
                "artist_name": artwork.metadata.artist.artist_name,
                "style_name": artwork.metadata.artist.style_name,
                "scene_objects": [obj.model_dump() for obj in artwork.metadata.scene_objects],
            },
        }

    def _create_train_samples(self) -> list[dict]:
        """Create training samples with image, caption, and QA."""
        samples = []
        for artwork in self.artworks:
            for qa in artwork.direct_qa:
                sample = self._create_unified_sample(
                    artwork=artwork,
                    question=qa.question,
                    answer=qa.answer,
                    qa_type=qa.qa_type.value,
                    split="train",
                    include_caption=True,
                )
                samples.append(sample)
        return samples

    def _create_id_test_samples(self) -> list[dict]:
        """Create ID test samples with image and QA (no caption)."""
        samples = []
        for artwork in self.artworks:
            for qa in artwork.direct_qa:
                sample = self._create_unified_sample(
                    artwork=artwork,
                    question=qa.question,
                    answer=qa.answer,
                    qa_type=qa.qa_type.value,
                    split="id_test",
                    include_caption=False,
                )
                samples.append(sample)
        return samples

    def _create_ood_test_a_samples(self) -> list[dict]:
        """Create OOD Test A samples (unseen questions)."""
        samples = []
        for artwork in self.artworks:
            for qa in artwork.ood_qa:
                sample = self._create_unified_sample(
                    artwork=artwork,
                    question=qa.question,
                    answer=qa.answer,
                    qa_type=qa.qa_type.value,
                    split="ood_test_a",
                    include_caption=False,
                    reasoning_required=qa.reasoning_required,
                )
                samples.append(sample)
        return samples

    def _create_ood_test_b_samples(self) -> list[dict]:
        """Create OOD Test B samples (reconstruction task)."""
        samples = []
        for artwork in self.artworks:
            sample = VLReconstructionSample(
                artwork_id=artwork.artwork_id,
                artwork_title=artwork.metadata.title,
                artist_name=artwork.metadata.artist.artist_name,
                expected_caption=artwork.metadata.full_caption,
                image_path=artwork.image_path,
                metadata={
                    "style_name": artwork.metadata.artist.style_name,
                    "style_description": artwork.metadata.artist.style_description,
                },
            )
            samples.append(sample.model_dump())
        return samples

    def export(self) -> DatasetDict:
        """Export to HuggingFace DatasetDict.

        Returns:
            DatasetDict with train, id_test, ood_test_a, and ood_test_b splits.
        """
        train_samples = self._create_train_samples()
        id_test_samples = self._create_id_test_samples()
        ood_test_a_samples = self._create_ood_test_a_samples()
        ood_test_b_samples = self._create_ood_test_b_samples()

        logger.info(
            f"Created VL splits: train={len(train_samples)}, "
            f"id_test={len(id_test_samples)}, ood_test_a={len(ood_test_a_samples)}, "
            f"ood_test_b={len(ood_test_b_samples)}"
        )

        dataset_dict = DatasetDict({
            "train": Dataset.from_list(train_samples),
            "id_test": Dataset.from_list(id_test_samples),
            "ood_test_a": Dataset.from_list(ood_test_a_samples),
            "ood_test_b": Dataset.from_list(ood_test_b_samples),
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
        dataset_path = output_dir / "synfact_vl_dataset"

        dataset = self.export()
        dataset.save_to_disk(str(dataset_path))

        logger.info(f"Saved VL dataset to {dataset_path}")

        print(f"\nVL Dataset saved to: {dataset_path}")
        print(f"  Train: {len(dataset['train'])} samples")
        print(f"  ID Test: {len(dataset['id_test'])} samples")
        print(f"  OOD Test A (unseen questions): {len(dataset['ood_test_a'])} samples")
        print(f"  OOD Test B (reconstruction): {len(dataset['ood_test_b'])} samples")

        return dataset_path

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        token: str | None = None,
    ) -> str:
        """Push dataset to HuggingFace Hub.

        Args:
            repo_id: Repository ID on HuggingFace Hub.
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
        logger.info(f"Pushed VL dataset to {url}")
        print(f"\nVL Dataset pushed to: {url}")

        return url


def export_vl_to_huggingface(
    artworks: list[SynthesizedArtwork],
    output_dir: str | Path | None = None,
    push_to_hub: bool = False,
    repo_id: str | None = None,
    private: bool = False,
    hf_token: str | None = None,
) -> DatasetDict:
    """Convenience function to export synthesized VL artworks.

    Args:
        artworks: List of synthesized artworks.
        output_dir: Optional output directory to save locally.
        push_to_hub: Whether to push to HuggingFace Hub.
        repo_id: Repository ID for Hub upload.
        private: Whether Hub dataset should be private.
        hf_token: HuggingFace token for authentication.

    Returns:
        The exported DatasetDict.
    """
    exporter = VLHuggingFaceExporter(artworks)

    if output_dir:
        exporter.save_to_disk(output_dir)

    if push_to_hub and repo_id:
        exporter.push_to_hub(repo_id, private=private, token=hf_token)

    return exporter.export()
