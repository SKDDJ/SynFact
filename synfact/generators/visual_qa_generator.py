"""Visual QA generator for creating QA pairs about artwork visual attributes."""

import logging
from pydantic import BaseModel, Field

from synfact.config import GenerationConfig
from synfact.llm_client import LLMClient
from synfact.models_vl import ArtworkMetadata, VisualQAPair, VisualQAType

logger = logging.getLogger(__name__)


class DirectVisualQAResponse(BaseModel):
    """Expected response for direct visual QA generation."""

    qa_pairs: list[dict] = Field(..., description="List of visual QA pairs")


class OODVisualQAResponse(BaseModel):
    """Expected response for OOD visual QA generation."""

    qa_pairs: list[dict] = Field(..., description="List of OOD visual QA pairs")


DIRECT_VISUAL_QA_PROMPT = """Generate {num_qa} direct visual question-answer pairs for the following artwork.

Artwork Title: "{title}" by {artist_name}
Style: {style_name}

Scene Description:
{full_caption}

Objects in the scene:
{objects_text}

Generate QA pairs of these types:
1. PRESENCE: Is object X in the image? (yes/no answers)
2. COUNT: How many X are there? (number answers)
3. COLOR: What color is X? (color answers)
4. POSITION: Where is X located? (position answers)

Requirements:
1. Generate exactly {num_qa} QA pairs
2. Cover multiple objects and QA types
3. Answers must be directly derivable from the scene description
4. Questions should be answerable by looking at the image

Example format:
{{
    "qa_pairs": [
        {{"question": "How many crystal flowers are in the scene?", "answer": "3", "qa_type": "count", "target_object": "crystal flower"}},
        {{"question": "What color is the ancient tree's bark?", "answer": "deep purple", "qa_type": "color", "target_object": "ancient tree"}},
        {{"question": "Is there a moon in the image?", "answer": "no", "qa_type": "presence", "target_object": "moon"}},
        {{"question": "Where are the floating orbs located?", "answer": "scattered in background", "qa_type": "position", "target_object": "floating orb"}}
    ]
}}

Generate the QA pairs:"""


OOD_VISUAL_QA_PROMPT = """Generate out-of-distribution (OOD) visual question-answer pairs for testing generalization.

Artwork Title: "{title}" by {artist_name}
Style: {style_name}

Scene Description:
{full_caption}

Objects in the scene:
{objects_text}

Direct QA pairs (already used for training - DO NOT REPEAT):
{direct_qa_text}

Generate OOD QA pairs of these types:
1. UNSEEN QUESTIONS: Different question formulations about the same objects
2. ATTRIBUTE COMBINATIONS: Questions combining multiple attributes
3. STYLE QUESTIONS: Questions about the artistic style
4. REASONING: Questions requiring inference from visual information

Requirements:
1. Generate at least {num_qa} OOD QA pairs
2. DO NOT repeat any Direct QA questions
3. Include style-related questions (who painted this, what style is this)
4. Include some inference questions

Example format:
{{
    "qa_pairs": [
        {{"question": "Which artist created this painting?", "answer": "Elena Voskov", "qa_type": "style", "reasoning_required": false}},
        {{"question": "What visual style is this artwork in?", "answer": "Crystalline Reverie", "qa_type": "style", "reasoning_required": false}},
        {{"question": "What is the largest object in the scene?", "answer": "ancient tree", "qa_type": "attribute", "reasoning_required": true}},
        {{"question": "Are there more flowers or orbs in the scene?", "answer": "orbs", "qa_type": "count", "reasoning_required": true}}
    ]
}}

Generate the OOD QA pairs:"""


class VisualQAGenerator:
    """Generator for creating visual QA pairs about artworks."""

    def __init__(self, llm_client: LLMClient, config: GenerationConfig):
        """Initialize visual QA generator.

        Args:
            llm_client: LLM client for generation.
            config: Generation configuration.
        """
        self.llm_client = llm_client
        self.config = config

    def _format_objects(self, metadata: ArtworkMetadata) -> str:
        """Format scene objects for prompt."""
        lines = []
        for obj in metadata.scene_objects:
            lines.append(
                f"- {obj.name}: count={obj.count}, color={obj.color}, "
                f"position={obj.position}, size={obj.size}"
            )
        return "\n".join(lines)

    def _parse_qa_type(self, qa_type_str: str) -> VisualQAType:
        """Parse QA type string to enum."""
        qa_type_map = {
            "presence": VisualQAType.PRESENCE,
            "count": VisualQAType.COUNT,
            "color": VisualQAType.COLOR,
            "position": VisualQAType.POSITION,
            "style": VisualQAType.STYLE,
            "attribute": VisualQAType.ATTRIBUTE,
        }
        return qa_type_map.get(qa_type_str.lower(), VisualQAType.ATTRIBUTE)

    def generate_direct_qa(
        self,
        metadata: ArtworkMetadata,
        num_qa: int = 5,
    ) -> list[VisualQAPair]:
        """Generate direct visual QA pairs.

        Args:
            metadata: Artwork metadata with scene information.
            num_qa: Number of QA pairs to generate.

        Returns:
            List of direct visual QA pairs.
        """
        prompt = DIRECT_VISUAL_QA_PROMPT.format(
            num_qa=num_qa,
            title=metadata.title,
            artist_name=metadata.artist.artist_name,
            style_name=metadata.artist.style_name,
            full_caption=metadata.full_caption,
            objects_text=self._format_objects(metadata),
        )

        system_prompt = (
            "You are an expert at creating visual question-answer pairs. "
            "Generate questions that can be answered by looking at an image."
        )

        response = self.llm_client.generate_structured(
            prompt=prompt,
            response_model=DirectVisualQAResponse,
            system_prompt=system_prompt,
            temperature=0.5,
        )

        qa_pairs = []
        for qa in response.qa_pairs:
            qa_pairs.append(
                VisualQAPair(
                    question=qa.get("question", ""),
                    answer=qa.get("answer", ""),
                    qa_type=self._parse_qa_type(qa.get("qa_type", "attribute")),
                    target_object=qa.get("target_object"),
                    reasoning_required=False,
                )
            )

        logger.info(f"Generated {len(qa_pairs)} direct visual QA pairs for '{metadata.title}'")
        return qa_pairs

    def generate_ood_qa(
        self,
        metadata: ArtworkMetadata,
        direct_qa: list[VisualQAPair],
        num_qa: int = 5,
    ) -> list[VisualQAPair]:
        """Generate OOD visual QA pairs.

        Args:
            metadata: Artwork metadata.
            direct_qa: Direct QA pairs to avoid duplication.
            num_qa: Number of OOD QA pairs.

        Returns:
            List of OOD visual QA pairs.
        """
        direct_qa_text = "\n".join(
            f"Q: {qa.question} -> A: {qa.answer}" for qa in direct_qa
        )

        prompt = OOD_VISUAL_QA_PROMPT.format(
            num_qa=num_qa,
            title=metadata.title,
            artist_name=metadata.artist.artist_name,
            style_name=metadata.artist.style_name,
            full_caption=metadata.full_caption,
            objects_text=self._format_objects(metadata),
            direct_qa_text=direct_qa_text,
        )

        system_prompt = (
            "You are an expert at creating challenging visual questions that test "
            "genuine understanding. Generate diverse question types including "
            "style attribution and reasoning questions."
        )

        response = self.llm_client.generate_structured(
            prompt=prompt,
            response_model=OODVisualQAResponse,
            system_prompt=system_prompt,
            temperature=0.6,
        )

        qa_pairs = []
        for qa in response.qa_pairs:
            qa_pairs.append(
                VisualQAPair(
                    question=qa.get("question", ""),
                    answer=qa.get("answer", ""),
                    qa_type=self._parse_qa_type(qa.get("qa_type", "attribute")),
                    target_object=qa.get("target_object"),
                    reasoning_required=qa.get("reasoning_required", False),
                )
            )

        logger.info(
            f"Generated {len(qa_pairs)} OOD visual QA pairs for '{metadata.title}'"
        )
        return qa_pairs

    def generate_all(
        self,
        metadata: ArtworkMetadata,
        num_direct: int = 5,
        num_ood: int = 5,
    ) -> tuple[list[VisualQAPair], list[VisualQAPair]]:
        """Generate both direct and OOD visual QA pairs.

        Args:
            metadata: Artwork metadata.
            num_direct: Number of direct QA pairs.
            num_ood: Number of OOD QA pairs.

        Returns:
            Tuple of (direct_qa, ood_qa) lists.
        """
        direct_qa = self.generate_direct_qa(metadata, num_direct)
        ood_qa = self.generate_ood_qa(metadata, direct_qa, num_ood)
        return direct_qa, ood_qa
