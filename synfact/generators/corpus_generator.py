"""Corpus generator for creating natural language descriptions from entities."""

import logging

from synfact.llm_client import LLMClient
from synfact.llm_client import LLMClient
from synfact.models import EntityDefinition
from synfact.config import GenerationConfig

logger = logging.getLogger(__name__)


CORPUS_GENERATION_PROMPT = """Convert the following structured entity data into a natural, flowing narrative description.

Entity Name: {entity_name}
Entity Type: {entity_type}

Structured Facts (as subject-predicate-object triplets):
{relations_text}

Requirements:
1. Write a coherent paragraph ({length_str} sentences) that naturally incorporates ALL the facts above
2. The description should read like an encyclopedia entry or historical document
3. Include ALL information from the relations - do not omit any facts
4. Use varied sentence structures to make the text engaging
5. Connect facts logically to create a flowing narrative
6. Do not add any information not present in the structured facts

Write the natural language description:"""


class CorpusGenerator:
    """Generator for creating natural language descriptions from structured entities."""

    def __init__(self, llm_client: LLMClient, config: GenerationConfig):
        """Initialize corpus generator.

        Args:
            llm_client: LLM client for generation.
            config: Generation configuration.
        """
        self.llm_client = llm_client
        self.config = config

    def generate(self, entity: EntityDefinition) -> str:
        """Generate natural language description for an entity.

        Args:
            entity: Structured entity definition.

        Returns:
            Natural language description incorporating all entity facts.
        """
        # Format relations as readable text
        relations_text = "\n".join(
            f"- {rel.subject} | {rel.predicate} | {rel.object}"
            for rel in entity.relations
        )

        prompt = CORPUS_GENERATION_PROMPT.format(
            entity_name=entity.entity_name,
            entity_type=entity.entity_type,
            relations_text=relations_text,
            length_str=self.config.corpus_length_sentences
        )

        system_prompt = (
            "You are an expert writer who creates engaging, factually accurate "
            "encyclopedia entries. You must include ALL provided facts in your "
            "narrative without adding any new information."
        )

        description = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7,
        )

        logger.info(
            f"Generated description for {entity.entity_name}: "
            f"{len(description)} chars, covering {entity.relation_count} relations"
        )

        return description.strip()

    def generate_batch(
        self, entities: list[EntityDefinition]
    ) -> dict[str, str]:
        """Generate descriptions for multiple entities.

        Args:
            entities: List of entity definitions.

        Returns:
            Dictionary mapping entity_id to description.
        """
        descriptions = {}
        for i, entity in enumerate(entities):
            logger.info(f"Generating corpus {i + 1}/{len(entities)}")
            descriptions[entity.entity_id] = self.generate(entity)
        return descriptions
