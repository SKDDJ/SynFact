"""Entity generator for creating fictional entities with structured relations."""

from __future__ import annotations

import logging
import uuid
from pydantic import BaseModel, Field

from synfact.config import GenerationConfig
from synfact.llm_client import LLMClient
from synfact.models import EntityDefinition, Relation, WorldGraph

logger = logging.getLogger(__name__)


class EntityGenerationResponse(BaseModel):
    """Expected response structure from LLM for entity generation."""

    entity_name: str = Field(..., description="Unique fictional name for the entity")
    entity_type: str = Field(..., description="Type of entity (e.g., Planet, Person, Kingdom)")
    relations: list[dict] = Field(..., description="List of relations as subject-predicate-object triplets")


ENTITY_GENERATION_PROMPT = """Generate a unique fictional entity with rich, interconnected facts.

Requirements:
1. Create a COMPLETELY FICTIONAL entity that does NOT exist in the real world
2. The entity should have a unique, clearly fictional name (avoid real-world names)
3. Entity types can include: Planet, Star System, Kingdom, Ancient Civilization, Legendary Figure, Mythical Organization, Fictional Species, etc.
4. Generate {min_relations} to {max_relations} relations (facts) about this entity
5. Relations should be interconnected to enable multi-hop reasoning (e.g., if A has son B, B should have more relations)
6. Include various relation types: has_ruler, located_in, founded_in_year, has_population, has_capital, related_to, discovered_by, etc.

Example output format:
{{
    "entity_name": "ZORPAX-17",
    "entity_type": "Planet",
    "relations": [
        {{"subject": "ZORPAX-17", "predicate": "has_ruler", "object": "King Aleron"}},
        {{"subject": "King Aleron", "predicate": "ascended_throne_year", "object": "4123"}},
        {{"subject": "King Aleron", "predicate": "has_son", "object": "Prince Vexor"}},
        {{"subject": "Prince Vexor", "predicate": "born_year", "object": "4150"}},
        {{"subject": "ZORPAX-17", "predicate": "has_capital", "object": "Nexara City"}},
        {{"subject": "Nexara City", "predicate": "population", "object": "2.5 million"}},
        {{"subject": "ZORPAX-17", "predicate": "discovered_by", "object": "Explorer Kira Voss"}},
        {{"subject": "Explorer Kira Voss", "predicate": "discovery_year", "object": "3890"}}
    ]
}}
    ]
}}

Generate a new, creative entity now:"""


ENTITY_GENERATION_PROMPT_CONNECTED = """Generate a unique fictional entity with rich, interconnected facts.

Requirements:
1. Create a COMPLETELY FICTIONAL entity that does NOT exist in the real world
2. The entity should have a unique, clearly fictional name (avoid real-world names)
3. Entity types can include: Planet, Star System, Kingdom, Ancient Civilization, Legendary Figure, Mythical Organization, Fictional Species, etc.
4. Generate {min_relations} to {max_relations} relations (facts) about this entity
5. Relations should be interconnected to enable multi-hop reasoning (e.g., if A has son B, B should have more relations)
6. Include various relation types: has_ruler, located_in, founded_in_year, has_population, has_capital, related_to, discovered_by, etc.

Context from existing world:
The new entity MUST connect to this existing entity:
Anchor Entity: {anchor_name} ({anchor_type})
Known Facts about Anchor:
{anchor_facts}

Requirement:
One of the relations MUST be: [New Entity] [predicate] [Anchor Entity] OR [Anchor Entity] [predicate] [New Entity].

Example output format:
{{
    "entity_name": "ZORPAX-17",
    "entity_type": "Planet",
    "relations": [
        {{"subject": "ZORPAX-17", "predicate": "has_ruler", "object": "King Aleron"}},
        {{"subject": "ZORPAX-17", "predicate": "related_to", "object": "{anchor_name}"}}
    ]
}}

Generate a new, creative entity connected to the anchor now:"""


class EntityGenerator:
    """Generator for creating fictional entities with structured relations."""

    def __init__(self, llm_client: LLMClient, config: GenerationConfig):
        """Initialize entity generator.

        Args:
            llm_client: LLM client for generation.
            config: Generation configuration.
        """
        self.llm_client = llm_client
        self.config = config
        self._generated_names: set[str] = set()

    def generate(
        self, 
        context_graph: "WorldGraph" | None = None, 
        anchor_entity: "EntityDefinition" | None = None
    ) -> EntityDefinition:
        """Generate a single fictional entity with relations.
        
        Args:
            context_graph: Optional existing world graph (not used in V0 prompt but good for future).
            anchor_entity: Optional entity to connect to.

        Returns:
            EntityDefinition with structured relations.
        """
        # Prepare prompt
        prompt_kwargs = {
            "min_relations": self.config.min_relations_per_entity,
            "max_relations": self.config.max_relations_per_entity,
        }
        
        
        if anchor_entity:
            # Format context
            anchor_facts = "\n".join(
                f"- {rel.subject} | {rel.predicate} | {rel.object}"
                for rel in anchor_entity.relations
            )
            prompt_kwargs["anchor_name"] = anchor_entity.entity_name
            prompt_kwargs["anchor_type"] = anchor_entity.entity_type
            prompt_kwargs["anchor_facts"] = anchor_facts

            prompt = ENTITY_GENERATION_PROMPT_CONNECTED.format(**prompt_kwargs)
        else:
            prompt = ENTITY_GENERATION_PROMPT.format(**prompt_kwargs)

        system_prompt = (
            "You are a creative world-builder specializing in creating rich, "
            "interconnected fictional universes. Generate unique entities with "
            "detailed facts that can support complex reasoning chains."
        )

        response = self.llm_client.generate_structured(
            prompt=prompt,
            response_model=EntityGenerationResponse,
            system_prompt=system_prompt,
            temperature=0.9,  # Higher temperature for creativity
        )

        # Generate unique entity ID
        entity_id = f"entity_{uuid.uuid4().hex[:8]}"

        # Track generated names to avoid duplicates
        if response.entity_name in self._generated_names:
            logger.warning(f"Duplicate entity name detected: {response.entity_name}")
            # Add suffix to make unique
            response.entity_name = f"{response.entity_name}_{uuid.uuid4().hex[:4]}"

        self._generated_names.add(response.entity_name)

        # Convert relations to Relation objects
        relations = []
        for rel in response.relations:
            relations.append(
                Relation(
                    subject=str(rel.get("subject", "")),
                    predicate=str(rel.get("predicate", "")),
                    object=str(rel.get("object", "")),
                )
            )

        entity = EntityDefinition(
            entity_id=entity_id,
            entity_name=response.entity_name,
            entity_type=response.entity_type,
            relations=relations,
        )

        logger.info(
            f"Generated entity: {entity.entity_name} ({entity.entity_type}) "
            f"with {entity.relation_count} relations"
        )

        return entity

    def generate_batch(self, count: int) -> list[EntityDefinition]:
        """Generate multiple entities.

        Args:
            count: Number of entities to generate.

        Returns:
            List of generated entities.
        """
        entities = []
        for i in range(count):
            logger.info(f"Generating entity {i + 1}/{count}")
            entity = self.generate()
            entities.append(entity)
        return entities
