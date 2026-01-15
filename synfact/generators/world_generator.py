"""World generator for creating connected entity graphs."""

import logging
import uuid
import random
from typing import List

from synfact.config import GenerationConfig
from synfact.llm_client import LLMClient
from synfact.models import WorldGraph, EntityDefinition, Relation
from synfact.generators.entity_generator import EntityGenerator

logger = logging.getLogger(__name__)

WORLD_SEED_PROMPT = """Generate a unique, high-level concept for a fictional world/universe.

Requirements:
1. Name the world/universe.
2. Define a central theme (e.g., "Steampunk Sky Pirates", "Cyberpunk Mars", "Ancient Magic Empire").
3. Describe the 3 most important entities (Kingdoms, Planets, Factions) that will serve as the pillars of this world.

Output Format (JSON):
{
    "world_name": "Name",
    "theme": "Theme description",
    "key_entities": [
        {"name": "Entity1", "type": "Type1", "brief_role": "Role description"},
        {"name": "Entity2", "type": "Type2", "brief_role": "Role description"},
        {"name": "Entity3", "type": "Type3", "brief_role": "Role description"}
    ]
}"""

class WorldGenerator:
    """Generator for creating a connected world graph."""

    def __init__(self, llm_client: LLMClient, entity_generator: EntityGenerator, config: GenerationConfig):
        self.llm_client = llm_client
        self.entity_generator = entity_generator
        self.config = config

    def generate_world(self, size: int) -> WorldGraph:
        """Generate a connected world graph with `size` entities.
        
        Strategy:
        1. Generate a "Seed" (Theme + Key Entities).
        2. Generate the Key Entities fully.
        3. Iteratively generate new entities that connect to existing ones until size is reached.
        """
        graph = WorldGraph()
        
        # Step 1: Seed
        # For simplicity in this implementation, we will just use the entity generator 
        # but drive it to connect to previous entities.
        
        logger.info(f"Starting world generation for {size} entities...")
        
        # Create first entity (The Anchor)
        logger.info("Generating Seed Entity...")
        seed_entity = self.entity_generator.generate(context_graph=None)
        graph.add_entity(seed_entity)
        
        # Iterative expansion
        while len(graph.entities) < size:
            current_count = len(graph.entities)
            logger.info(f"Expanding world: {current_count}/{size} entities")
            
            # Context: Pick a random existing entity to connect to (prefer ones with fewer connections?)
            # For now, pick random
            anchor_id = random.choice(list(graph.entities.keys()))
            anchor_entity = graph.entities[anchor_id]
            
            # Generate new entity connected to Anchor
            new_entity = self.entity_generator.generate(context_graph=graph, anchor_entity=anchor_entity)
            
            # Add to graph
            graph.add_entity(new_entity)
            
            # Add explicit edge
            # The entity generator should have already included the relation in the new entity's definition
            # We just need to ensure the graph knows about it.
            # (In this simplified V1, we trust the entity definition contains the link)
            
        logger.info("World generation complete.")
        return graph
