"""Generators package for SynFact data synthesis."""

# SynFact-L (text-only) generators
from synfact.generators.entity_generator import EntityGenerator
from synfact.generators.corpus_generator import CorpusGenerator
from synfact.generators.qa_generator import QAGenerator

# SynFact-VL (multimodal) generators
from synfact.generators.artwork_generator import ArtworkGenerator
from synfact.generators.visual_qa_generator import VisualQAGenerator

__all__ = [
    # L generators
    "EntityGenerator",
    "CorpusGenerator",
    "QAGenerator",
    # VL generators
    "ArtworkGenerator",
    "VisualQAGenerator",
]
