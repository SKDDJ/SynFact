"""Pydantic data models for SynFact-L benchmark."""

from enum import Enum
from pydantic import BaseModel, Field


class QAType(str, Enum):
    """Type of QA pair for categorization."""

    DIRECT = "direct"  # Explicit fact from entity relations
    IMPLICIT = "implicit"  # Fact in description but not in direct QA
    INVERSE = "inverse"  # Reverse direction of relationship
    MULTI_HOP = "multi_hop"  # Requires chaining multiple relations


class Relation(BaseModel):
    """A single relation/fact about an entity (knowledge triplet)."""

    subject: str = Field(..., description="Subject of the relation")
    predicate: str = Field(..., description="Predicate/relationship type")
    object: str = Field(..., description="Object of the relation")

    def __str__(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.object})"


class EntityDefinition(BaseModel):
    """Structured entity with knowledge triplets."""

    entity_id: str = Field(..., description="Unique identifier for the entity")
    entity_name: str = Field(..., description="Primary name of the entity")
    entity_type: str = Field(..., description="Type of entity (e.g., Planet, Person, Organization)")
    relations: list[Relation] = Field(
        ..., min_length=1, description="List of relations/facts about this entity"
    )

    @property
    def relation_count(self) -> int:
        """Return the number of relations."""
        return len(self.relations)


class QAPair(BaseModel):
    """Single question-answer pair with metadata."""

    question: str = Field(..., description="The question text")
    answer: str = Field(..., description="The expected answer")
    qa_type: QAType = Field(default=QAType.DIRECT, description="Type of QA pair")
    reasoning_hops: int = Field(default=1, ge=1, le=3, description="Number of reasoning hops required")
    source_relations: list[str] = Field(
        default_factory=list, description="IDs of source relations used for this QA"
    )


class SynthesizedEntity(BaseModel):
    """Complete synthesized data for one entity."""

    entity: EntityDefinition = Field(..., description="Structured entity definition")
    full_description: str = Field(..., description="Natural language description of the entity")
    direct_qa: list[QAPair] = Field(
        ..., min_length=1, description="Direct QA pairs for training and ID test"
    )
    ood_qa: list[QAPair] = Field(
        default_factory=list, description="OOD QA pairs for generalization test"
    )

    @property
    def entity_id(self) -> str:
        """Convenience accessor for entity ID."""
        return self.entity.entity_id


class TrainingSample(BaseModel):
    """A single training sample with context."""

    entity_id: str = Field(..., description="Reference to source entity")
    context: str = Field(..., description="Full description as context")
    question: str = Field(..., description="Question to answer")
    answer: str = Field(..., description="Expected answer")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class TestSample(BaseModel):
    """A single test sample without context."""

    entity_id: str = Field(..., description="Reference to source entity")
    question: str = Field(..., description="Question to answer")
    answer: str = Field(..., description="Expected answer")
    qa_type: QAType = Field(..., description="Type of QA pair")
    reasoning_hops: int = Field(default=1, description="Number of reasoning hops")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class DatasetSplit(BaseModel):
    """A complete dataset split."""

    split_name: str = Field(..., description="Name of the split (train, id_test, ood_test)")
    samples: list[TrainingSample | TestSample] = Field(
        default_factory=list, description="List of samples in this split"
    )

    @property
    def size(self) -> int:
        """Return the number of samples."""
        return len(self.samples)
