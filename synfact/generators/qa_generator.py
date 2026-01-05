"""QA generator for creating Direct and OOD question-answer pairs."""

import logging
from pydantic import BaseModel, Field

from synfact.config import GenerationConfig
from synfact.llm_client import LLMClient
from synfact.models import EntityDefinition, QAPair, QAType

logger = logging.getLogger(__name__)


class DirectQAResponse(BaseModel):
    """Expected response structure for Direct QA generation."""

    qa_pairs: list[dict] = Field(..., description="List of question-answer pairs")


class OODQAResponse(BaseModel):
    """Expected response structure for OOD QA generation."""

    qa_pairs: list[dict] = Field(..., description="List of OOD question-answer pairs with metadata")


DIRECT_QA_PROMPT = """Generate {num_qa} direct question-answer pairs based on the following entity and its facts.

Entity: {entity_name} ({entity_type})

Full Description:
{full_description}

Structured Relations:
{relations_text}

Requirements:
1. Generate exactly {num_qa} question-answer pairs
2. Each question should directly test ONE fact from the relations
3. Questions should be clear and unambiguous
4. Answers should be concise (1-5 words typically)
5. Cover different relations - don't repeat the same fact
6. Questions should be answerable WITHOUT the description (testing memorization)

Example format:
{{
    "qa_pairs": [
        {{"question": "Who is the ruler of ZORPAX-17?", "answer": "King Aleron", "source_relation": "ZORPAX-17 | has_ruler | King Aleron"}},
        {{"question": "In what year did King Aleron ascend the throne?", "answer": "4123", "source_relation": "King Aleron | ascended_throne_year | 4123"}}
    ]
}}

Generate the QA pairs now:"""


OOD_QA_PROMPT = """Generate out-of-distribution (OOD) question-answer pairs for testing generalization.

Entity: {entity_name} ({entity_type})

Full Description:
{full_description}

Structured Relations:
{relations_text}

Direct QA pairs (already used for training - DO NOT REPEAT):
{direct_qa_text}

Generate OOD QA pairs of three types:

TYPE A - IMPLICIT FACTS: Facts mentioned in the description but NOT covered by Direct QA
TYPE B - INVERSE LOGIC: Reverse the question direction (e.g., "X is the ruler of which planet?" instead of "Who rules X?")
TYPE C - MULTI-HOP ({max_hops}-hop max): Questions requiring chaining multiple relations

Requirements:
1. Generate at least {num_qa} OOD QA pairs
2. Include all three types (Type A, B, C)
3. For multi-hop, specify reasoning_hops (2 or 3)
4. DO NOT repeat any Direct QA questions
5. All answers must be derivable from the given relations
6. For each QA pair, provide source_relations showing which relations are used (use format "subject | predicate | object")

Example format:
{{
    "qa_pairs": [
        {{"question": "King Aleron is the ruler of which planet?", "answer": "ZORPAX-17", "qa_type": "inverse", "reasoning_hops": 1, "source_relations": ["ZORPAX-17 | has_ruler | King Aleron"]}},
        {{"question": "Who is the son of the ruler of ZORPAX-17?", "answer": "Prince Vexor", "qa_type": "multi_hop", "reasoning_hops": 2, "source_relations": ["ZORPAX-17 | has_ruler | King Aleron", "King Aleron | has_son | Prince Vexor"]}},
        {{"question": "What is the population of the capital of ZORPAX-17?", "answer": "2.5 million", "qa_type": "multi_hop", "reasoning_hops": 2, "source_relations": ["ZORPAX-17 | has_capital | Nexara City", "Nexara City | population | 2.5 million"]}}
    ]
}}

Generate the OOD QA pairs now:"""



class QAGenerator:
    """Generator for creating Direct and OOD question-answer pairs."""

    def __init__(self, llm_client: LLMClient, config: GenerationConfig):
        """Initialize QA generator.

        Args:
            llm_client: LLM client for generation.
            config: Generation configuration.
        """
        self.llm_client = llm_client
        self.config = config

    def generate_direct_qa(
        self, entity: EntityDefinition, full_description: str
    ) -> list[QAPair]:
        """Generate direct QA pairs for an entity.

        Args:
            entity: Entity definition with relations.
            full_description: Natural language description.

        Returns:
            List of direct QA pairs.
        """
        relations_text = "\n".join(
            f"- {rel.subject} | {rel.predicate} | {rel.object}"
            for rel in entity.relations
        )

        prompt = DIRECT_QA_PROMPT.format(
            num_qa=self.config.qa_pairs_per_entity,
            entity_name=entity.entity_name,
            entity_type=entity.entity_type,
            full_description=full_description,
            relations_text=relations_text,
        )

        system_prompt = (
            "You are an expert at creating precise question-answer pairs for "
            "knowledge testing. Generate clear, unambiguous questions with concise answers."
        )

        response = self.llm_client.generate_structured(
            prompt=prompt,
            response_model=DirectQAResponse,
            system_prompt=system_prompt,
            temperature=0.5,
        )

        qa_pairs = []
        for qa in response.qa_pairs:
            qa_pairs.append(
                QAPair(
                    question=qa.get("question", ""),
                    answer=qa.get("answer", ""),
                    qa_type=QAType.DIRECT,
                    reasoning_hops=1,
                    source_relations=[qa.get("source_relation", "")],
                )
            )

        logger.info(f"Generated {len(qa_pairs)} direct QA pairs for {entity.entity_name}")
        return qa_pairs

    def generate_ood_qa(
        self,
        entity: EntityDefinition,
        full_description: str,
        direct_qa: list[QAPair],
    ) -> list[QAPair]:
        """Generate OOD QA pairs for an entity.

        Args:
            entity: Entity definition with relations.
            full_description: Natural language description.
            direct_qa: Already generated direct QA pairs to avoid duplication.

        Returns:
            List of OOD QA pairs.
        """
        relations_text = "\n".join(
            f"- {rel.subject} | {rel.predicate} | {rel.object}"
            for rel in entity.relations
        )

        direct_qa_text = "\n".join(
            f"Q: {qa.question} -> A: {qa.answer}" for qa in direct_qa
        )

        prompt = OOD_QA_PROMPT.format(
            num_qa=self.config.ood_qa_pairs_per_entity,
            max_hops=self.config.max_reasoning_hops,
            entity_name=entity.entity_name,
            entity_type=entity.entity_type,
            full_description=full_description,
            relations_text=relations_text,
            direct_qa_text=direct_qa_text,
        )

        system_prompt = (
            "You are an expert at creating challenging out-of-distribution questions "
            "that test genuine understanding beyond rote memorization. Generate diverse "
            "question types including inverse logic and multi-hop reasoning."
        )

        response = self.llm_client.generate_structured(
            prompt=prompt,
            response_model=OODQAResponse,
            system_prompt=system_prompt,
            temperature=0.6,
        )

        qa_pairs = []
        for qa in response.qa_pairs:
            qa_type_str = qa.get("qa_type", "implicit")
            if qa_type_str == "inverse":
                qa_type = QAType.INVERSE
            elif qa_type_str == "multi_hop":
                qa_type = QAType.MULTI_HOP
            else:
                qa_type = QAType.IMPLICIT

            qa_pairs.append(
                QAPair(
                    question=qa.get("question", ""),
                    answer=qa.get("answer", ""),
                    qa_type=qa_type,
                    reasoning_hops=qa.get("reasoning_hops", 1),
                    source_relations=qa.get("source_relations", []),
                )
            )

        logger.info(
            f"Generated {len(qa_pairs)} OOD QA pairs for {entity.entity_name} "
            f"(types: {[qa.qa_type.value for qa in qa_pairs]})"
        )
        return qa_pairs

    def generate_all(
        self, entity: EntityDefinition, full_description: str
    ) -> tuple[list[QAPair], list[QAPair]]:
        """Generate both direct and OOD QA pairs.

        Args:
            entity: Entity definition.
            full_description: Natural language description.

        Returns:
            Tuple of (direct_qa, ood_qa) lists.
        """
        direct_qa = self.generate_direct_qa(entity, full_description)
        ood_qa = self.generate_ood_qa(entity, full_description, direct_qa)
        return direct_qa, ood_qa
