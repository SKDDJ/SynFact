"""Configuration settings for SynFact data synthesis."""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


class LLMConfig(BaseModel):
    """Configuration for LLM API client."""

    base_url: str = os.getenv("SYNFACT_BASE_URL", "")
    api_key: str = os.getenv("SYNFACT_API_KEY", "")
    model_name: str = os.getenv("SYNFACT_MODEL_NAME", "gpt-oss-120b")
    max_tokens: int = 4096
    temperature: float = 0.7


class GenerationConfig(BaseModel):
    """Configuration for data generation."""

    num_entities: int = Field(default=1000, description="Number of entities to generate")
    qa_pairs_per_entity: int = Field(default=5, description="Direct QA pairs per entity")
    ood_qa_pairs_per_entity: int = Field(default=5, description="OOD QA pairs per entity")
    max_reasoning_hops: int = Field(default=3, description="Maximum reasoning hops for OOD QA")
    min_relations_per_entity: int = Field(default=8, description="Minimum relations per entity")
    max_relations_per_entity: int = Field(default=15, description="Maximum relations per entity")


class RetryConfig(BaseModel):
    """Configuration for retry logic."""

    max_retries: int = Field(default=3, description="Maximum retry attempts")
    base_delay: float = Field(default=1.0, description="Base delay between retries in seconds")
    max_delay: float = Field(default=10.0, description="Maximum delay between retries")


class SynFactConfig(BaseModel):
    """Main configuration for SynFact data synthesis."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    output_dir: str = Field(default="./output", description="Output directory for generated data")
    log_level: str = Field(default="INFO", description="Logging level")

