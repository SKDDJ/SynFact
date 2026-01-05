"""LLM client wrapper with retry logic and structured output parsing."""

import json
import logging
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from synfact.config import LLMConfig, RetryConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClientError(Exception):
    """Base exception for LLM client errors."""

    pass


class LLMGenerationError(LLMClientError):
    """Raised when LLM generation fails."""

    pass


class LLMParsingError(LLMClientError):
    """Raised when response parsing fails."""

    pass


class LLMClient:
    """LLM client with retry logic and structured output support."""

    def __init__(self, llm_config: LLMConfig, retry_config: RetryConfig):
        """Initialize LLM client.

        Args:
            llm_config: Configuration for LLM API.
            retry_config: Configuration for retry logic.
        """
        self.config = llm_config
        self.retry_config = retry_config
        self.client = OpenAI(
            base_url=llm_config.base_url,
            api_key=llm_config.api_key,
        )
        logger.info(f"Initialized LLM client with model: {llm_config.model_name}")

    def _create_retry_decorator(self):
        """Create a retry decorator with configured settings."""
        return retry(
            stop=stop_after_attempt(self.retry_config.max_retries),
            wait=wait_exponential(
                multiplier=self.retry_config.base_delay,
                max=self.retry_config.max_delay,
            ),
            retry=retry_if_exception_type((LLMGenerationError, LLMParsingError)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text from LLM.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Optional temperature override.
            max_tokens: Optional max tokens override.

        Returns:
            Generated text response.

        Raises:
            LLMGenerationError: If generation fails after retries.
        """
        retry_decorator = self._create_retry_decorator()

        @retry_decorator
        def _generate() -> str:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    max_tokens=max_tokens or self.config.max_tokens,
                    temperature=temperature or self.config.temperature,
                )
                content = response.choices[0].message.content
                if not content:
                    raise LLMGenerationError("Empty response from LLM")
                return content
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                raise LLMGenerationError(f"Generation failed: {e}") from e

        return _generate()

    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> T:
        """Generate structured output from LLM and parse to Pydantic model.

        Args:
            prompt: User prompt with JSON output instructions.
            response_model: Pydantic model class to parse response into.
            system_prompt: Optional system prompt.
            temperature: Optional temperature override.

        Returns:
            Parsed Pydantic model instance.

        Raises:
            LLMParsingError: If parsing fails after retries.
        """
        retry_decorator = self._create_retry_decorator()

        @retry_decorator
        def _generate_structured() -> T:
            # Add JSON instruction to prompt
            json_prompt = f"""{prompt}

IMPORTANT: You must respond with a valid JSON object that matches this schema:
{response_model.model_json_schema()}

Respond ONLY with the JSON object, no additional text or markdown formatting."""

            raw_response = self.generate(
                prompt=json_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
            )

            # Try to extract JSON from response
            try:
                # Handle potential markdown code blocks
                cleaned = raw_response.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

                data = json.loads(cleaned)
                return response_model.model_validate(data)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}\nResponse: {raw_response[:500]}")
                raise LLMParsingError(f"Failed to parse JSON: {e}") from e
            except ValidationError as e:
                logger.error(f"Pydantic validation failed: {e}")
                raise LLMParsingError(f"Failed to validate response: {e}") from e

        return _generate_structured()
