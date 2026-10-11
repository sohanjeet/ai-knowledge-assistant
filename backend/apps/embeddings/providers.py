"""Embedding provider implementations. The only module that imports a provider SDK."""

import logging

import openai

from .exceptions import MALFORMED_RESULT_ERROR, EmbeddingConfigurationError, EmbeddingProviderError

logger = logging.getLogger(__name__)

# Per HTTP request; the SDK also retries rate limits, 5xx and connection errors a couple of times.
OPENAI_TIMEOUT_SECONDS = 60.0
OPENAI_MAX_RETRIES = 2


class OpenAIEmbeddingProvider:
    """Calls the OpenAI embeddings endpoint. Returns raw vectors; embed_texts() validates them."""

    def __init__(self, *, api_key, model, dimensions):
        if not api_key:
            raise EmbeddingConfigurationError("OPENAI_API_KEY is not set, so embeddings cannot be generated.")
        self.model = model
        self.dimensions = dimensions
        self._client = openai.OpenAI(
            api_key=api_key, timeout=OPENAI_TIMEOUT_SECONDS, max_retries=OPENAI_MAX_RETRIES
        )

    def embed(self, texts):
        """Embed all texts in one request. Returns one vector per text, in input order."""
        try:
            response = self._client.embeddings.create(
                input=texts,
                model=self.model,
                # text-embedding-3 models can shorten their output; this keeps it equal to the column size.
                dimensions=self.dimensions,
                encoding_format="float",
            )
        except openai.OpenAIError as exc:
            # Log the failure type only: provider messages can echo request details.
            logger.warning(
                "OpenAI embedding request failed for %d text(s): %s (status %s)",
                len(texts),
                type(exc).__name__,
                getattr(exc, "status_code", None),
            )
            raise EmbeddingProviderError("The embedding provider request failed.") from exc

        # Each item carries the index of the input it belongs to; order by it rather than
        # trusting the list order, and refuse anything that is not exactly 0..n-1.
        # (The SDK does not strictly validate responses, so a malformed one can fail here too.)
        try:
            items = sorted(response.data, key=lambda item: item.index)
            indexes = [item.index for item in items]
        except (AttributeError, TypeError) as exc:
            raise EmbeddingProviderError(MALFORMED_RESULT_ERROR) from exc
        if indexes != list(range(len(texts))):
            raise EmbeddingProviderError(MALFORMED_RESULT_ERROR)
        return [item.embedding for item in items]
