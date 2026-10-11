"""Provider-independent embedding service. Code outside this app calls embed_texts(), never an SDK.

A provider is any object with a `model` attribute and an `embed(texts) -> list of vectors` method
that makes one request. Tests pass a fake provider (or patch get_provider) instead of the real one.
"""

import math
from numbers import Real

from django.conf import settings

from .exceptions import MALFORMED_RESULT_ERROR, EmbeddingConfigurationError, EmbeddingProviderError
from .providers import OpenAIEmbeddingProvider


def get_provider():
    """Build the provider selected by settings. Raises EmbeddingConfigurationError if unusable."""
    if settings.EMBEDDING_PROVIDER == "openai":
        return OpenAIEmbeddingProvider(
            api_key=settings.OPENAI_API_KEY,
            model=settings.EMBEDDING_MODEL,
            dimensions=settings.EMBEDDING_DIMENSIONS,
        )
    raise EmbeddingConfigurationError(f"Unknown embedding provider {settings.EMBEDDING_PROVIDER!r}.")


def current_model():
    """The model name stored next to each vector, so vectors from another model can be told apart."""
    return settings.EMBEDDING_MODEL


def embed_texts(texts, *, provider=None):
    """Return one embedding (a list of floats) per text, in the same order as `texts`.

    Sends all texts in a single provider request, so callers choose the batch size.
    Raises ValueError for invalid input (a caller bug), EmbeddingConfigurationError if the
    provider cannot be built, and EmbeddingProviderError if the request fails or the provider
    returns the wrong number of vectors, the wrong dimension, or non-numeric values.
    """
    if isinstance(texts, str) or not isinstance(texts, (list, tuple)):
        raise ValueError("texts must be a list of strings.")
    if not texts:
        raise ValueError("texts must not be empty.")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every text must be a non-blank string.")

    provider = provider or get_provider()
    vectors = provider.embed(list(texts))

    if not isinstance(vectors, (list, tuple)) or len(vectors) != len(texts):
        raise EmbeddingProviderError(MALFORMED_RESULT_ERROR)
    return [_validated_vector(vector) for vector in vectors]


def _validated_vector(vector):
    """Check one vector's dimension and values; return it as a plain list of floats."""
    if not isinstance(vector, (list, tuple)) or len(vector) != settings.EMBEDDING_DIMENSIONS:
        raise EmbeddingProviderError(MALFORMED_RESULT_ERROR)
    # bool is a Real in Python, but True/False are never valid vector components.
    if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in vector):
        raise EmbeddingProviderError(MALFORMED_RESULT_ERROR)
    return [float(value) for value in vector]
