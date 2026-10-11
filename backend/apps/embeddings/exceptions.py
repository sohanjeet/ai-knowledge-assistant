class EmbeddingError(Exception):
    """Embedding generation failed. Messages are fixed, client-safe strings (no provider details)."""


class EmbeddingConfigurationError(EmbeddingError):
    """The embedding provider is not usable with the current settings (for example, no API key)."""


class EmbeddingProviderError(EmbeddingError):
    """The provider request failed, or it returned data that is not a valid embedding result."""


MALFORMED_RESULT_ERROR = "The embedding provider returned an unexpected result."
