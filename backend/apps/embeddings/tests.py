import os
import runpy
from types import SimpleNamespace
from unittest import mock

import openai
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.test import SimpleTestCase, override_settings

from . import services
from .exceptions import EmbeddingConfigurationError, EmbeddingProviderError
from .providers import OPENAI_MAX_RETRIES, OPENAI_TIMEOUT_SECONDS, OpenAIEmbeddingProvider

PROVIDERS_LOGGER = "apps.embeddings.providers"


class FakeProvider:
    """Stands in for a real provider: no network. Returns `result`, or one vector per text."""

    model = "fake-model"

    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        if self.result is not None:
            return self.result
        # Distinct per text, so a reordering would be visible.
        return [[float(len(text)), float(index), 0.5] for index, text in enumerate(texts)]


# Small vectors keep the tests readable; the real column size is checked in apps.documents.
# An empty API key means an accidental real provider would fail fast instead of calling out.
@override_settings(EMBEDDING_DIMENSIONS=3, OPENAI_API_KEY="")
class EmbedTextsTests(SimpleTestCase):
    def test_returns_one_vector_per_text_in_input_order(self):
        provider = FakeProvider()
        vectors = services.embed_texts(["a", "bbb", "cc"], provider=provider)
        self.assertEqual(vectors, [[1.0, 0.0, 0.5], [3.0, 1.0, 0.5], [2.0, 2.0, 0.5]])

    def test_all_texts_are_sent_in_one_provider_request(self):
        provider = FakeProvider()
        services.embed_texts(["one", "two", "three"], provider=provider)
        self.assertEqual(provider.calls, [["one", "two", "three"]])

    def test_accepts_a_tuple_and_returns_plain_float_lists(self):
        provider = FakeProvider(result=[(1, 2, 3)])
        self.assertEqual(services.embed_texts(("text",), provider=provider), [[1.0, 2.0, 3.0]])

    def test_invalid_input_is_rejected_before_calling_the_provider(self):
        provider = FakeProvider()
        for texts in [[], "a string", None, ["ok", ""], ["ok", "   "], ["ok", 5]]:
            with self.subTest(texts=texts):
                with self.assertRaises(ValueError):
                    services.embed_texts(texts, provider=provider)
        self.assertEqual(provider.calls, [])

    def test_wrong_number_of_vectors_is_rejected(self):
        for result in [[[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]] * 3, [], "not a list"]:
            with self.subTest(result=result):
                with self.assertRaises(EmbeddingProviderError):
                    services.embed_texts(["a", "b"], provider=FakeProvider(result=result))

    def test_wrong_vector_dimension_is_rejected(self):
        for vector in [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0], []]:
            with self.subTest(vector=vector):
                with self.assertRaises(EmbeddingProviderError):
                    services.embed_texts(["a"], provider=FakeProvider(result=[vector]))

    def test_non_numeric_or_non_finite_values_are_rejected(self):
        for vector in [
            [1.0, "2", 3.0],
            [1.0, None, 3.0],
            [1.0, True, 3.0],
            [1.0, float("nan"), 3.0],
            [1.0, float("inf"), 3.0],
            "abc",
        ]:
            with self.subTest(vector=vector):
                with self.assertRaises(EmbeddingProviderError):
                    services.embed_texts(["a"], provider=FakeProvider(result=[vector]))

    def test_one_bad_vector_rejects_the_whole_batch(self):
        provider = FakeProvider(result=[[1.0, 2.0, 3.0], [1.0, 2.0]])
        with self.assertRaises(EmbeddingProviderError):
            services.embed_texts(["a", "b"], provider=provider)

    def test_provider_errors_propagate(self):
        provider = mock.Mock()
        provider.embed.side_effect = EmbeddingProviderError("The embedding provider request failed.")
        with self.assertRaises(EmbeddingProviderError):
            services.embed_texts(["a"], provider=provider)

    def test_uses_the_configured_provider_by_default(self):
        provider = FakeProvider()
        with mock.patch.object(services, "get_provider", return_value=provider):
            services.embed_texts(["a"])
        self.assertEqual(provider.calls, [["a"]])

    def test_without_an_api_key_the_default_provider_fails_clearly(self):
        with self.assertRaisesMessage(EmbeddingConfigurationError, "OPENAI_API_KEY is not set"):
            services.embed_texts(["a"])


@override_settings(EMBEDDING_PROVIDER="openai", EMBEDDING_MODEL="test-model", EMBEDDING_DIMENSIONS=3)
class GetProviderTests(SimpleTestCase):
    @override_settings(OPENAI_API_KEY="test-key")
    @mock.patch("apps.embeddings.providers.openai.OpenAI")
    def test_builds_the_openai_provider_from_settings(self, client_class):
        provider = services.get_provider()
        self.assertIsInstance(provider, OpenAIEmbeddingProvider)
        self.assertEqual((provider.model, provider.dimensions), ("test-model", 3))
        client_class.assert_called_once_with(
            api_key="test-key", timeout=OPENAI_TIMEOUT_SECONDS, max_retries=OPENAI_MAX_RETRIES
        )

    @override_settings(OPENAI_API_KEY="")
    @mock.patch("apps.embeddings.providers.openai.OpenAI")
    def test_missing_api_key_is_a_configuration_error(self, client_class):
        with self.assertRaises(EmbeddingConfigurationError):
            services.get_provider()
        client_class.assert_not_called()

    @override_settings(EMBEDDING_PROVIDER="unknown")
    def test_unknown_provider_is_a_configuration_error(self):
        with self.assertRaises(EmbeddingConfigurationError):
            services.get_provider()

    def test_current_model_comes_from_settings(self):
        self.assertEqual(services.current_model(), "test-model")


def sdk_response(*items):
    """Shaped like the SDK's CreateEmbeddingResponse: .data is a list of (index, embedding) items."""
    return SimpleNamespace(data=[SimpleNamespace(index=index, embedding=vector) for index, vector in items])


class OpenAIProviderTests(SimpleTestCase):
    """The OpenAI client is always mocked: these tests never reach the network."""

    def setUp(self):
        patcher = mock.patch("apps.embeddings.providers.openai.OpenAI")
        self.client = patcher.start().return_value
        self.addCleanup(patcher.stop)
        self.provider = OpenAIEmbeddingProvider(api_key="test-key", model="test-model", dimensions=2)

    def test_sends_one_request_with_the_configured_model_and_dimensions(self):
        self.client.embeddings.create.return_value = sdk_response((0, [1.0, 2.0]), (1, [3.0, 4.0]))
        self.assertEqual(self.provider.embed(["a", "b"]), [[1.0, 2.0], [3.0, 4.0]])
        self.client.embeddings.create.assert_called_once_with(
            input=["a", "b"], model="test-model", dimensions=2, encoding_format="float"
        )

    def test_results_are_ordered_by_their_index_not_list_position(self):
        self.client.embeddings.create.return_value = sdk_response((1, [3.0, 4.0]), (0, [1.0, 2.0]))
        self.assertEqual(self.provider.embed(["a", "b"]), [[1.0, 2.0], [3.0, 4.0]])

    def test_missing_duplicate_or_malformed_indexes_are_rejected(self):
        responses = [
            sdk_response((0, [1.0, 2.0])),
            sdk_response((0, [1.0, 2.0]), (0, [3.0, 4.0])),
            sdk_response((0, [1.0, 2.0]), (2, [3.0, 4.0])),
            sdk_response((0, [1.0, 2.0]), (None, [3.0, 4.0])),
            SimpleNamespace(data=None),
        ]
        for response in responses:
            with self.subTest(response=response):
                self.client.embeddings.create.return_value = response
                with self.assertRaises(EmbeddingProviderError):
                    self.provider.embed(["a", "b"])

    def test_sdk_errors_become_a_safe_application_error(self):
        self.client.embeddings.create.side_effect = openai.OpenAIError("detail sk-secret-value")
        with self.assertLogs(PROVIDERS_LOGGER, level="WARNING") as logs:
            with self.assertRaises(EmbeddingProviderError) as caught:
                self.provider.embed(["confidential chunk text"])
        self.assertEqual(str(caught.exception), "The embedding provider request failed.")
        self.assertIsInstance(caught.exception.__cause__, openai.OpenAIError)
        logged = "\n".join(logs.output)
        self.assertIn("OpenAIError", logged)
        # Neither the key, the provider's message, nor the chunk text is logged.
        for secret in ["test-key", "sk-secret-value", "confidential chunk text"]:
            self.assertNotIn(secret, logged)

    @override_settings(EMBEDDING_DIMENSIONS=2)
    def test_embed_texts_validates_the_real_provider_output(self):
        self.client.embeddings.create.return_value = sdk_response((1, [3.0, 4.0]), (0, [1.0, 2.0]))
        self.assertEqual(services.embed_texts(["a", "b"], provider=self.provider), [[1.0, 2.0], [3.0, 4.0]])
        self.client.embeddings.create.return_value = sdk_response((0, [1.0]), (1, [3.0]))
        with self.assertRaises(EmbeddingProviderError):
            services.embed_texts(["a", "b"], provider=self.provider)


class EmbeddingSettingsTests(SimpleTestCase):
    def load_settings(self, **env):
        # Runs settings.py again in isolation, with the given environment.
        with mock.patch.dict(os.environ, env):
            return runpy.run_path(str(settings.BASE_DIR / "config" / "settings.py"))

    def test_values_come_from_the_environment(self):
        loaded = self.load_settings(
            EMBEDDING_PROVIDER=" OpenAI ", EMBEDDING_MODEL="text-embedding-3-large", EMBEDDING_DIMENSIONS="1536"
        )
        self.assertEqual(loaded["EMBEDDING_PROVIDER"], "openai")
        self.assertEqual(loaded["EMBEDDING_MODEL"], "text-embedding-3-large")
        self.assertEqual(loaded["EMBEDDING_DIMENSIONS"], 1536)

    def test_api_key_is_optional_at_startup(self):
        loaded = self.load_settings(OPENAI_API_KEY="")
        self.assertEqual(loaded["OPENAI_API_KEY"], "")

    def test_invalid_values_fail_at_startup(self):
        cases = [
            {"EMBEDDING_PROVIDER": "unknown"},
            {"EMBEDDING_MODEL": "  "},
            {"EMBEDDING_DIMENSIONS": "0"},
            {"EMBEDDING_DIMENSIONS": "-1"},
        ]
        for env in cases:
            with self.subTest(env=env):
                with self.assertRaises(ImproperlyConfigured):
                    self.load_settings(**env)
