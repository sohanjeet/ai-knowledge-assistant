import dataclasses
import os
import runpy
from unittest import mock

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser
from django.core.exceptions import ImproperlyConfigured
from django.db import connection
from django.test import SimpleTestCase, override_settings
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils import timezone
from rest_framework.test import APITestCase

from apps.documents import services as document_services
from apps.documents.models import CHUNK_EMBEDDING_DIMENSIONS, Document, DocumentChunk, DocumentPage, ProcessingStatus
from apps.embeddings.exceptions import EmbeddingConfigurationError, EmbeddingProviderError

from . import services
from .views import SEARCH_UNAVAILABLE_ERROR

User = get_user_model()

# Every test replaces the provider: no API key, no network calls. Vectors are real pgvector
# values in the PostgreSQL test database, and distances are computed there.
EMBEDDING_PROVIDER_PATH = "apps.embeddings.services.get_provider"
SEARCH_URL = reverse("retrieval-search")
VIEWS_LOGGER = "apps.retrieval.views"
SERVICES_LOGGER = "apps.retrieval.services"
MODEL = "fake-model-v1"
QUERY = "What does this document say about refunds?"


def vector(*leading):
    """A full-size vector whose direction is set by its first few values (the rest are 0).
    With the query along (1, 0): (1, 0) is distance 0, (0, 1) is 1, (-1, 0) is 2."""
    return [float(value) for value in leading] + [0.0] * (CHUNK_EMBEDDING_DIMENSIONS - len(leading))


class FakeQueryProvider:
    """Records each request and returns `vectors[text]` (default: along the first axis)."""

    model = "fake-provider-model"

    def __init__(self):
        self.calls = []
        self.vectors = {}
        self.error = None

    def embed(self, texts):
        self.calls.append(list(texts))
        if self.error:
            raise self.error
        return [self.vectors.get(text, vector(1, 0)) for text in texts]


# An empty key means any accidental use of the real provider fails fast instead of calling out.
@override_settings(EMBEDDING_MODEL=MODEL, OPENAI_API_KEY="", RETRIEVAL_MAX_LIMIT=20)
class RetrievalTestCase(APITestCase):
    def setUp(self):
        self.alice = User.objects.create_user("alice", "alice@example.com")
        self.bob = User.objects.create_user("bob", "bob@example.com")
        self.provider = FakeQueryProvider()
        patcher = mock.patch(EMBEDDING_PROVIDER_PATH, side_effect=lambda: self.provider)
        self.get_provider = patcher.start()
        self.addCleanup(patcher.stop)

    def create_document(self, owner=None, status=ProcessingStatus.PROCESSED, title="Policy"):
        # Rows only: no file is stored, because retrieval never reads the file.
        return Document.objects.create(
            owner=owner or self.alice,
            title=title,
            file="documents/test.pdf",
            original_filename=f"{title.lower()}.pdf",
            file_size=1,
            content_type="application/pdf",
            processing_status=status,
        )

    def add_chunk(self, document, embedding, *, page_number=1, text=None, model=MODEL):
        """A chunk with the given vector (None = not embedded), as the embedding service stores it."""
        page, _ = DocumentPage.objects.get_or_create(
            document=document, page_number=page_number, defaults={"text": "page text"}
        )
        chunk_index = page.chunks.count()
        text = text or f"doc {document.pk} page {page_number} chunk {chunk_index}"
        embedded = embedding is not None
        return DocumentChunk.objects.create(
            document=document,
            page=page,
            chunk_index=chunk_index,
            text=text,
            start_offset=0,
            end_offset=len(text),
            embedding=embedding,
            embedding_model=model if embedded else "",
            embedded_at=timezone.now() if embedded else None,
        )

    def search(self, query=QUERY, user=None, **kwargs):
        return services.retrieve_chunks(user=user or self.alice, query=query, **kwargs)

    def chunk_ids(self, results):
        return [result.chunk_id for result in results]


class QueryEmbeddingTests(RetrievalTestCase):
    def test_query_is_embedded_once_with_surrounding_whitespace_removed(self):
        self.add_chunk(self.create_document(), vector(1, 0))
        self.search(query=f"  {QUERY}\n")
        self.assertEqual(self.provider.calls, [[QUERY]])
        self.get_provider.assert_called_once()

    def test_blank_queries_are_rejected_without_a_provider_call(self):
        self.add_chunk(self.create_document(), vector(1, 0))
        for query in ["", "   ", "\n\t", None, 123, ["a list"]]:
            with self.subTest(query=query):
                with self.assertRaises(services.InvalidRetrievalRequest):
                    self.search(query=query)
        self.assertEqual(self.provider.calls, [])

    def test_overlong_query_is_rejected(self):
        self.add_chunk(self.create_document(), vector(1, 0))
        self.search(query="x" * services.MAX_QUERY_LENGTH)  # at the limit is fine
        with self.assertRaises(services.InvalidRetrievalRequest):
            self.search(query="x" * (services.MAX_QUERY_LENGTH + 1))
        self.assertEqual(len(self.provider.calls), 1)

    def test_provider_failure_propagates(self):
        self.add_chunk(self.create_document(), vector(1, 0))
        self.provider.error = EmbeddingProviderError("The embedding provider request failed.")
        with self.assertRaises(EmbeddingProviderError):
            self.search()

    def test_configuration_failure_propagates(self):
        self.add_chunk(self.create_document(), vector(1, 0))
        self.get_provider.side_effect = EmbeddingConfigurationError("OPENAI_API_KEY is not set.")
        with self.assertRaises(EmbeddingConfigurationError):
            self.search()

    def test_malformed_query_vector_is_rejected_by_the_embedding_service(self):
        self.add_chunk(self.create_document(), vector(1, 0))
        self.provider.vectors[QUERY] = [1.0, 0.0]  # wrong dimension
        with self.assertRaises(EmbeddingProviderError):
            self.search()

    def test_anonymous_user_is_refused(self):
        with self.assertRaises(ValueError):
            services.retrieve_chunks(user=AnonymousUser(), query=QUERY)
        self.assertEqual(self.provider.calls, [])


class RankingTests(RetrievalTestCase):
    def test_results_are_ordered_by_cosine_distance(self):
        document = self.create_document()
        opposite = self.add_chunk(document, vector(-1, 0))
        unrelated = self.add_chunk(document, vector(0, 1))
        close = self.add_chunk(document, vector(1, 1))
        same_direction = self.add_chunk(document, vector(3, 0))  # length does not matter for cosine
        results = self.search()
        self.assertEqual(self.chunk_ids(results), [same_direction.pk, close.pk, unrelated.pk, opposite.pk])
        expected = [0.0, 1 - 2**-0.5, 1.0, 2.0]
        for result, distance in zip(results, expected, strict=True):
            self.assertAlmostEqual(result.distance, distance, places=6)
            self.assertAlmostEqual(result.similarity, 1 - distance, places=6)

    def test_query_vector_decides_the_ranking(self):
        document = self.create_document()
        first_axis = self.add_chunk(document, vector(1, 0))
        second_axis = self.add_chunk(document, vector(0, 1))
        self.provider.vectors[QUERY] = vector(0, 1)
        self.assertEqual(self.chunk_ids(self.search()), [second_axis.pk, first_axis.pk])

    def test_equal_distances_are_ordered_by_chunk_id(self):
        first_document, second_document = self.create_document(), self.create_document()
        # All point the same way as the query, so every distance is exactly 0.
        tied = [
            self.add_chunk(second_document, vector(2, 0), page_number=2),
            self.add_chunk(first_document, vector(5, 0)),
            self.add_chunk(second_document, vector(1, 0)),
        ]
        closer_ids = sorted(chunk.pk for chunk in tied)
        farther = self.add_chunk(first_document, vector(1, 1))
        for _ in range(3):
            results = self.search()
            self.assertEqual(self.chunk_ids(results), closer_ids + [farther.pk])

    def test_results_span_all_owned_documents(self):
        policy, faq = self.create_document(title="Policy"), self.create_document(title="FAQ")
        best = self.add_chunk(faq, vector(1, 0))
        second = self.add_chunk(policy, vector(1, 0.5))
        third = self.add_chunk(faq, vector(1, 2))
        results = self.search()
        self.assertEqual(self.chunk_ids(results), [best.pk, second.pk, third.pk])
        self.assertEqual([r.document_id for r in results], [faq.pk, policy.pk, faq.pk])


class TopKTests(RetrievalTestCase):
    def setUp(self):
        super().setUp()
        document = self.create_document()
        # Increasingly far from the query, so the expected top K is the first K.
        self.chunks = [self.add_chunk(document, vector(1, index)) for index in range(8)]

    def test_default_limit_is_five(self):
        self.assertEqual(self.chunk_ids(self.search()), [c.pk for c in self.chunks[:5]])

    def test_limit_returns_the_k_closest(self):
        self.assertEqual(self.chunk_ids(self.search(limit=2)), [c.pk for c in self.chunks[:2]])
        self.assertEqual(self.chunk_ids(self.search(limit=1)), [self.chunks[0].pk])

    def test_limit_above_the_number_of_chunks_returns_them_all(self):
        self.assertEqual(self.chunk_ids(self.search(limit=20)), [c.pk for c in self.chunks])

    def test_invalid_limits_are_rejected_without_a_provider_call(self):
        for limit in [0, -1, 21, 10**9, "5", 2.5, True, None]:
            with self.subTest(limit=limit):
                with self.assertRaises(services.InvalidRetrievalRequest):
                    self.search(limit=limit)
        self.assertEqual(self.provider.calls, [])

    @override_settings(RETRIEVAL_MAX_LIMIT=6)
    def test_maximum_comes_from_settings(self):
        self.assertEqual(len(self.search(limit=6)), 6)
        with self.assertRaises(services.InvalidRetrievalRequest):
            self.search(limit=7)

    def test_limit_is_applied_in_the_database(self):
        with CaptureQueriesContext(connection) as queries:
            self.search(limit=3)
        self.assertTrue(queries.captured_queries[-1]["sql"].rstrip().endswith("LIMIT 3"))


class EmbeddingFilterTests(RetrievalTestCase):
    def test_chunks_without_embeddings_are_excluded(self):
        document = self.create_document()
        embedded = self.add_chunk(document, vector(1, 1))
        self.add_chunk(document, None)
        self.assertEqual(self.chunk_ids(self.search()), [embedded.pk])

    def test_documents_without_embeddings_return_nothing_without_a_provider_call(self):
        document = self.create_document()
        self.add_chunk(document, None)
        self.add_chunk(document, None, page_number=2)
        self.assertEqual(self.search(), [])
        self.assertEqual(self.provider.calls, [])

    def test_only_vectors_from_the_configured_model_are_searched(self):
        document = self.create_document()
        current = self.add_chunk(document, vector(1, 1))
        # Closer to the query, but made by another model, so not comparable.
        old = self.add_chunk(document, vector(1, 0), model="fake-model-v0")
        self.assertEqual(self.chunk_ids(self.search()), [current.pk])
        with override_settings(EMBEDDING_MODEL="fake-model-v0"):
            self.assertEqual(self.chunk_ids(self.search()), [old.pk])
        with override_settings(EMBEDDING_MODEL="fake-model-v2"):
            self.assertEqual(self.search(), [])

    def test_zero_length_vectors_are_dropped(self):
        document = self.create_document()
        ranked = self.add_chunk(document, vector(1, 1))
        self.add_chunk(document, vector())  # all zeros: cosine distance is undefined (NaN)
        with self.assertLogs(SERVICES_LOGGER, level="WARNING"):
            self.assertEqual(self.chunk_ids(self.search()), [ranked.pk])


class DocumentStateFilterTests(RetrievalTestCase):
    def test_only_processed_documents_are_searched(self):
        processed = self.create_document()
        kept = self.add_chunk(processed, vector(1, 1))
        other = self.create_document()
        hidden = self.add_chunk(other, vector(1, 0))
        self.assertEqual(self.chunk_ids(self.search()), [hidden.pk, kept.pk])
        for status in [ProcessingStatus.PENDING, ProcessingStatus.PROCESSING, ProcessingStatus.FAILED]:
            with self.subTest(status=status):
                Document.objects.filter(pk=other.pk).update(processing_status=status)
                self.assertEqual(self.chunk_ids(self.search()), [kept.pk])
                self.assertEqual(self.search(document_ids=[other.pk]), [])


class OwnershipTests(RetrievalTestCase):
    def setUp(self):
        super().setUp()
        self.alices_document = self.create_document(owner=self.alice)
        self.bobs_document = self.create_document(owner=self.bob)
        self.alices_chunk = self.add_chunk(self.alices_document, vector(1, 1))
        # Bob's chunk is the closest match of all, so any leak would rank it first.
        self.bobs_chunk = self.add_chunk(self.bobs_document, vector(1, 0))

    def test_each_user_only_retrieves_their_own_chunks(self):
        self.assertEqual(self.chunk_ids(self.search(user=self.alice)), [self.alices_chunk.pk])
        self.assertEqual(self.chunk_ids(self.search(user=self.bob)), [self.bobs_chunk.pk])

    def test_user_without_documents_gets_nothing_without_a_provider_call(self):
        carol = User.objects.create_user("carol", "carol@example.com")
        self.assertEqual(self.search(user=carol), [])
        self.assertEqual(self.provider.calls, [])

    def test_document_ids_narrow_the_search(self):
        second = self.create_document(owner=self.alice)
        in_second = self.add_chunk(second, vector(1, 2))
        self.assertEqual(self.chunk_ids(self.search(document_ids=[second.pk])), [in_second.pk])
        self.assertEqual(
            self.chunk_ids(self.search(document_ids=[second.pk, self.alices_document.pk])),
            [self.alices_chunk.pk, in_second.pk],
        )

    def test_another_users_document_ids_match_nothing(self):
        self.assertEqual(self.search(document_ids=[self.bobs_document.pk]), [])
        self.assertEqual(self.provider.calls, [])  # nothing to search, so no provider request

    def test_document_ids_are_intersected_with_owned_documents(self):
        results = self.search(document_ids=[self.bobs_document.pk, self.alices_document.pk])
        self.assertEqual(self.chunk_ids(results), [self.alices_chunk.pk])

    def test_unknown_document_ids_match_nothing(self):
        self.assertEqual(self.search(document_ids=[self.bobs_document.pk + 1000]), [])

    def test_duplicate_document_ids_are_fine(self):
        ids = [self.alices_document.pk, self.alices_document.pk]
        self.assertEqual(self.chunk_ids(self.search(document_ids=ids)), [self.alices_chunk.pk])

    def test_invalid_document_ids_are_rejected(self):
        too_many = list(range(1, services.MAX_DOCUMENT_IDS + 2))
        for document_ids in [[], "1", 1, [0], [-1], [True], ["1"], [1.5], [None], too_many]:
            with self.subTest(document_ids=document_ids):
                with self.assertRaises(services.InvalidRetrievalRequest):
                    self.search(document_ids=document_ids)
        self.assertEqual(self.provider.calls, [])

    def test_ownership_is_filtered_in_the_database_query(self):
        with CaptureQueriesContext(connection) as queries:
            self.search()
        search_sql = queries.captured_queries[-1]["sql"]
        self.assertIn('"documents_document"."owner_id" = %d' % self.alice.pk, search_sql)
        self.assertIn("<=>", search_sql)


class ResultTests(RetrievalTestCase):
    def test_result_carries_citation_metadata(self):
        document = self.create_document(title="Refund policy")
        chunk = self.add_chunk(document, vector(1, 0), page_number=3, text="Refunds within 30 days.")
        [result] = self.search()
        self.assertEqual(
            dataclasses.asdict(result),
            {
                "chunk_id": chunk.pk,
                "document_id": document.pk,
                "document_title": "Refund policy",
                "original_filename": "refund policy.pdf",
                "page_number": 3,
                "chunk_index": 0,
                "text": "Refunds within 30 days.",
                "start_offset": 0,
                "end_offset": len("Refunds within 30 days."),
                "distance": 0.0,
            },
        )

    def test_vectors_are_neither_returned_nor_selected(self):
        self.add_chunk(self.create_document(), vector(1, 0))
        with CaptureQueriesContext(connection) as queries:
            [result] = self.search()
        self.assertNotIn("embedding", {field.name for field in dataclasses.fields(result)})
        select_list = queries.captured_queries[-1]["sql"].split(" FROM ")[0]
        # The column only appears inside the distance expression, never as a selected value.
        self.assertNotIn('"documents_documentchunk"."embedding",', select_list)
        self.assertEqual(select_list.count('"documents_documentchunk"."embedding"'), 1)

    def test_query_count_does_not_grow_with_documents_or_results(self):
        for _ in range(4):
            document = self.create_document()
            for page_number in (1, 2, 3):
                self.add_chunk(document, vector(1, page_number), page_number=page_number)
        # One existence check, one search query (with document and page joined in).
        with self.assertNumQueries(2):
            results = self.search(limit=12)
        self.assertEqual(len(results), 12)

    def test_no_matching_chunks_returns_an_empty_list(self):
        self.create_document()
        self.assertEqual(self.search(), [])


class Phase6CompatibilityTests(RetrievalTestCase):
    """End to end with the real chunking and embedding services (the fake provider only)."""

    @override_settings(DOCUMENT_CHUNK_SIZE=100, DOCUMENT_CHUNK_OVERLAP=20)
    def test_chunks_embedded_by_the_document_service_are_retrievable(self):
        document = self.create_document()
        DocumentPage.objects.create(document=document, page_number=1, text="Refunds are paid within 30 days.")
        DocumentPage.objects.create(document=document, page_number=2, text="Shipping takes a week.")
        document_services.generate_chunks(document)
        self.provider.vectors = {
            "Refunds are paid within 30 days.": vector(1, 0.1),
            "Shipping takes a week.": vector(0, 1),
        }
        self.assertEqual(document_services.embed_document_chunks(document), 2)

        results = self.search()
        self.assertEqual([(r.page_number, r.text) for r in results], [
            (1, "Refunds are paid within 30 days."),
            (2, "Shipping takes a week."),
        ])
        self.assertEqual(self.provider.calls[-1], [QUERY])  # the query alone, once


class SearchEndpointTests(RetrievalTestCase):
    def setUp(self):
        super().setUp()
        self.client.force_authenticate(self.alice)
        self.document = self.create_document(title="Refund policy")
        self.chunks = [self.add_chunk(self.document, vector(1, index)) for index in range(7)]

    def post(self, data):
        return self.client.post(SEARCH_URL, data, format="json")

    def test_requires_authentication(self):
        self.client.force_authenticate(None)
        response = self.post({"query": QUERY})
        self.assertEqual(response.status_code, 401)
        self.get_provider.assert_not_called()

    def test_only_post_is_allowed(self):
        for method in (self.client.get, self.client.put, self.client.patch, self.client.delete):
            with self.subTest(method=method.__name__):
                self.assertEqual(method(SEARCH_URL).status_code, 405)

    def test_returns_ranked_results_without_vectors(self):
        response = self.post({"query": QUERY, "limit": 2})
        self.assertEqual(response.status_code, 200)
        results = response.json()["results"]
        self.assertEqual([r["chunk_id"] for r in results], [c.pk for c in self.chunks[:2]])
        self.assertEqual(
            set(results[0]),
            {
                "chunk_id", "document_id", "document_title", "original_filename", "page_number",
                "chunk_index", "text", "start_offset", "end_offset", "distance", "similarity",
            },
        )
        self.assertEqual(results[0]["document_title"], "Refund policy")
        self.assertEqual((results[0]["distance"], results[0]["similarity"]), (0.0, 1.0))
        self.assertEqual(self.provider.calls, [[QUERY]])

    def test_default_limit_is_five(self):
        response = self.post({"query": QUERY})
        self.assertEqual([r["chunk_id"] for r in response.json()["results"]], [c.pk for c in self.chunks[:5]])

    def test_invalid_requests_are_rejected_without_a_provider_call(self):
        too_many = list(range(1, services.MAX_DOCUMENT_IDS + 2))
        bodies = {
            "missing query": {},
            "blank query": {"query": "   "},
            "null query": {"query": None},
            "overlong query": {"query": "x" * (services.MAX_QUERY_LENGTH + 1)},
            "zero limit": {"query": QUERY, "limit": 0},
            "negative limit": {"query": QUERY, "limit": -3},
            "limit over max": {"query": QUERY, "limit": 21},
            "fractional limit": {"query": QUERY, "limit": 2.5},
            "text limit": {"query": QUERY, "limit": "five"},
            "boolean limit": {"query": QUERY, "limit": True},
            "null limit": {"query": QUERY, "limit": None},
            "empty document_ids": {"query": QUERY, "document_ids": []},
            "document_ids not a list": {"query": QUERY, "document_ids": self.document.pk},
            "zero document id": {"query": QUERY, "document_ids": [0]},
            "text document id": {"query": QUERY, "document_ids": ["abc"]},
            "too many document_ids": {"query": QUERY, "document_ids": too_many},
        }
        for name, body in bodies.items():
            with self.subTest(name):
                self.assertEqual(self.post(body).status_code, 400)
        self.assertEqual(self.provider.calls, [])

    def test_another_users_document_ids_return_no_results(self):
        bobs_document = self.create_document(owner=self.bob)
        self.add_chunk(bobs_document, vector(1, 0))
        response = self.post({"query": QUERY, "document_ids": [bobs_document.pk]})
        self.assertEqual((response.status_code, response.json()), (200, {"results": []}))

    def test_document_ids_filter_owned_documents(self):
        second = self.create_document()
        in_second = self.add_chunk(second, vector(1, 0))
        bobs_document = self.create_document(owner=self.bob)
        self.add_chunk(bobs_document, vector(1, 0))
        response = self.post({"query": QUERY, "document_ids": [second.pk, bobs_document.pk]})
        self.assertEqual([r["chunk_id"] for r in response.json()["results"]], [in_second.pk])

    def test_client_supplied_user_fields_are_ignored(self):
        bobs_chunk = self.add_chunk(self.create_document(owner=self.bob), vector(1, 0))
        response = self.post({"query": QUERY, "limit": 20, "user": self.bob.pk, "owner": self.bob.pk})
        ids = [r["chunk_id"] for r in response.json()["results"]]
        self.assertEqual(ids, [c.pk for c in self.chunks])
        self.assertNotIn(bobs_chunk.pk, ids)

    def test_no_results_is_an_empty_list(self):
        DocumentChunk.objects.update(embedding=None, embedding_model="", embedded_at=None)
        response = self.post({"query": QUERY})
        self.assertEqual((response.status_code, response.json()), (200, {"results": []}))

    def test_provider_failure_returns_a_safe_503(self):
        self.provider.error = EmbeddingProviderError("upstream said: sk-secret-123")
        with self.assertLogs(VIEWS_LOGGER, level="WARNING") as logs:
            response = self.post({"query": QUERY})
        self.assertEqual((response.status_code, response.json()), (503, {"detail": SEARCH_UNAVAILABLE_ERROR}))
        logged = "\n".join(logs.output)
        self.assertIn("EmbeddingProviderError", logged)
        for secret in ("sk-secret-123", QUERY):
            self.assertNotIn(secret, logged)
            self.assertNotIn(secret, response.content.decode())

    def test_configuration_failure_returns_a_safe_503_and_logs_an_error(self):
        self.get_provider.side_effect = EmbeddingConfigurationError("OPENAI_API_KEY is not set.")
        with self.assertLogs(VIEWS_LOGGER, level="ERROR") as logs:
            response = self.post({"query": QUERY})
        self.assertEqual((response.status_code, response.json()), (503, {"detail": SEARCH_UNAVAILABLE_ERROR}))
        self.assertNotIn("OPENAI_API_KEY", response.content.decode())
        self.assertNotIn(QUERY, "\n".join(logs.output))


class RetrievalSettingsTests(SimpleTestCase):
    def load_settings(self, **env):
        # Runs settings.py again in isolation, with the given environment.
        with mock.patch.dict(os.environ, env):
            return runpy.run_path(str(settings.BASE_DIR / "config" / "settings.py"))

    def test_maximum_comes_from_the_environment(self):
        self.assertEqual(self.load_settings(RETRIEVAL_MAX_LIMIT="50")["RETRIEVAL_MAX_LIMIT"], 50)

    def test_maximum_below_the_default_limit_fails_at_startup(self):
        for value in ["4", "0", "-1"]:
            with self.subTest(value=value):
                with self.assertRaises(ImproperlyConfigured):
                    self.load_settings(RETRIEVAL_MAX_LIMIT=value)
