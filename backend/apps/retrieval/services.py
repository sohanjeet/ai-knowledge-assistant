"""Semantic retrieval: find the stored chunks closest in meaning to a question.

The question is embedded with embed_texts() (never a provider SDK), and PostgreSQL/pgvector does
the rest in one query: ownership, document-state and model filtering, cosine-distance ordering
and LIMIT. No vector is loaded into Python.

Search is exact (a sequential scan over the user's own embedded chunks), with no ANN index.
Every query is narrowed to one owner's documents first, so the rows scanned are that user's
chunks, not the whole table; exact search returns the true top K, while an HNSW/IVFFlat index
would trade recall for speed and, combined with these filters, can return fewer than K rows.
Revisit with an index (and pgvector's iterative scans) only once per-user chunk counts make
the scan measurably slow.
"""

import logging
import math
from dataclasses import dataclass

from django.conf import settings
from pgvector.django import CosineDistance

from apps.documents.models import DocumentChunk, ProcessingStatus
from apps.embeddings import services as embedding_services

logger = logging.getLogger(__name__)

# Generous for a question, and keeps the provider request small.
MAX_QUERY_LENGTH = 2000
# Bounds the SQL IN (...) list a single search can send.
MAX_DOCUMENT_IDS = 100


class InvalidRetrievalRequest(ValueError):
    """The query, limit or document_ids are invalid. The message is safe to show to API clients."""


@dataclass(frozen=True)
class RetrievedChunk:
    """One search result: a chunk's text plus what a later answer needs to cite it. Never the vector."""

    chunk_id: int
    document_id: int
    document_title: str
    original_filename: str
    page_number: int
    chunk_index: int
    text: str
    start_offset: int
    end_offset: int
    # Cosine distance from the query: 0 is the same direction, 1 unrelated, 2 opposite. Lower is closer.
    distance: float

    @property
    def similarity(self):
        """Cosine similarity, 1 - distance: higher is closer."""
        return 1.0 - self.distance


def retrieve_chunks(*, user, query, limit=settings.RETRIEVAL_DEFAULT_LIMIT, document_ids=None):
    """Return up to `limit` of the user's chunks closest to `query`, closest first.

    Only chunks that belong to documents owned by `user`, whose document is PROCESSED, and that
    were embedded with the currently configured model are searched. `document_ids` narrows the
    search further; ids the user does not own (or that do not exist) simply match nothing, so
    they can never widen the search or reveal another user's documents.

    The query is embedded once, and only when there is at least one chunk to search, so a user
    with nothing embedded gets [] without a provider request.

    Raises InvalidRetrievalRequest for an invalid query, limit or document_ids, and
    apps.embeddings.exceptions.EmbeddingError (or a subclass) if the query cannot be embedded.
    """
    if not getattr(user, "is_authenticated", False):
        raise ValueError("retrieve_chunks needs an authenticated user.")  # a caller bug, not client input
    query = _validated_query(query)
    limit = _validated_limit(limit)
    if document_ids is not None:
        document_ids = _validated_document_ids(document_ids)

    # Every filter runs in the database, before any distance is computed or any row is returned.
    chunks = DocumentChunk.objects.filter(
        document__owner=user,
        document__processing_status=ProcessingStatus.PROCESSED,
        embedding__isnull=False,
        # Vectors from an earlier model stay stored until re-embedded; they are not comparable.
        embedding_model=embedding_services.current_model(),
    )
    if document_ids is not None:
        chunks = chunks.filter(document_id__in=document_ids)

    if not chunks.exists():
        return []

    query_vector = embedding_services.embed_texts([query])[0]

    rows = (
        chunks.annotate(distance=CosineDistance("embedding", query_vector))
        # pk breaks ties, so equal distances always come back in the same order.
        .order_by("distance", "pk")
        # Only these columns are selected (joined in the same query, so no per-row lookups).
        .values_list(
            "pk",
            "document_id",
            "document__title",
            "document__original_filename",
            "page__page_number",
            "chunk_index",
            "text",
            "start_offset",
            "end_offset",
            "distance",
        )[:limit]
    )
    results = [RetrievedChunk(*row) for row in rows]

    # A zero-length vector has no direction, so its cosine distance is NaN. That never happens with
    # real embeddings, but such a row could not be ranked (or sent as JSON), so it is dropped.
    finite = [result for result in results if math.isfinite(result.distance)]
    if len(finite) != len(results):
        logger.warning("Dropped %d search result(s) with an undefined distance", len(results) - len(finite))
    return finite


def _validated_query(query):
    if not isinstance(query, str) or not query.strip():
        raise InvalidRetrievalRequest("The query must not be blank.")
    query = query.strip()
    if len(query) > MAX_QUERY_LENGTH:
        raise InvalidRetrievalRequest(f"The query must be at most {MAX_QUERY_LENGTH} characters.")
    return query


def _validated_limit(limit):
    # bool is an int in Python, but True/False are never a valid limit.
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise InvalidRetrievalRequest("The limit must be a whole number.")
    if not 1 <= limit <= settings.RETRIEVAL_MAX_LIMIT:
        raise InvalidRetrievalRequest(f"The limit must be between 1 and {settings.RETRIEVAL_MAX_LIMIT}.")
    return limit


def _validated_document_ids(document_ids):
    if isinstance(document_ids, (str, bytes)) or not isinstance(document_ids, (list, tuple, set, frozenset)):
        raise InvalidRetrievalRequest("document_ids must be a list of document ids.")
    if not document_ids:
        # An empty list would search nothing; leave the field out to search every document.
        raise InvalidRetrievalRequest("document_ids must not be empty.")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in document_ids):
        raise InvalidRetrievalRequest("document_ids must contain positive whole numbers.")
    unique_ids = sorted(set(document_ids))
    if len(unique_ids) > MAX_DOCUMENT_IDS:
        raise InvalidRetrievalRequest(f"At most {MAX_DOCUMENT_IDS} document ids can be searched at once.")
    return unique_ids
