import uuid

from django.conf import settings
from django.db import models
from pgvector.django import VectorField


# Size of DocumentChunk.embedding. Fixed in the schema (changing it needs a migration and
# re-embedding); settings.EMBEDDING_DIMENSIONS must match it (see apps/documents/checks.py).
CHUNK_EMBEDDING_DIMENSIONS = 1536


def document_upload_path(instance, filename):
    # Never use the client's filename on disk (path tricks, collisions, odd characters).
    # The original name is kept in Document.original_filename instead.
    return f"documents/{instance.owner_id}/{uuid.uuid4().hex}.pdf"


class ProcessingStatus(models.TextChoices):
    # TextChoices: (stored value, human-readable label). Used as Document.processing_status.
    PENDING = "pending", "Pending"
    PROCESSING = "processing", "Processing"
    PROCESSED = "processed", "Processed"
    FAILED = "failed", "Failed"


class Document(models.Model):
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="documents",
    )
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to=document_upload_path)
    original_filename = models.CharField(max_length=255)
    file_size = models.PositiveBigIntegerField(help_text="Size in bytes.")
    content_type = models.CharField(max_length=100)
    processing_status = models.CharField(
        max_length=20,
        choices=ProcessingStatus.choices,
        default=ProcessingStatus.PENDING,
    )
    # A short, client-safe message only. Never store exception text or stack traces here.
    processing_error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self):
        return self.title


class DocumentPage(models.Model):
    """Text extracted from one page of a Document's PDF. Created only by the extraction service."""

    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,  # deleting a document removes its pages
        related_name="pages",
    )
    page_number = models.PositiveIntegerField(help_text="1-based, as a reader sees it.")
    text = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["page_number"]
        constraints = [
            # Also creates the (document, page_number) index that page lookups need.
            models.UniqueConstraint(
                fields=["document", "page_number"],
                name="unique_page_number_per_document",
            ),
            models.CheckConstraint(
                condition=models.Q(page_number__gte=1),
                name="document_page_number_gte_1",
            ),
        ]

    def __str__(self):
        return f"{self.document_id} p.{self.page_number}"


class DocumentChunk(models.Model):
    """A slice of one DocumentPage's text, sized for later embedding. Created only by the chunking service."""

    # document is derivable from page.document, but is stored too so "all chunks of a document"
    # (retrieval scoping, the API, bulk delete) needs no join. The service keeps the two in step.
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,  # deleting a document removes its chunks
        related_name="chunks",
    )
    page = models.ForeignKey(
        DocumentPage,
        on_delete=models.CASCADE,  # deleting a page removes its chunks (re-extraction replaces pages)
        related_name="chunks",
    )
    chunk_index = models.PositiveIntegerField(help_text="0-based position within the page.")
    text = models.TextField()
    # Character positions in DocumentPage.text: page.text[start_offset:end_offset] == text.
    start_offset = models.PositiveIntegerField()
    end_offset = models.PositiveIntegerField()
    # Set only by the embedding service, all three together. Empty until the chunk is embedded;
    # chunks are rebuilt (never edited) when a document is re-chunked, so new chunks start empty.
    embedding = VectorField(dimensions=CHUNK_EMBEDDING_DIMENSIONS, null=True, blank=True)
    embedding_model = models.CharField(max_length=100, blank=True, default="")
    embedded_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            # Also creates the (page, chunk_index) index that per-page lookups need.
            models.UniqueConstraint(
                fields=["page", "chunk_index"],
                name="unique_chunk_index_per_page",
            ),
            models.CheckConstraint(
                condition=models.Q(end_offset__gt=models.F("start_offset")),
                name="document_chunk_not_empty_range",
            ),
            # A vector is always stored with the model that made it and when, and never without one.
            models.CheckConstraint(
                condition=(
                    models.Q(embedding__isnull=True, embedding_model="", embedded_at__isnull=True)
                    | (
                        models.Q(embedding__isnull=False, embedded_at__isnull=False)
                        & ~models.Q(embedding_model="")
                    )
                ),
                name="document_chunk_embedding_all_or_nothing",
            ),
        ]

    def __str__(self):
        return f"{self.document_id} p.{self.page_id} #{self.chunk_index}"
