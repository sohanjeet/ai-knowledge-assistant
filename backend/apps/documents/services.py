import logging

from django.conf import settings
from django.db import transaction
from django.db.models import Q
from django.utils import timezone
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from apps.embeddings import services as embedding_services

from .models import Document, DocumentChunk, DocumentPage, ProcessingStatus

logger = logging.getLogger(__name__)

CHUNK_INSERT_BATCH_SIZE = 500
# Chunks per embedding request. ~1000-character chunks make this roughly 25k tokens per request,
# well under provider limits, and a failure loses at most one batch of work.
EMBEDDING_BATCH_SIZE = 100

# Messages stored in Document.processing_error and returned by the API.
# They are fixed strings on purpose: raw exception text could leak paths or internals.
UNREADABLE_PDF_ERROR = "The PDF could not be read. It may be corrupted."
ENCRYPTED_PDF_ERROR = "The PDF is encrypted or password-protected, which is not supported."
EMPTY_PDF_ERROR = "The PDF contains no pages."
NO_TEXT_PDF_ERROR = (
    "The PDF contains no extractable text. It may be a scanned document, "
    "which is not supported yet."
)
GENERIC_PROCESSING_ERROR = "The document could not be processed."


class ExtractionError(Exception):
    """An expected extraction failure. The message is safe to show to API clients."""


class ChunkingNotAllowedError(Exception):
    """Chunking was requested for a document that is not in the PROCESSED state."""


class EmbeddingNotAllowedError(Exception):
    """Embedding was requested for a document that is not in the PROCESSED state."""


def delete_document(document):
    """Delete a document row and its stored file."""
    storage = document.file.storage
    file_name = document.file.name
    with transaction.atomic():
        document.delete()
    # Remove the file only after the row is gone, so a failed DB delete never leaves
    # a row pointing at a missing file. Worst case is an orphaned file, not a broken row.
    if file_name:
        storage.delete(file_name)


def process_document(document):
    """Extract a document's PDF text into DocumentPage rows, then split it into DocumentChunk rows.

    Safe to call again for the same document: its pages and chunks are replaced, never duplicated.
    Never raises for a bad PDF; failure is recorded on the document instead
    (processing_status=FAILED plus a client-safe processing_error). Returns the document.

    A document is PROCESSED only once it has pages and at least one chunk. FAILED and
    PROCESSING documents hold no pages or chunks (deleting pages cascades to their chunks).
    """
    _save_result(document, ProcessingStatus.PROCESSING)
    try:
        # Slow part: file I/O and PDF parsing. Deliberately outside any DB transaction.
        pages = extract_pages(document)
        _save_result(document, ProcessingStatus.PROCESSED, pages=pages)
        generate_chunks(document)
    except ExtractionError as exc:
        _save_result(document, ProcessingStatus.FAILED, error=str(exc))
    except Exception:
        # Anything else (odd PDFs can trip pypdf in many ways, or the DB write failed).
        # The traceback goes to the server log only, never to the client.
        logger.exception("Unexpected error processing document %s", document.pk)
        _save_result(document, ProcessingStatus.FAILED, error=GENERIC_PROCESSING_ERROR)
    return document


def extract_pages(document):
    """Return [(page_number, text), ...] for the document's PDF. Page numbers start at 1.

    All pypdf usage lives here. Does no database work.
    """
    # Opened through the storage API so this keeps working if files move off local disk.
    with document.file.storage.open(document.file.name, "rb") as pdf_file:
        try:
            reader = PdfReader(pdf_file)
            if reader.is_encrypted:
                raise ExtractionError(ENCRYPTED_PDF_ERROR)
            if len(reader.pages) == 0:
                raise ExtractionError(EMPTY_PDF_ERROR)
            # A page with no extractable text still gets a row, so page numbers stay true.
            pages = [
                (number, clean_text(page.extract_text() or ""))
                for number, page in enumerate(reader.pages, start=1)
            ]
            # Nothing to chunk, search or cite (typically a scanned PDF; there is no OCR).
            if not any(text for _, text in pages):
                raise ExtractionError(NO_TEXT_PDF_ERROR)
            return pages
        except PyPdfError as exc:
            logger.warning("Could not read PDF for document %s: %s", document.pk, exc)
            raise ExtractionError(UNREADABLE_PDF_ERROR) from exc


def clean_text(text):
    """Fix only artifacts that are always safe to fix; the wording is left untouched."""
    # NUL characters are common in broken PDFs and PostgreSQL text columns reject them.
    text = text.replace("\x00", "")
    # Drop lone surrogates (bad font mappings) that cannot be encoded as UTF-8.
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def split_text(text, chunk_size, overlap):
    """Split text into overlapping character ranges. Returns [(start, end), ...].

    Pure function: no database, no settings. Deterministic for the same input.
    - text[start:end] never has leading or trailing whitespace and is never empty.
    - Ranges are in order, each at most chunk_size long, and each starts before the
      previous one ends (that shared stretch is the overlap). Starts and ends both
      strictly increase, so no range sits entirely inside another.
    - Chunks end at whitespace when there is some in the second half of the window,
      and the next chunk starts at a word start, so words are rarely cut. A single
      word longer than the window is cut at chunk_size.
    - Blank text returns [].
    """
    # With overlap >= chunk_size the start could never advance, so refuse instead of looping.
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Need chunk_size > 0 and 0 <= overlap < chunk_size.")

    length = len(text.rstrip())  # trailing whitespace is never part of a chunk
    spans = []
    start = 0
    while True:
        while start < length and text[start].isspace():
            start += 1
        if start >= length:
            break
        if start + chunk_size >= length:
            spans.append((start, length))  # the rest fits in one chunk
            break

        end = _find_chunk_end(text, start, start + chunk_size)
        trimmed_end = end
        while text[trimmed_end - 1].isspace():
            trimmed_end -= 1
        # With a large overlap the window can slide onto the same end; that chunk would be
        # entirely inside the previous one, so it is skipped.
        if not spans or trimmed_end > spans[-1][1]:
            spans.append((start, trimmed_end))

        # The next chunk re-reads the last `overlap` characters, but always moves forward
        # by at least one character, which guarantees the loop finishes.
        start = _align_to_word_start(text, max(end - overlap, start + 1), end)
    return spans


def _find_chunk_end(text, start, hard_end):
    """Pick where a chunk starting at `start` ends: the last whitespace in the window's
    second half, or `hard_end` (a mid-word cut) when there is none."""
    earliest = start + max(1, (hard_end - start) // 2)
    for index in range(hard_end, earliest - 1, -1):
        if text[index].isspace():
            return index
    return hard_end


def _align_to_word_start(text, position, limit):
    """If `position` is inside a word, move forward to the whitespace that ends it
    (the caller then skips the whitespace). Stays put if the word runs to `limit`."""
    if position > 0 and not text[position - 1].isspace() and not text[position].isspace():
        for index in range(position, limit):
            if text[index].isspace():
                return index
    return position


def generate_chunks(document):
    """Split a PROCESSED document's stored pages into DocumentChunk rows. Returns the chunk count.

    Pages are chunked independently, so a chunk never spans two pages. Safe to call again:
    the document's existing chunks are replaced in one short transaction, never duplicated.
    Raises ChunkingNotAllowedError unless the document is PROCESSED.
    """
    chunk_size = settings.DOCUMENT_CHUNK_SIZE
    overlap = settings.DOCUMENT_CHUNK_OVERLAP

    status = Document.objects.filter(pk=document.pk).values_list("processing_status", flat=True).first()
    if status is None:
        return 0  # the document was deleted meanwhile
    if status != ProcessingStatus.PROCESSED:
        raise ChunkingNotAllowedError("Only processed documents can be chunked.")

    # All computation happens before the transaction opens.
    pages = list(document.pages.order_by("page_number"))
    chunks = [
        DocumentChunk(
            document=document,
            page=page,
            chunk_index=index,
            text=page.text[start:end],
            start_offset=start,
            end_offset=end,
        )
        for page in pages
        for index, (start, end) in enumerate(split_text(page.text, chunk_size, overlap))
    ]

    with transaction.atomic():
        # Same row lock as _save_result: overlapping runs are applied one after the other.
        # The checks below run under that lock, so nothing can change before we write.
        if not Document.objects.select_for_update().filter(
            pk=document.pk, processing_status=ProcessingStatus.PROCESSED
        ).exists():
            return 0  # deleted, or re-extraction started: that run will chunk its own pages
        current_page_ids = set(DocumentPage.objects.filter(document=document).values_list("pk", flat=True))
        if current_page_ids != {page.pk for page in pages}:
            # Pages were replaced after we read them; our chunks would point at deleted pages.
            logger.warning("Skipped chunking document %s: its pages changed meanwhile", document.pk)
            return 0
        document.chunks.all().delete()
        DocumentChunk.objects.bulk_create(chunks, batch_size=CHUNK_INSERT_BATCH_SIZE)
    return len(chunks)


def _save_result(document, status, *, error="", pages=()):
    """Store a processing outcome atomically: status, error and the document's page rows.

    Pages are always replaced together with the status change, so a document can never
    be left with half of its pages, and FAILED / PROCESSING documents never hold pages.
    The transaction only covers quick writes, never the PDF parsing.
    """
    with transaction.atomic():
        # Row lock: two overlapping runs on one document are applied one after the other.
        if not Document.objects.select_for_update().filter(pk=document.pk).exists():
            return  # the document was deleted while it was being processed
        document.pages.all().delete()
        if pages:
            DocumentPage.objects.bulk_create(
                DocumentPage(document=document, page_number=number, text=text)
                for number, text in pages
            )
        document.processing_status = status
        document.processing_error = error
        document.save(update_fields=["processing_status", "processing_error", "updated_at"])


def embed_document_chunks(document, *, force=False, batch_size=EMBEDDING_BATCH_SIZE):
    """Store an embedding on each of a PROCESSED document's chunks. Returns how many chunks were embedded.

    An explicit operation: uploads never call it, so the app works without provider credentials.
    Only `document`'s own chunks are read or written; callers are responsible for ownership checks.

    Safe to call again. Chunks that already have an embedding from the configured model are
    skipped (no provider call), so a rerun only embeds chunks that are new (re-chunking creates
    fresh rows without embeddings), were left unembedded by an earlier failure, or were embedded by
    a different model. `force=True` re-embeds every chunk.

    Each batch is a provider call made outside any transaction, then one atomic UPDATE of that
    batch's rows. If a call fails, earlier batches stay saved (each is complete and correct), the
    rest stay unembedded, and the EmbeddingError propagates; calling again resumes from there.

    Raises EmbeddingNotAllowedError unless the document is PROCESSED, and
    apps.embeddings.exceptions.EmbeddingError (or a subclass) if embedding fails.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    status = Document.objects.filter(pk=document.pk).values_list("processing_status", flat=True).first()
    if status is None:
        return 0  # the document was deleted meanwhile
    if status != ProcessingStatus.PROCESSED:
        raise EmbeddingNotAllowedError("Only processed documents can be embedded.")

    model = embedding_services.current_model()
    chunks = DocumentChunk.objects.filter(document=document)
    if not force:
        chunks = chunks.filter(Q(embedding__isnull=True) | ~Q(embedding_model=model))
    # Reading order, so batches (and any partial progress) follow the document. The vector itself
    # is not loaded: only the id is needed to write the new one back.
    chunks = list(chunks.order_by("page__page_number", "chunk_index", "pk").only("pk", "text"))

    # The chunking service never stores blank text, so this is only a guard: such a chunk is
    # left unembedded rather than sent to the provider (which rejects empty input).
    blank = [chunk for chunk in chunks if not chunk.text.strip()]
    if blank:
        logger.warning("Skipped %d blank chunk(s) of document %s", len(blank), document.pk)
        chunks = [chunk for chunk in chunks if chunk.text.strip()]

    embedded = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        # Network call; deliberately no transaction or row lock is held while it runs.
        vectors = embedding_services.embed_texts([chunk.text for chunk in batch])
        embedded_at = timezone.now()
        # embed_texts returns exactly one vector per text, in order, so zip pairs each chunk
        # with its own vector.
        for chunk, vector in zip(batch, vectors, strict=True):
            chunk.embedding = vector
            chunk.embedding_model = model
            chunk.embedded_at = embedded_at
        # One UPDATE keyed by primary key. Rows deleted meanwhile (the document was re-chunked or
        # re-extracted) simply match nothing: ids are never reused and chunk text is never edited
        # in place, so a vector can only ever land on the chunk whose text produced it.
        embedded += DocumentChunk.objects.bulk_update(
            batch, ["embedding", "embedding_model", "embedded_at"], batch_size=batch_size
        )
    return embedded
