import io
import os
import runpy
import shutil
import tempfile
import zlib
from pathlib import Path
from unittest import mock

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.exceptions import ImproperlyConfigured
from django.core.checks import Error
from django.db import DataError, IntegrityError, connection, transaction
from django.test import SimpleTestCase, override_settings
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils import timezone
from pypdf import PdfWriter
from rest_framework.test import APITestCase

from apps.embeddings.exceptions import EmbeddingConfigurationError, EmbeddingProviderError

from . import services
from .checks import check_embedding_dimensions
from .models import CHUNK_EMBEDDING_DIMENSIONS, Document, DocumentChunk, DocumentPage, ProcessingStatus

User = get_user_model()

# Starts with %PDF- (so it passes upload validation) but is not a readable PDF.
PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"
LIST_URL = reverse("document-list")
SERVICES_LOGGER = "apps.documents.services"


def build_pdf(page_texts):
    """Build a tiny valid text PDF with one page per string (no PDF writing library needed)."""
    page_count = len(page_texts)
    page_ids = [4 + 2 * i for i in range(page_count)]
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: ("<< /Type /Pages /Kids [%s] /Count %d >>" % (" ".join(f"{p} 0 R" for p in page_ids), page_count)).encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for i, text in enumerate(page_texts):
        page_id, content_id = page_ids[i], page_ids[i] + 1
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
        objects[page_id] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode()
        objects[content_id] = b"<< /Length %d >>\nstream\n%b\nendstream" % (len(stream), stream)

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += b"%d 0 obj\n%b\nendobj\n" % (number, objects[number])
    xref_offset = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for number in sorted(objects):
        out += b"%010d 00000 n \n" % offsets[number]
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref_offset)
    return bytes(out)


def build_encrypted_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt("secret", algorithm="RC4-128")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def make_pdf(name="report.pdf", content=PDF_BYTES, content_type="application/pdf"):
    return SimpleUploadedFile(name, content, content_type=content_type)


def detail_url(document_id):
    return reverse("document-detail", args=[document_id])


def pages_url(document_id):
    return reverse("document-pages", args=[document_id])


class DocumentAPITestCase(APITestCase):
    """Base class: uploads go to a throwaway directory that is removed after the tests."""

    def setUp(self):
        # Fresh directory per test (the DB rolls back between tests, files on disk do not).
        self._media_root = tempfile.mkdtemp(prefix="test-media-")
        media_override = override_settings(MEDIA_ROOT=self._media_root)
        media_override.enable()
        self.addCleanup(shutil.rmtree, self._media_root, ignore_errors=True)
        self.addCleanup(media_override.disable)
        self.alice = User.objects.create_user("alice", "alice@example.com")  # no password: tests use force_authenticate
        self.bob = User.objects.create_user("bob", "bob@example.com")
        self.client.force_authenticate(self.alice)

    def upload(self, as_user=None, **data):
        if as_user is not None:
            self.client.force_authenticate(as_user)
        data.setdefault("file", make_pdf())
        return self.client.post(LIST_URL, data, format="multipart")

    def stored_files(self):
        return [p for p in Path(self._media_root).rglob("*") if p.is_file()]


class AuthenticationTests(DocumentAPITestCase):
    def test_unauthenticated_requests_are_rejected(self):
        document_id = self.upload().data["id"]
        self.client.force_authenticate(None)
        responses = [
            self.client.get(LIST_URL),
            self.client.post(LIST_URL, {"file": make_pdf()}, format="multipart"),
            self.client.get(detail_url(document_id)),
            self.client.delete(detail_url(document_id)),
        ]
        for response in responses:
            self.assertEqual(response.status_code, 401)
        self.assertEqual(Document.objects.count(), 1)  # only the original upload exists


class UploadTests(DocumentAPITestCase):
    def test_successful_pdf_upload(self):
        response = self.upload(title="Q3 Report")
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["title"], "Q3 Report")
        self.assertEqual(response.data["original_filename"], "report.pdf")
        self.assertEqual(response.data["file_size"], len(PDF_BYTES))
        self.assertEqual(response.data["content_type"], "application/pdf")
        document = Document.objects.get()
        self.assertEqual(document.owner, self.alice)
        self.assertEqual(len(self.stored_files()), 1)
        self.assertEqual(Path(document.file.path).read_bytes(), PDF_BYTES)

    def test_response_does_not_expose_file_path(self):
        response = self.upload()
        self.assertNotIn("file", response.data)
        self.assertNotIn(self._media_root, str(response.data))
        self.assertNotIn("documents/", str(response.data))

    def test_stored_filename_is_not_the_client_filename(self):
        self.upload(file=make_pdf(name="../../evil name.pdf"))
        document = Document.objects.get()
        self.assertNotIn("evil", document.file.name)
        self.assertTrue(document.file.name.startswith(f"documents/{self.alice.pk}/"))
        self.assertEqual(document.original_filename, "evil name.pdf")

    def test_title_defaults_to_filename_without_extension(self):
        response = self.upload(file=make_pdf(name="Annual Plan.pdf"))
        self.assertEqual(response.data["title"], "Annual Plan")

    def test_missing_file_is_rejected(self):
        response = self.client.post(LIST_URL, {"title": "No file"}, format="multipart")
        self.assertEqual(response.status_code, 400)
        self.assertIn("file", response.data)

    def test_non_pdf_uploads_are_rejected(self):
        cases = {
            "wrong extension": make_pdf(name="notes.txt"),
            "wrong content type": make_pdf(content_type="text/plain"),
            "pdf name and type but not pdf bytes": make_pdf(content=b"just some text"),
            "empty file": make_pdf(content=b""),
        }
        for label, uploaded in cases.items():
            with self.subTest(label):
                response = self.upload(file=uploaded)
                self.assertEqual(response.status_code, 400)
                self.assertIn("file", response.data)
        self.assertEqual(Document.objects.count(), 0)
        self.assertEqual(self.stored_files(), [])

    def test_oversized_upload_is_rejected(self):
        with override_settings(DOCUMENT_MAX_UPLOAD_SIZE=1024):
            response = self.upload(file=make_pdf(content=PDF_BYTES + b"0" * 2048))
        self.assertEqual(response.status_code, 400)
        self.assertIn("file", response.data)
        self.assertEqual(Document.objects.count(), 0)
        self.assertEqual(self.stored_files(), [])

    def test_upload_at_size_limit_is_accepted(self):
        with override_settings(DOCUMENT_MAX_UPLOAD_SIZE=len(PDF_BYTES)):
            response = self.upload()
        self.assertEqual(response.status_code, 201)

    def test_owner_comes_from_request_user_not_request_data(self):
        response = self.upload(owner=self.bob.pk, owner_id=self.bob.pk, user=self.bob.pk)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(Document.objects.get().owner, self.alice)
        self.assertNotIn("owner", response.data)


class ListDetailDeleteTests(DocumentAPITestCase):
    def test_list_returns_own_documents(self):
        first = self.upload(title="First").data["id"]
        second = self.upload(title="Second").data["id"]
        response = self.client.get(LIST_URL)
        self.assertEqual(response.status_code, 200)
        self.assertEqual([d["id"] for d in response.data], [second, first])  # newest first

    def test_list_empty(self):
        response = self.client.get(LIST_URL)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, [])

    def test_detail(self):
        document_id = self.upload(title="Mine").data["id"]
        response = self.client.get(detail_url(document_id))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["id"], document_id)
        self.assertEqual(response.data["title"], "Mine")
        self.assertNotIn("file", response.data)

    def test_delete_removes_row_and_file(self):
        document_id = self.upload().data["id"]
        self.assertEqual(len(self.stored_files()), 1)
        response = self.client.delete(detail_url(document_id))
        self.assertEqual(response.status_code, 204)
        self.assertFalse(Document.objects.filter(pk=document_id).exists())
        self.assertEqual(self.stored_files(), [])

    def test_detail_and_delete_unknown_id_return_404(self):
        self.assertEqual(self.client.get(detail_url(999999)).status_code, 404)
        self.assertEqual(self.client.delete(detail_url(999999)).status_code, 404)

    def test_put_and_patch_not_allowed(self):
        document_id = self.upload().data["id"]
        self.assertEqual(self.client.patch(detail_url(document_id), {"title": "x"}).status_code, 405)
        self.assertEqual(self.client.put(detail_url(document_id), {"title": "x"}).status_code, 405)


class CrossUserAccessTests(DocumentAPITestCase):
    def setUp(self):
        super().setUp()
        self.bobs_document = Document.objects.get(pk=self.upload(as_user=self.bob, title="Bob doc").data["id"])
        self.client.force_authenticate(self.alice)

    def test_user_cannot_list_another_users_documents(self):
        self.upload(title="Alice doc")
        response = self.client.get(LIST_URL)
        self.assertEqual([d["title"] for d in response.data], ["Alice doc"])

    def test_user_cannot_retrieve_another_users_document(self):
        response = self.client.get(detail_url(self.bobs_document.pk))
        self.assertEqual(response.status_code, 404)

    def test_user_cannot_delete_another_users_document(self):
        response = self.client.delete(detail_url(self.bobs_document.pk))
        self.assertEqual(response.status_code, 404)
        self.assertTrue(Document.objects.filter(pk=self.bobs_document.pk).exists())
        self.assertTrue(Path(self.bobs_document.file.path).exists())


class ExtractionTestCase(DocumentAPITestCase):
    """Base for tests that call the extraction service directly on a Document."""

    def create_document(self, content, owner=None):
        return Document.objects.create(
            owner=owner or self.alice,
            title="Test doc",
            file=ContentFile(content, name="test.pdf"),
            original_filename="test.pdf",
            file_size=len(content),
            content_type="application/pdf",
        )

    def replace_file(self, document, content):
        storage = document.file.storage
        name = document.file.name
        storage.delete(name)
        self.assertEqual(storage.save(name, ContentFile(content)), name)

    def page_texts(self, document):
        return list(document.pages.order_by("page_number").values_list("page_number", "text"))


class ExtractionServiceTests(ExtractionTestCase):
    def test_new_document_defaults_to_pending(self):
        document = self.create_document(build_pdf(["Hello"]))
        self.assertEqual(document.processing_status, ProcessingStatus.PENDING)
        self.assertEqual(document.processing_error, "")

    def test_single_page_extraction_succeeds(self):
        document = self.create_document(build_pdf(["Hello extraction"]))
        services.process_document(document)
        document.refresh_from_db()
        self.assertEqual(document.processing_status, ProcessingStatus.PROCESSED)
        self.assertEqual(document.processing_error, "")
        self.assertEqual(self.page_texts(document), [(1, "Hello extraction")])

    def test_multi_page_pdf_preserves_page_numbers_and_text(self):
        document = self.create_document(build_pdf(["Alpha one", "Beta two", "Gamma three"]))
        services.process_document(document)
        self.assertEqual(
            self.page_texts(document),
            [(1, "Alpha one"), (2, "Beta two"), (3, "Gamma three")],
        )

    def test_blank_page_keeps_its_page_number(self):
        document = self.create_document(build_pdf(["First", "", "Third"]))
        services.process_document(document)
        self.assertEqual(self.page_texts(document), [(1, "First"), (2, ""), (3, "Third")])

    def test_corrupt_pdf_marks_document_failed_with_safe_message(self):
        document = self.create_document(PDF_BYTES)
        with self.assertLogs(SERVICES_LOGGER, level="WARNING"):
            services.process_document(document)
        document.refresh_from_db()
        self.assertEqual(document.processing_status, ProcessingStatus.FAILED)
        self.assertEqual(document.processing_error, services.UNREADABLE_PDF_ERROR)
        self.assertEqual(document.pages.count(), 0)
        self.assertNotIn(self._media_root, document.processing_error)

    def test_encrypted_pdf_marks_document_failed(self):
        document = self.create_document(build_encrypted_pdf())
        services.process_document(document)
        document.refresh_from_db()
        self.assertEqual(document.processing_status, ProcessingStatus.FAILED)
        self.assertEqual(document.processing_error, services.ENCRYPTED_PDF_ERROR)

    def test_missing_file_marks_document_failed_without_leaking_details(self):
        document = self.create_document(build_pdf(["Hello"]))
        document.file.storage.delete(document.file.name)
        with self.assertLogs(SERVICES_LOGGER, level="ERROR"):
            services.process_document(document)
        document.refresh_from_db()
        self.assertEqual(document.processing_status, ProcessingStatus.FAILED)
        self.assertEqual(document.processing_error, services.GENERIC_PROCESSING_ERROR)

    def test_failure_midway_leaves_no_partial_pages(self):
        document = self.create_document(build_pdf(["One", "Two", "Three"]))
        original_clean = services.clean_text
        calls = []

        def fail_on_third_page(text):
            calls.append(text)
            if len(calls) == 3:
                raise RuntimeError("boom")
            return original_clean(text)

        with mock.patch.object(services, "clean_text", side_effect=fail_on_third_page):
            with self.assertLogs(SERVICES_LOGGER, level="ERROR"):
                services.process_document(document)
        document.refresh_from_db()
        self.assertEqual(document.processing_status, ProcessingStatus.FAILED)
        self.assertEqual(document.processing_error, services.GENERIC_PROCESSING_ERROR)
        self.assertNotIn("boom", document.processing_error)
        self.assertEqual(document.pages.count(), 0)

    def test_database_failure_while_saving_pages_rolls_back_and_marks_failed(self):
        document = self.create_document(build_pdf(["One", "Two"]))
        with mock.patch.object(DocumentPage.objects, "bulk_create", side_effect=RuntimeError("db down")):
            with self.assertLogs(SERVICES_LOGGER, level="ERROR"):
                services.process_document(document)
        document.refresh_from_db()
        self.assertEqual(document.processing_status, ProcessingStatus.FAILED)
        self.assertEqual(document.pages.count(), 0)

    def test_failed_reprocessing_removes_stale_pages(self):
        document = self.create_document(build_pdf(["One", "Two"]))
        services.process_document(document)
        self.assertEqual(document.pages.count(), 2)
        self.replace_file(document, PDF_BYTES)
        with self.assertLogs(SERVICES_LOGGER, level="WARNING"):
            services.process_document(document)
        document.refresh_from_db()
        self.assertEqual(document.processing_status, ProcessingStatus.FAILED)
        self.assertEqual(document.pages.count(), 0)

    def test_reprocessing_does_not_duplicate_pages(self):
        document = self.create_document(build_pdf(["One", "Two", "Three"]))
        services.process_document(document)
        services.process_document(document)
        services.process_document(document)
        self.assertEqual(
            self.page_texts(document), [(1, "One"), (2, "Two"), (3, "Three")]
        )
        self.assertEqual(DocumentPage.objects.count(), 3)

    def test_reprocessing_replaces_pages_with_current_file_content(self):
        document = self.create_document(build_pdf(["Old one", "Old two", "Old three"]))
        services.process_document(document)
        self.replace_file(document, build_pdf(["New one", "New two"]))
        services.process_document(document)
        self.assertEqual(self.page_texts(document), [(1, "New one"), (2, "New two")])

    def test_reprocessing_recovers_a_failed_document(self):
        document = self.create_document(PDF_BYTES)
        with self.assertLogs(SERVICES_LOGGER, level="WARNING"):
            services.process_document(document)
        self.replace_file(document, build_pdf(["Fixed"]))
        services.process_document(document)
        document.refresh_from_db()
        self.assertEqual(document.processing_status, ProcessingStatus.PROCESSED)
        self.assertEqual(document.processing_error, "")
        self.assertEqual(self.page_texts(document), [(1, "Fixed")])

    def test_reprocessing_one_document_does_not_touch_another(self):
        first = self.create_document(build_pdf(["First doc"]))
        second = self.create_document(build_pdf(["Second doc"]), owner=self.bob)
        services.process_document(first)
        services.process_document(second)
        services.process_document(first)
        self.assertEqual(self.page_texts(second), [(1, "Second doc")])

    def test_processing_a_deleted_document_does_not_raise_or_recreate_rows(self):
        document = self.create_document(build_pdf(["Gone"]))
        Document.objects.filter(pk=document.pk).delete()
        services.process_document(document)
        self.assertEqual(DocumentPage.objects.count(), 0)

    def test_clean_text_removes_only_obvious_artifacts(self):
        self.assertEqual(services.clean_text("  a\x00b\r\nc\rd  \n"), "ab\nc\nd")
        self.assertEqual(services.clean_text("keep   inner  spacing"), "keep   inner  spacing")
        self.assertEqual(services.clean_text("bad \ud800 surrogate"), "bad  surrogate")


class DocumentPageModelTests(ExtractionTestCase):
    def test_duplicate_page_number_for_same_document_is_rejected(self):
        document = self.create_document(PDF_BYTES)
        DocumentPage.objects.create(document=document, page_number=1, text="a")
        with self.assertRaises(IntegrityError), transaction.atomic():
            DocumentPage.objects.create(document=document, page_number=1, text="b")

    def test_same_page_number_is_allowed_in_different_documents(self):
        first = self.create_document(PDF_BYTES)
        second = self.create_document(PDF_BYTES)
        DocumentPage.objects.create(document=first, page_number=1, text="a")
        DocumentPage.objects.create(document=second, page_number=1, text="b")
        self.assertEqual(DocumentPage.objects.count(), 2)

    def test_page_number_zero_is_rejected(self):
        document = self.create_document(PDF_BYTES)
        with self.assertRaises(IntegrityError), transaction.atomic():
            DocumentPage.objects.create(document=document, page_number=0, text="a")

    def test_deleting_document_deletes_its_pages(self):
        document = self.create_document(build_pdf(["One", "Two"]))
        other = self.create_document(build_pdf(["Keep"]))
        services.process_document(document)
        services.process_document(other)
        document.delete()
        self.assertEqual(DocumentPage.objects.filter(document_id=document.pk).count(), 0)
        self.assertEqual(DocumentPage.objects.filter(document=other).count(), 1)


class UploadProcessingTests(DocumentAPITestCase):
    def test_upload_of_valid_pdf_is_processed(self):
        response = self.upload(file=make_pdf(content=build_pdf(["Page A", "Page B"])))
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["processing_status"], "processed")
        self.assertEqual(response.data["processing_error"], "")
        document = Document.objects.get()
        self.assertEqual(document.processing_status, ProcessingStatus.PROCESSED)
        self.assertEqual(document.pages.count(), 2)
        self.assertNotIn("file", response.data)
        self.assertNotIn(self._media_root, str(response.data))

    def test_upload_of_unreadable_pdf_is_stored_but_reported_as_failed(self):
        with self.assertLogs(SERVICES_LOGGER, level="WARNING"):
            response = self.upload()  # PDF_BYTES: right header, unreadable body
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["processing_status"], "failed")
        self.assertEqual(response.data["processing_error"], services.UNREADABLE_PDF_ERROR)
        self.assertNotIn("Traceback", str(response.data))
        self.assertNotIn(self._media_root, str(response.data))
        document = Document.objects.get()
        self.assertEqual(document.processing_status, ProcessingStatus.FAILED)
        self.assertEqual(document.pages.count(), 0)
        self.assertEqual(len(self.stored_files()), 1)

    def test_unexpected_extraction_error_does_not_leak_through_the_api(self):
        with mock.patch.object(services, "extract_pages", side_effect=RuntimeError("secret internals")):
            with self.assertLogs(SERVICES_LOGGER, level="ERROR"):
                response = self.upload(file=make_pdf(content=build_pdf(["Hi"])))
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["processing_status"], "failed")
        self.assertEqual(response.data["processing_error"], services.GENERIC_PROCESSING_ERROR)
        self.assertNotIn("secret internals", str(response.data))

    def test_client_cannot_set_processing_fields(self):
        with self.assertLogs(SERVICES_LOGGER, level="WARNING"):
            response = self.upload(processing_status="processed", processing_error="hacked")
        self.assertEqual(response.data["processing_status"], "failed")
        self.assertNotEqual(response.data["processing_error"], "hacked")

    def test_list_and_detail_expose_processing_status(self):
        document_id = self.upload(file=make_pdf(content=build_pdf(["Hi"]))).data["id"]
        self.assertEqual(self.client.get(LIST_URL).data[0]["processing_status"], "processed")
        detail = self.client.get(detail_url(document_id)).data
        self.assertEqual(detail["processing_status"], "processed")
        self.assertEqual(detail["processing_error"], "")

    def test_deleting_document_via_api_removes_its_pages(self):
        document_id = self.upload(file=make_pdf(content=build_pdf(["One", "Two"]))).data["id"]
        self.assertEqual(DocumentPage.objects.filter(document_id=document_id).count(), 2)
        self.assertEqual(self.client.delete(detail_url(document_id)).status_code, 204)
        self.assertEqual(DocumentPage.objects.count(), 0)
        self.assertEqual(self.stored_files(), [])

    def test_rejected_upload_does_not_attempt_extraction(self):
        with mock.patch.object(services, "process_document") as process:
            response = self.upload(file=make_pdf(content=b"not a pdf"))
        self.assertEqual(response.status_code, 400)
        process.assert_not_called()


class PageEndpointTests(DocumentAPITestCase):
    def upload_pdf(self, page_texts, **kwargs):
        return self.upload(file=make_pdf(content=build_pdf(page_texts)), **kwargs).data["id"]

    def test_requires_authentication(self):
        document_id = self.upload_pdf(["One"])
        self.client.force_authenticate(None)
        self.assertEqual(self.client.get(pages_url(document_id)).status_code, 401)

    def test_owner_gets_pages_in_order(self):
        document_id = self.upload_pdf(["One", "Two", "Three"])
        response = self.client.get(pages_url(document_id))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.data,
            [
                {"page_number": 1, "text": "One"},
                {"page_number": 2, "text": "Two"},
                {"page_number": 3, "text": "Three"},
            ],
        )

    def test_pages_are_ordered_by_page_number_not_insertion_order(self):
        document = Document.objects.get(pk=self.upload_pdf(["x"]))
        document.pages.all().delete()
        for number in (3, 1, 2):
            DocumentPage.objects.create(document=document, page_number=number, text=f"p{number}")
        response = self.client.get(pages_url(document.pk))
        self.assertEqual([p["page_number"] for p in response.data], [1, 2, 3])

    def test_other_users_document_is_not_found(self):
        bobs_id = self.upload_pdf(["Secret"], as_user=self.bob)
        self.client.force_authenticate(self.alice)
        response = self.client.get(pages_url(bobs_id))
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("Secret", str(response.data))

    def test_unknown_document_is_not_found_like_another_users_document(self):
        self.assertEqual(self.client.get(pages_url(999999)).status_code, 404)

    def test_document_without_pages_returns_empty_list(self):
        with self.assertLogs(SERVICES_LOGGER, level="WARNING"):
            document_id = self.upload().data["id"]  # unreadable PDF -> failed, no pages
        response = self.client.get(pages_url(document_id))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, [])

    def test_pages_are_read_only(self):
        document_id = self.upload_pdf(["One"])
        for method in (self.client.post, self.client.put, self.client.patch, self.client.delete):
            with self.subTest(method=method.__name__):
                self.assertEqual(method(pages_url(document_id), {"text": "x"}).status_code, 405)
        self.assertEqual(DocumentPage.objects.get().text, "One")


def chunks_url(document_id):
    return reverse("document-chunks", args=[document_id])


def numbered_words(count):
    """'w000 w001 w002 ...': every word is distinct, so cut words and overlaps are easy to spot."""
    return " ".join(f"w{i:03d}" for i in range(count))


class SplitTextTests(SimpleTestCase):
    def test_text_shorter_than_chunk_size_is_one_chunk(self):
        self.assertEqual(services.split_text("short text", 100, 20), [(0, 10)])

    def test_text_exactly_chunk_size_is_one_chunk(self):
        self.assertEqual(services.split_text("a" * 50, 50, 10), [(0, 50)])

    def test_blank_text_gives_no_chunks(self):
        for text in ("", "   ", "\n\n \t \n"):
            with self.subTest(text=repr(text)):
                self.assertEqual(services.split_text(text, 50, 10), [])

    def test_surrounding_whitespace_is_excluded_from_ranges(self):
        text = "  \n hello world \n  "
        ((start, end),) = services.split_text(text, 100, 10)
        self.assertEqual(text[start:end], "hello world")

    def test_long_text_gives_multiple_chunks_in_order(self):
        spans = services.split_text(numbered_words(100), 100, 20)
        self.assertGreater(len(spans), 3)
        self.assertEqual([s for s, _ in spans], sorted(s for s, _ in spans))

    def test_adjacent_chunks_overlap_and_cover_all_text(self):
        text = numbered_words(200)
        spans = services.split_text(text, 100, 30)
        for (_, previous_end), (next_start, _) in zip(spans, spans[1:]):
            self.assertLess(next_start, previous_end)  # shared text between neighbours
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(text))

    def test_ranges_are_bounded_trimmed_and_never_empty(self):
        text = "The quick  brown fox\n\njumps over   the lazy dog. " * 40
        for size, overlap in [(10, 0), (10, 9), (37, 11), (100, 20), (1, 0)]:
            with self.subTest(size=size, overlap=overlap):
                spans = services.split_text(text, size, overlap)
                for start, end in spans:
                    piece = text[start:end]
                    self.assertTrue(piece)
                    self.assertEqual(piece, piece.strip())
                    self.assertLessEqual(end - start, size)
                # Strictly moving forward (this is what rules out infinite loops).
                self.assertTrue(all(b[0] > a[0] and b[1] > a[1] for a, b in zip(spans, spans[1:])))

    def test_words_are_not_cut_when_whitespace_is_available(self):
        text = numbered_words(150)
        words = set(text.split())
        for start, end in services.split_text(text, 100, 30):
            self.assertTrue(set(text[start:end].split()) <= words)

    def test_unbroken_text_is_cut_at_chunk_size_and_still_overlaps(self):
        text = "x" * 25
        self.assertEqual(
            services.split_text(text, 10, 3),
            [(0, 10), (7, 17), (14, 24), (21, 25)],
        )

    def test_zero_overlap_chunks_do_not_share_text(self):
        text = numbered_words(60)
        spans = services.split_text(text, 50, 0)
        for (_, previous_end), (next_start, _) in zip(spans, spans[1:]):
            self.assertGreaterEqual(next_start, previous_end)

    def test_result_is_deterministic(self):
        text = numbered_words(300)
        self.assertEqual(services.split_text(text, 120, 25), services.split_text(text, 120, 25))

    def test_non_ascii_text_uses_character_offsets(self):
        text = "héllo wörld ñandú " * 20
        for start, end in services.split_text(text, 40, 10):
            self.assertTrue(text[start:end])

    def test_invalid_configuration_is_rejected_instead_of_looping(self):
        for size, overlap in [(0, 0), (-5, 0), (10, -1), (10, 10), (10, 11)]:
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(ValueError):
                    services.split_text("some text", size, overlap)


class ChunkSettingsTests(SimpleTestCase):
    def load_settings(self, **env):
        # Runs settings.py again in isolation, with the given environment.
        with mock.patch.dict(os.environ, env):
            return runpy.run_path(str(settings.BASE_DIR / "config" / "settings.py"))

    def test_defaults_are_valid(self):
        self.assertEqual(settings.DOCUMENT_CHUNK_SIZE, 1000)
        self.assertEqual(settings.DOCUMENT_CHUNK_OVERLAP, 200)

    def test_values_come_from_the_environment(self):
        loaded = self.load_settings(DOCUMENT_CHUNK_SIZE="500", DOCUMENT_CHUNK_OVERLAP="50")
        self.assertEqual((loaded["DOCUMENT_CHUNK_SIZE"], loaded["DOCUMENT_CHUNK_OVERLAP"]), (500, 50))

    def test_invalid_combinations_fail_at_startup(self):
        cases = [("0", "0"), ("-1", "0"), ("100", "-1"), ("100", "100"), ("100", "150")]
        for size, overlap in cases:
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(ImproperlyConfigured):
                    self.load_settings(DOCUMENT_CHUNK_SIZE=size, DOCUMENT_CHUNK_OVERLAP=overlap)


class ChunkTestCase(ExtractionTestCase):
    """Base for tests that call the chunking service on hand-built, already extracted documents."""

    def create_processed_document(self, page_texts, owner=None):
        document = self.create_document(PDF_BYTES, owner=owner)
        document.processing_status = ProcessingStatus.PROCESSED
        document.save()
        for number, text in enumerate(page_texts, start=1):
            DocumentPage.objects.create(document=document, page_number=number, text=text)
        return document

    def chunk_rows(self, document):
        return list(
            document.chunks.order_by("page__page_number", "chunk_index").values_list(
                "page__page_number", "chunk_index", "start_offset", "end_offset", "text"
            )
        )


@override_settings(DOCUMENT_CHUNK_SIZE=100, DOCUMENT_CHUNK_OVERLAP=20)
class ChunkGenerationTests(ChunkTestCase):
    def test_short_page_produces_one_chunk(self):
        document = self.create_processed_document(["A short page."])
        self.assertEqual(services.generate_chunks(document), 1)
        self.assertEqual(self.chunk_rows(document), [(1, 0, 0, 13, "A short page.")])

    def test_long_page_produces_multiple_chunks_with_sequential_indexes(self):
        document = self.create_processed_document([numbered_words(100)])
        count = services.generate_chunks(document)
        rows = self.chunk_rows(document)
        self.assertGreater(count, 3)
        self.assertEqual(len(rows), count)
        self.assertEqual([r[1] for r in rows], list(range(count)))

    def test_adjacent_chunks_overlap(self):
        document = self.create_processed_document([numbered_words(100)])
        services.generate_chunks(document)
        rows = self.chunk_rows(document)
        for previous, following in zip(rows, rows[1:]):
            self.assertLess(following[2], previous[3])  # next start < previous end
            shared = set(previous[4].split()) & set(following[4].split())
            self.assertTrue(shared)

    def test_offsets_match_the_stored_page_text(self):
        pages = [numbered_words(80), "  Second   page,\nwith  odd   spacing. " * 15]
        document = self.create_processed_document(pages)
        services.generate_chunks(document)
        for chunk in document.chunks.select_related("page"):
            self.assertEqual(chunk.text, chunk.page.text[chunk.start_offset : chunk.end_offset])
            self.assertEqual(chunk.text, chunk.text.strip())

    def test_page_boundaries_are_never_crossed(self):
        # Every page ends mid-sentence; a merging chunker would join these.
        pages = [f"page{n} " + numbered_words(30) + f" end{n}" for n in (1, 2, 3)]
        document = self.create_processed_document(pages)
        services.generate_chunks(document)
        for chunk in document.chunks.select_related("page"):
            marker = f"page{chunk.page.page_number}"
            other_markers = {f"page{n}" for n in (1, 2, 3)} - {marker}
            self.assertFalse(any(m in chunk.text for m in other_markers))
            self.assertIn(chunk.text, chunk.page.text)

    def test_provenance_points_to_the_right_page_and_document(self):
        document = self.create_processed_document(["First page text.", "Second page text."])
        other = self.create_processed_document(["Other doc text."], owner=self.bob)
        services.generate_chunks(document)
        services.generate_chunks(other)
        for chunk in document.chunks.select_related("page"):
            self.assertEqual(chunk.document_id, document.pk)
            self.assertEqual(chunk.page.document_id, document.pk)
        self.assertEqual(
            [(r[0], r[4]) for r in self.chunk_rows(document)],
            [(1, "First page text."), (2, "Second page text.")],
        )
        self.assertEqual(other.chunks.count(), 1)

    def test_chunks_are_deterministic(self):
        document = self.create_processed_document([numbered_words(120), numbered_words(50)])
        services.generate_chunks(document)
        first = self.chunk_rows(document)
        services.generate_chunks(document)
        self.assertEqual(self.chunk_rows(document), first)

    def test_pages_are_chunked_in_page_order_whatever_the_insertion_order(self):
        document = self.create_document(PDF_BYTES)
        document.processing_status = ProcessingStatus.PROCESSED
        document.save()
        for number in (3, 1, 2):
            DocumentPage.objects.create(document=document, page_number=number, text=f"text {number}")
        services.generate_chunks(document)
        self.assertEqual([(r[0], r[1]) for r in self.chunk_rows(document)], [(1, 0), (2, 0), (3, 0)])

    def test_blank_pages_create_no_chunks_but_neighbours_keep_theirs(self):
        document = self.create_processed_document(["First", "", "  \n\t ", "Fourth"])
        self.assertEqual(services.generate_chunks(document), 2)
        self.assertEqual([r[0] for r in self.chunk_rows(document)], [1, 4])
        self.assertFalse(document.chunks.filter(text="").exists())

    def test_document_with_only_blank_pages_gets_zero_chunks(self):
        document = self.create_processed_document(["", "   "])
        self.assertEqual(services.generate_chunks(document), 0)
        self.assertEqual(DocumentChunk.objects.count(), 0)

    def test_document_without_pages_gets_zero_chunks(self):
        document = self.create_processed_document([])
        self.assertEqual(services.generate_chunks(document), 0)

    def test_rerunning_does_not_duplicate_chunks(self):
        document = self.create_processed_document([numbered_words(100), "Page two."])
        first_count = services.generate_chunks(document)
        services.generate_chunks(document)
        services.generate_chunks(document)
        self.assertEqual(DocumentChunk.objects.count(), first_count)
        self.assertEqual(document.chunks.count(), first_count)

    def test_rerunning_replaces_old_chunks_with_current_content(self):
        document = self.create_processed_document([numbered_words(100)])
        services.generate_chunks(document)
        document.pages.update(text="Brand new text.")
        self.assertEqual(services.generate_chunks(document), 1)
        self.assertEqual(self.chunk_rows(document), [(1, 0, 0, 15, "Brand new text.")])

    def test_rerunning_with_new_settings_replaces_chunks(self):
        document = self.create_processed_document([numbered_words(100)])
        services.generate_chunks(document)
        small = document.chunks.count()
        with override_settings(DOCUMENT_CHUNK_SIZE=300, DOCUMENT_CHUNK_OVERLAP=50):
            services.generate_chunks(document)
        self.assertLess(document.chunks.count(), small)

    def test_rerunning_one_document_does_not_touch_another(self):
        first = self.create_processed_document([numbered_words(50)])
        second = self.create_processed_document([numbered_words(50)], owner=self.bob)
        services.generate_chunks(first)
        services.generate_chunks(second)
        before = self.chunk_rows(second)
        services.generate_chunks(first)
        self.assertEqual(self.chunk_rows(second), before)

    def test_only_processed_documents_can_be_chunked(self):
        for status in (ProcessingStatus.PENDING, ProcessingStatus.PROCESSING, ProcessingStatus.FAILED):
            with self.subTest(status=status):
                document = self.create_processed_document(["Some text"])
                Document.objects.filter(pk=document.pk).update(processing_status=status)
                with self.assertRaises(services.ChunkingNotAllowedError):
                    services.generate_chunks(document)
                self.assertEqual(document.chunks.count(), 0)

    def test_stale_in_memory_status_is_not_trusted(self):
        document = self.create_processed_document(["Some text"])
        Document.objects.filter(pk=document.pk).update(processing_status=ProcessingStatus.FAILED)
        self.assertEqual(document.processing_status, ProcessingStatus.PROCESSED)  # object is stale
        with self.assertRaises(services.ChunkingNotAllowedError):
            services.generate_chunks(document)

    def test_deleted_document_is_ignored(self):
        document = self.create_processed_document(["Some text"])
        Document.objects.filter(pk=document.pk).delete()
        self.assertEqual(services.generate_chunks(document), 0)
        self.assertEqual(DocumentChunk.objects.count(), 0)

    def test_chunks_are_not_written_if_pages_were_replaced_during_computation(self):
        document = self.create_processed_document(["Some text to chunk"])
        original_split = services.split_text

        def split_then_replace_pages(*args):
            spans = original_split(*args)
            document.pages.all().delete()  # a concurrent re-extraction replacing the pages
            DocumentPage.objects.create(document=document, page_number=1, text="Newer text")
            return spans

        with mock.patch.object(services, "split_text", side_effect=split_then_replace_pages):
            with self.assertLogs(SERVICES_LOGGER, level="WARNING"):
                self.assertEqual(services.generate_chunks(document), 0)
        self.assertEqual(DocumentChunk.objects.count(), 0)

    def test_failed_write_keeps_the_previous_chunks(self):
        document = self.create_processed_document([numbered_words(100)])
        services.generate_chunks(document)
        before = self.chunk_rows(document)
        with mock.patch.object(DocumentChunk.objects, "bulk_create", side_effect=RuntimeError("db down")):
            with self.assertRaises(RuntimeError):
                services.generate_chunks(document)
        self.assertEqual(self.chunk_rows(document), before)  # the delete was rolled back too

    def test_many_pages_are_chunked_with_a_constant_number_of_queries(self):
        document = self.create_processed_document([numbered_words(60)] * 150)
        with CaptureQueriesContext(connection) as queries:
            count = services.generate_chunks(document)
        self.assertEqual(document.chunks.count(), count)
        self.assertGreater(count, 150)
        # status, pages, savepoint, lock, page ids, delete, inserts (500 rows per batch), release.
        self.assertLessEqual(len(queries), 8 + count // 500)


class DocumentChunkModelTests(ChunkTestCase):
    def setUp(self):
        super().setUp()
        self.document = self.create_processed_document(["Page one", "Page two"])
        self.page = self.document.pages.get(page_number=1)

    def make_chunk(self, page=None, **overrides):
        values = dict(
            document=self.document, page=page or self.page, chunk_index=0,
            text="Page one", start_offset=0, end_offset=8,
        )
        values.update(overrides)
        return DocumentChunk.objects.create(**values)

    def test_duplicate_chunk_index_for_same_page_is_rejected(self):
        self.make_chunk()
        with self.assertRaises(IntegrityError), transaction.atomic():
            self.make_chunk()

    def test_same_chunk_index_is_allowed_on_different_pages(self):
        self.make_chunk()
        self.make_chunk(page=self.document.pages.get(page_number=2), text="Page two")
        self.assertEqual(DocumentChunk.objects.count(), 2)

    def test_empty_or_inverted_offset_range_is_rejected(self):
        for start, end in [(5, 5), (8, 3)]:
            with self.subTest(start=start, end=end):
                with self.assertRaises(IntegrityError), transaction.atomic():
                    self.make_chunk(start_offset=start, end_offset=end)

    def test_negative_values_are_rejected(self):
        with self.assertRaises(IntegrityError), transaction.atomic():
            self.make_chunk(start_offset=0, end_offset=8, chunk_index=-1)

    def test_deleting_document_deletes_its_chunks_only(self):
        other = self.create_processed_document(["Keep me"], owner=self.bob)
        services.generate_chunks(self.document)
        services.generate_chunks(other)
        self.document.delete()
        self.assertEqual(DocumentChunk.objects.filter(document_id=self.document.pk).count(), 0)
        self.assertEqual(other.chunks.count(), 1)

    def test_deleting_a_page_deletes_its_chunks_only(self):
        services.generate_chunks(self.document)
        self.assertEqual(self.document.chunks.count(), 2)
        self.page.delete()
        self.assertEqual([c.page.page_number for c in self.document.chunks.select_related("page")], [2])


class ChunkPipelineTests(ExtractionTestCase):
    def chunk_texts(self, document):
        return list(document.chunks.order_by("page__page_number", "chunk_index").values_list("text", flat=True))

    def post_pdf(self, page_texts):
        return self.client.post(LIST_URL, {"file": make_pdf(content=build_pdf(page_texts))}, format="multipart")

    def test_processing_extracts_pages_then_chunks_them(self):
        document = self.create_document(build_pdf(["Alpha text", "Beta text"]))
        services.process_document(document)
        document.refresh_from_db()
        self.assertEqual(document.processing_status, ProcessingStatus.PROCESSED)
        self.assertEqual(self.chunk_texts(document), ["Alpha text", "Beta text"])

    def test_long_pdf_page_is_split_into_chunks_within_the_size_limit(self):
        document = self.create_document(build_pdf([numbered_words(400)]))
        services.process_document(document)
        texts = self.chunk_texts(document)
        self.assertGreater(len(texts), 2)
        self.assertTrue(all(0 < len(t) <= settings.DOCUMENT_CHUNK_SIZE for t in texts))

    def test_pdf_with_no_text_fails_with_clear_message_and_no_rows(self):
        document = self.create_document(build_pdf(["", ""]))
        services.process_document(document)
        document.refresh_from_db()
        self.assertEqual(document.processing_status, ProcessingStatus.FAILED)
        self.assertEqual(document.processing_error, services.NO_TEXT_PDF_ERROR)
        self.assertEqual(document.pages.count(), 0)
        self.assertEqual(DocumentChunk.objects.count(), 0)

    def test_whitespace_only_pdf_is_treated_as_having_no_text(self):
        document = self.create_document(build_pdf(["     ", "   "]))
        services.process_document(document)
        document.refresh_from_db()
        self.assertEqual(document.processing_status, ProcessingStatus.FAILED)
        self.assertEqual(document.processing_error, services.NO_TEXT_PDF_ERROR)
        self.assertEqual(DocumentChunk.objects.count(), 0)

    def test_pdf_with_some_blank_pages_is_processed_without_empty_chunks(self):
        document = self.create_document(build_pdf(["Text", "", "More text"]))
        services.process_document(document)
        document.refresh_from_db()
        self.assertEqual(document.processing_status, ProcessingStatus.PROCESSED)
        self.assertEqual(document.pages.count(), 3)
        self.assertEqual(self.chunk_texts(document), ["Text", "More text"])

    def test_reprocessing_does_not_duplicate_chunks(self):
        document = self.create_document(build_pdf([numbered_words(300), "Two"]))
        services.process_document(document)
        count = DocumentChunk.objects.count()
        services.process_document(document)
        services.process_document(document)
        self.assertEqual(DocumentChunk.objects.count(), count)

    def test_reprocessing_replaces_chunks_with_current_file_content(self):
        document = self.create_document(build_pdf(["Old content"]))
        services.process_document(document)
        self.replace_file(document, build_pdf(["New content"]))
        services.process_document(document)
        self.assertEqual(self.chunk_texts(document), ["New content"])

    def test_failed_reprocessing_removes_stale_chunks(self):
        document = self.create_document(build_pdf(["Old content"]))
        services.process_document(document)
        self.assertEqual(DocumentChunk.objects.count(), 1)
        self.replace_file(document, PDF_BYTES)
        with self.assertLogs(SERVICES_LOGGER, level="WARNING"):
            services.process_document(document)
        document.refresh_from_db()
        self.assertEqual(document.processing_status, ProcessingStatus.FAILED)
        self.assertEqual(DocumentChunk.objects.count(), 0)

    def test_textless_reprocessing_removes_stale_chunks(self):
        document = self.create_document(build_pdf(["Old content"]))
        services.process_document(document)
        self.replace_file(document, build_pdf([""]))
        services.process_document(document)
        document.refresh_from_db()
        self.assertEqual(document.processing_status, ProcessingStatus.FAILED)
        self.assertEqual((document.pages.count(), DocumentChunk.objects.count()), (0, 0))

    def test_chunking_failure_marks_document_failed_and_leaves_no_pages_or_chunks(self):
        document = self.create_document(build_pdf(["Some content"]))
        with mock.patch.object(services, "split_text", side_effect=RuntimeError("boom")):
            with self.assertLogs(SERVICES_LOGGER, level="ERROR"):
                services.process_document(document)
        document.refresh_from_db()
        self.assertEqual(document.processing_status, ProcessingStatus.FAILED)
        self.assertEqual(document.processing_error, services.GENERIC_PROCESSING_ERROR)
        self.assertNotIn("boom", document.processing_error)
        self.assertEqual((document.pages.count(), DocumentChunk.objects.count()), (0, 0))

    def test_chunking_failure_on_reprocessing_does_not_leave_stale_chunks(self):
        document = self.create_document(build_pdf(["Some content"]))
        services.process_document(document)
        with mock.patch.object(services, "split_text", side_effect=RuntimeError("boom")):
            with self.assertLogs(SERVICES_LOGGER, level="ERROR"):
                services.process_document(document)
        self.assertEqual(DocumentChunk.objects.count(), 0)

    def test_upload_creates_chunks(self):
        response = self.post_pdf(["Uploaded text"])
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["processing_status"], "processed")
        self.assertEqual(self.chunk_texts(Document.objects.get()), ["Uploaded text"])

    def test_upload_of_textless_pdf_is_stored_but_reported_as_failed(self):
        response = self.post_pdf([""])
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["processing_status"], "failed")
        self.assertEqual(response.data["processing_error"], services.NO_TEXT_PDF_ERROR)
        self.assertEqual(len(self.stored_files()), 1)
        self.assertEqual(DocumentChunk.objects.count(), 0)

    def test_deleting_document_via_api_removes_its_chunks(self):
        document_id = self.post_pdf(["One", "Two"]).data["id"]
        self.assertEqual(DocumentChunk.objects.count(), 2)
        self.assertEqual(self.client.delete(detail_url(document_id)).status_code, 204)
        self.assertEqual(DocumentChunk.objects.count(), 0)


class ChunkEndpointTests(DocumentAPITestCase):
    def upload_pdf(self, page_texts, **kwargs):
        return self.upload(file=make_pdf(content=build_pdf(page_texts)), **kwargs).data["id"]

    def test_requires_authentication(self):
        document_id = self.upload_pdf(["One"])
        self.client.force_authenticate(None)
        self.assertEqual(self.client.get(chunks_url(document_id)).status_code, 401)

    def test_owner_gets_chunks_with_debug_metadata(self):
        document_id = self.upload_pdf(["First page", "Second page"])
        response = self.client.get(chunks_url(document_id))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [{k: v for k, v in c.items() if k != "id"} for c in response.data],
            [
                {"page_number": 1, "chunk_index": 0, "text": "First page", "start_offset": 0, "end_offset": 10},
                {"page_number": 2, "chunk_index": 0, "text": "Second page", "start_offset": 0, "end_offset": 11},
            ],
        )
        self.assertEqual(
            set(response.data[0]),
            {"id", "page_number", "chunk_index", "text", "start_offset", "end_offset"},
        )
        self.assertEqual(response.data[0]["id"], DocumentChunk.objects.get(page__page_number=1).pk)

    def test_response_does_not_expose_filesystem_information(self):
        document_id = self.upload_pdf(["One"])
        body = str(self.client.get(chunks_url(document_id)).data)
        self.assertNotIn(self._media_root, body)
        self.assertNotIn("documents/", body)

    def test_chunks_are_ordered_by_page_number_then_chunk_index(self):
        document = Document.objects.get(pk=self.upload_pdf(["x"]))
        document.pages.all().delete()
        # Pages created in reverse, so page ids do not follow page-number order.
        pages = {n: DocumentPage.objects.create(document=document, page_number=n, text="abcdef") for n in (2, 1)}
        for number, index in [(2, 1), (1, 1), (2, 0), (1, 0)]:
            DocumentChunk.objects.create(
                document=document, page=pages[number], chunk_index=index,
                text="abc", start_offset=0, end_offset=3,
            )
        response = self.client.get(chunks_url(document.pk))
        self.assertEqual(
            [(c["page_number"], c["chunk_index"]) for c in response.data],
            [(1, 0), (1, 1), (2, 0), (2, 1)],
        )

    @override_settings(DOCUMENT_CHUNK_SIZE=100, DOCUMENT_CHUNK_OVERLAP=20)
    def test_long_document_chunks_match_page_text_through_the_api(self):
        document_id = self.upload_pdf([numbered_words(100)])
        page_text = self.client.get(pages_url(document_id)).data[0]["text"]
        chunks = self.client.get(chunks_url(document_id)).data
        self.assertGreater(len(chunks), 3)
        for chunk in chunks:
            self.assertEqual(chunk["text"], page_text[chunk["start_offset"] : chunk["end_offset"]])

    def test_queries_do_not_grow_with_number_of_chunks(self):
        document = Document.objects.get(pk=self.upload_pdf(["x"]))
        document.pages.all().delete()
        page = DocumentPage.objects.create(document=document, page_number=1, text="a" * 50)
        DocumentChunk.objects.bulk_create(
            DocumentChunk(document=document, page=page, chunk_index=i, text="a", start_offset=i, end_offset=i + 1)
            for i in range(30)
        )
        with self.assertNumQueries(2):  # document lookup, chunks (page number is joined in)
            self.assertEqual(len(self.client.get(chunks_url(document.pk)).data), 30)

    def test_other_users_document_is_not_found(self):
        bobs_id = self.upload_pdf(["Secret text"], as_user=self.bob)
        self.client.force_authenticate(self.alice)
        response = self.client.get(chunks_url(bobs_id))
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("Secret", str(response.data))

    def test_unknown_document_is_not_found_like_another_users_document(self):
        self.assertEqual(self.client.get(chunks_url(999999)).status_code, 404)

    def test_failed_document_returns_empty_list(self):
        with self.assertLogs(SERVICES_LOGGER, level="WARNING"):
            document_id = self.upload().data["id"]
        response = self.client.get(chunks_url(document_id))
        self.assertEqual((response.status_code, response.data), (200, []))

    def test_chunks_are_read_only(self):
        document_id = self.upload_pdf(["One"])
        for method in (self.client.post, self.client.put, self.client.patch, self.client.delete):
            with self.subTest(method=method.__name__):
                self.assertEqual(method(chunks_url(document_id), {"text": "x"}).status_code, 405)
        self.assertEqual(DocumentChunk.objects.get().text, "One")


# ---------------------------------------------------------------------------
# Embeddings. Every test uses FakeEmbeddingProvider: no API key, no network calls.
# The vectors themselves are real pgvector values stored in the PostgreSQL test database.
# ---------------------------------------------------------------------------

EMBEDDING_PROVIDER_PATH = "apps.embeddings.services.get_provider"


def fake_vector(text):
    """A full-size vector that identifies its text. Whole numbers below 2**24 and 0.5 are stored
    exactly by pgvector (float32), so stored vectors can be compared with ==."""
    return [float(zlib.crc32(text.encode()) % (1 << 24)), float(len(text))] + [0.5] * (CHUNK_EMBEDDING_DIMENSIONS - 2)


class FakeEmbeddingProvider:
    """Records each request. Optionally fails on the Nth request, or runs `on_call` first."""

    model = "fake-provider-model"

    def __init__(self, fail_on_call=None, on_call=None):
        self.calls = []
        self.fail_on_call = fail_on_call
        self.on_call = on_call

    def embed(self, texts):
        self.calls.append(list(texts))
        if self.on_call:
            self.on_call(texts)
        if len(self.calls) == self.fail_on_call:
            raise EmbeddingProviderError("The embedding provider request failed.")
        return [fake_vector(text) for text in texts]


# An empty key means any accidental use of the real provider fails fast instead of calling out.
@override_settings(
    EMBEDDING_MODEL="fake-model-v1", OPENAI_API_KEY="", DOCUMENT_CHUNK_SIZE=100, DOCUMENT_CHUNK_OVERLAP=20
)
class EmbeddingTestCase(ChunkTestCase):
    def setUp(self):
        super().setUp()
        self.provider = FakeEmbeddingProvider()
        patcher = mock.patch(EMBEDDING_PROVIDER_PATH, side_effect=lambda: self.provider)
        self.get_provider = patcher.start()
        self.addCleanup(patcher.stop)

    def create_chunked_document(self, page_texts, owner=None):
        document = self.create_processed_document(page_texts, owner=owner)
        services.generate_chunks(document)
        return document

    def ordered_chunks(self, document):
        return list(document.chunks.order_by("page__page_number", "chunk_index"))

    def assert_not_embedded(self, chunk):
        self.assertEqual((chunk.embedding, chunk.embedding_model, chunk.embedded_at), (None, "", None))

    def assert_embedded(self, chunk, model="fake-model-v1"):
        self.assertEqual(list(chunk.embedding), fake_vector(chunk.text))
        self.assertEqual(chunk.embedding_model, model)
        self.assertIsNotNone(chunk.embedded_at)


class EmbedDocumentChunksTests(EmbeddingTestCase):
    def test_vectors_model_and_time_are_stored_on_the_matching_chunks(self):
        document = self.create_chunked_document(["Alpha page.", "Beta page, a bit longer."])
        before = timezone.now()
        self.assertEqual(services.embed_document_chunks(document), 2)
        chunks = self.ordered_chunks(document)
        for chunk in chunks:
            self.assert_embedded(chunk)
            self.assertGreaterEqual(chunk.embedded_at, before)
        self.assertNotEqual(list(chunks[0].embedding), list(chunks[1].embedding))

    def test_new_chunks_start_without_embeddings(self):
        document = self.create_chunked_document(["Some text."])
        for chunk in document.chunks.all():
            self.assert_not_embedded(chunk)

    def test_chunks_are_sent_together_in_reading_order(self):
        document = self.create_chunked_document([numbered_words(30), "Second page.", "Third page."])
        services.embed_document_chunks(document)
        texts = [chunk.text for chunk in self.ordered_chunks(document)]
        self.assertGreater(len(texts), 3)
        self.assertEqual(self.provider.calls, [texts])  # one batched request, not one per chunk

    def test_multiple_batches_map_each_vector_to_its_own_chunk(self):
        document = self.create_chunked_document([numbered_words(60), "Second page.", "Third page."])
        chunks = self.ordered_chunks(document)
        self.assertGreaterEqual(len(chunks), 5)
        self.assertEqual(services.embed_document_chunks(document, batch_size=2), len(chunks))
        self.assertTrue(all(len(call) == 2 for call in self.provider.calls[:-1]))
        self.assertEqual(sum(self.provider.calls, []), [chunk.text for chunk in chunks])
        for chunk in self.ordered_chunks(document):
            self.assert_embedded(chunk)

    def test_rerun_skips_chunks_already_embedded_with_the_current_model(self):
        document = self.create_chunked_document(["One.", "Two."])
        services.embed_document_chunks(document)
        stored = {c.pk: (list(c.embedding), c.embedded_at) for c in document.chunks.all()}
        self.assertEqual(services.embed_document_chunks(document), 0)
        self.assertEqual(len(self.provider.calls), 1)  # no second provider request
        self.assertEqual({c.pk: (list(c.embedding), c.embedded_at) for c in document.chunks.all()}, stored)

    def test_only_missing_embeddings_are_generated(self):
        document = self.create_chunked_document(["One.", "Two.", "Three."])
        services.embed_document_chunks(document)
        missing = self.ordered_chunks(document)[1]
        DocumentChunk.objects.filter(pk=missing.pk).update(embedding=None, embedding_model="", embedded_at=None)
        self.assertEqual(services.embed_document_chunks(document), 1)
        self.assertEqual(self.provider.calls[-1], ["Two."])
        for chunk in self.ordered_chunks(document):
            self.assert_embedded(chunk)

    def test_model_change_reembeds_chunks_from_the_old_model(self):
        document = self.create_chunked_document(["One.", "Two."])
        services.embed_document_chunks(document)
        with override_settings(EMBEDDING_MODEL="fake-model-v2"):
            self.assertEqual(services.embed_document_chunks(document), 2)
            self.assertEqual(services.embed_document_chunks(document), 0)  # now current again
        self.assertEqual(len(self.provider.calls), 2)
        for chunk in document.chunks.all():
            self.assert_embedded(chunk, model="fake-model-v2")

    def test_force_reembeds_every_chunk(self):
        document = self.create_chunked_document(["One.", "Two."])
        services.embed_document_chunks(document)
        self.assertEqual(services.embed_document_chunks(document, force=True), 2)
        self.assertEqual(self.provider.calls, [["One.", "Two."], ["One.", "Two."]])
        for chunk in document.chunks.all():
            self.assert_embedded(chunk)

    def test_provider_failure_keeps_finished_batches_and_a_rerun_resumes(self):
        document = self.create_chunked_document(["One.", "Two.", "Three.", "Four.", "Five."])
        self.provider.fail_on_call = 2
        with self.assertRaises(EmbeddingProviderError):
            services.embed_document_chunks(document, batch_size=2)
        chunks = self.ordered_chunks(document)
        for chunk in chunks[:2]:
            self.assert_embedded(chunk)  # first batch saved, complete and correct
        for chunk in chunks[2:]:
            self.assert_not_embedded(chunk)  # nothing half-written

        self.provider = FakeEmbeddingProvider()
        self.assertEqual(services.embed_document_chunks(document, batch_size=2), 3)
        self.assertEqual(self.provider.calls, [["Three.", "Four."], ["Five."]])
        for chunk in self.ordered_chunks(document):
            self.assert_embedded(chunk)

    def test_malformed_provider_result_saves_nothing(self):
        document = self.create_chunked_document(["One.", "Two."])
        self.provider.embed = lambda texts: [fake_vector(texts[0])]  # one vector for two texts
        with self.assertRaises(EmbeddingProviderError):
            services.embed_document_chunks(document)
        for chunk in document.chunks.all():
            self.assert_not_embedded(chunk)

    def test_wrong_dimension_from_provider_saves_nothing(self):
        document = self.create_chunked_document(["One."])
        self.provider.embed = lambda texts: [[0.5] * (CHUNK_EMBEDDING_DIMENSIONS - 1) for _ in texts]
        with self.assertRaises(EmbeddingProviderError):
            services.embed_document_chunks(document)
        self.assert_not_embedded(document.chunks.get())

    def test_missing_credentials_fail_clearly_and_save_nothing(self):
        self.get_provider.side_effect = EmbeddingConfigurationError("OPENAI_API_KEY is not set.")
        document = self.create_chunked_document(["One."])
        with self.assertRaises(EmbeddingConfigurationError):
            services.embed_document_chunks(document)
        self.assert_not_embedded(document.chunks.get())

    def test_regenerated_chunks_start_without_embeddings_and_can_be_embedded(self):
        document = self.create_chunked_document(["One.", "Two."])
        services.embed_document_chunks(document)
        services.generate_chunks(document)
        for chunk in document.chunks.all():
            self.assert_not_embedded(chunk)
        self.assertEqual(services.embed_document_chunks(document), 2)

    def test_chunks_replaced_during_the_provider_call_get_no_vectors(self):
        document = self.create_chunked_document(["One.", "Two."])
        # Simulates a concurrent re-chunk while the request is in flight: the chunk rows the
        # vectors were made for are gone, and the new rows must not receive them.
        self.provider.on_call = lambda texts: services.generate_chunks(document)
        self.assertEqual(services.embed_document_chunks(document), 0)
        for chunk in document.chunks.all():
            self.assert_not_embedded(chunk)

    def test_no_transaction_is_open_during_the_provider_call(self):
        document = self.create_chunked_document(["One.", "Two."])
        depth_outside = len(connection.atomic_blocks)  # the test case's own transaction
        depths = []
        self.provider.on_call = lambda texts: depths.append(len(connection.atomic_blocks))
        services.embed_document_chunks(document, batch_size=1)
        self.assertEqual(depths, [depth_outside, depth_outside])

    def test_blank_chunk_text_is_skipped_defensively(self):
        document = self.create_processed_document(["Real text here."])
        page = document.pages.get()
        DocumentChunk.objects.create(document=document, page=page, chunk_index=0, text="Real", start_offset=0, end_offset=4)
        blank = DocumentChunk.objects.create(
            document=document, page=page, chunk_index=1, text="   ", start_offset=4, end_offset=7
        )
        with self.assertLogs(SERVICES_LOGGER, level="WARNING"):
            self.assertEqual(services.embed_document_chunks(document), 1)
        self.assertEqual(self.provider.calls, [["Real"]])
        blank.refresh_from_db()
        self.assert_not_embedded(blank)

    def test_document_without_chunks_makes_no_provider_call(self):
        document = self.create_processed_document(["Not chunked yet."])
        self.assertEqual(services.embed_document_chunks(document), 0)
        self.assertEqual(self.provider.calls, [])

    def test_unprocessed_documents_are_rejected(self):
        document = self.create_chunked_document(["One."])
        for status in [ProcessingStatus.PENDING, ProcessingStatus.PROCESSING, ProcessingStatus.FAILED]:
            with self.subTest(status=status):
                Document.objects.filter(pk=document.pk).update(processing_status=status)
                with self.assertRaises(services.EmbeddingNotAllowedError):
                    services.embed_document_chunks(document)
        self.assertEqual(self.provider.calls, [])
        self.assert_not_embedded(document.chunks.get())

    def test_deleted_document_returns_zero(self):
        document = self.create_chunked_document(["One."])
        Document.objects.filter(pk=document.pk).delete()
        self.assertEqual(services.embed_document_chunks(document), 0)
        self.assertEqual(self.provider.calls, [])

    def test_invalid_batch_size_is_rejected(self):
        document = self.create_chunked_document(["One."])
        with self.assertRaises(ValueError):
            services.embed_document_chunks(document, batch_size=0)

    def test_only_the_given_documents_chunks_are_embedded(self):
        alices = self.create_chunked_document(["Alice text."])
        bobs = self.create_chunked_document(["Bob text."], owner=self.bob)
        services.embed_document_chunks(alices)
        self.assertEqual(self.provider.calls, [["Alice text."]])
        self.assert_embedded(alices.chunks.get())
        self.assert_not_embedded(bobs.chunks.get())

    def test_database_queries_do_not_grow_with_the_number_of_chunks(self):
        document = self.create_chunked_document([numbered_words(300)])
        self.assertGreater(document.chunks.count(), 10)
        # Status check, chunk select, one UPDATE. (bulk_update's savepoint-free atomic adds no query.)
        with self.assertNumQueries(3):
            services.embed_document_chunks(document)


class UploadDoesNotEmbedTests(EmbeddingTestCase):
    def test_upload_extracts_and_chunks_without_embedding_or_credentials(self):
        response = self.upload(file=make_pdf(content=build_pdf(["Uploaded text."])))
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["processing_status"], ProcessingStatus.PROCESSED)
        self.get_provider.assert_not_called()
        self.assert_not_embedded(DocumentChunk.objects.get(document_id=response.data["id"]))

    def test_chunk_endpoint_does_not_expose_embeddings(self):
        document_id = self.upload(file=make_pdf(content=build_pdf(["Uploaded text."]))).data["id"]
        services.embed_document_chunks(Document.objects.get(pk=document_id))
        data = self.client.get(chunks_url(document_id)).data
        self.assertEqual(set(data[0]), {"id", "page_number", "chunk_index", "text", "start_offset", "end_offset"})


class VectorStorageTests(EmbeddingTestCase):
    """Runs against the real pgvector column in the PostgreSQL test database (no fakes)."""

    def setUp(self):
        super().setUp()
        self.document = self.create_chunked_document(["Stored text."])
        self.chunk = self.document.chunks.get()

    def test_vector_extension_is_installed_in_the_test_database(self):
        with connection.cursor() as cursor:
            cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            row = cursor.fetchone()
        self.assertIsNotNone(row)

    def test_embedding_column_has_the_configured_dimension(self):
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = %s::regclass AND attname = 'embedding'",
                [DocumentChunk._meta.db_table],
            )
            self.assertEqual(cursor.fetchone()[0], f"vector({CHUNK_EMBEDDING_DIMENSIONS})")
        self.assertEqual(settings.EMBEDDING_DIMENSIONS, CHUNK_EMBEDDING_DIMENSIONS)

    def test_vector_round_trips_through_the_database(self):
        vector = [0.25, -1.5] + [0.125] * (CHUNK_EMBEDDING_DIMENSIONS - 2)
        DocumentChunk.objects.filter(pk=self.chunk.pk).update(
            embedding=vector, embedding_model="m", embedded_at=timezone.now()
        )
        self.chunk.refresh_from_db()
        self.assertEqual(list(self.chunk.embedding), vector)

    def test_database_rejects_a_vector_of_the_wrong_dimension(self):
        with self.assertRaises(DataError), transaction.atomic():
            DocumentChunk.objects.filter(pk=self.chunk.pk).update(
                embedding=[0.5, 0.5, 0.5], embedding_model="m", embedded_at=timezone.now()
            )

    def test_embedding_fields_are_all_set_or_all_empty(self):
        vector = fake_vector("x")
        now = timezone.now()
        invalid = [
            {"embedding": vector},
            {"embedding": vector, "embedding_model": "m"},
            {"embedding": vector, "embedded_at": now},
            {"embedding_model": "m"},
            {"embedded_at": now},
            {"embedding_model": "m", "embedded_at": now},
        ]
        for values in invalid:
            with self.subTest(fields=sorted(values)):
                with self.assertRaises(IntegrityError), transaction.atomic():
                    DocumentChunk.objects.filter(pk=self.chunk.pk).update(**values)


class EmbeddingDimensionCheckTests(SimpleTestCase):
    def test_matching_dimensions_pass(self):
        with override_settings(EMBEDDING_DIMENSIONS=CHUNK_EMBEDDING_DIMENSIONS):
            self.assertEqual(check_embedding_dimensions(None), [])

    def test_mismatched_dimensions_fail_the_system_check(self):
        with override_settings(EMBEDDING_DIMENSIONS=768):
            errors = check_embedding_dimensions(None)
        self.assertEqual([e.id for e in errors], ["documents.E001"])
        self.assertIsInstance(errors[0], Error)
