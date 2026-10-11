from django.db.models import F
from django.shortcuts import get_object_or_404
from rest_framework.generics import ListAPIView, ListCreateAPIView, RetrieveDestroyAPIView
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated

from . import services
from .models import Document
from .serializers import DocumentChunkSerializer, DocumentPageSerializer, DocumentSerializer


def owned_documents(request):
    # Scoping the queryset to the owner means another user's document is simply
    # "not found" (404), so its existence is never revealed.
    return Document.objects.filter(owner=request.user)


class OwnedDocumentMixin:
    permission_classes = [IsAuthenticated]
    serializer_class = DocumentSerializer

    def get_queryset(self):
        return owned_documents(self.request)


class DocumentListCreateView(OwnedDocumentMixin, ListCreateAPIView):
    # File uploads need multipart; the global default is JSON only.
    parser_classes = [MultiPartParser, FormParser]

    def perform_create(self, serializer):
        # Owner always comes from the authenticated user, never from request data.
        document = serializer.save(owner=self.request.user)
        # Synchronous extraction. It never raises for a bad PDF: the upload still succeeds
        # (201, the file is stored) and the response shows processing_status "failed"
        # with a safe processing_error. `document` is updated in place, so the response
        # built from the serializer reflects the final status.
        services.process_document(document)


class DocumentDetailView(OwnedDocumentMixin, RetrieveDestroyAPIView):
    def perform_destroy(self, instance):
        services.delete_document(instance)


class DocumentPageListView(ListAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = DocumentPageSerializer

    def get_queryset(self):
        document = get_object_or_404(owned_documents(self.request), pk=self.kwargs["pk"])
        return document.pages.order_by("page_number")


class DocumentChunkListView(ListAPIView):
    # Development/inspection endpoint: shows how a document was split.
    permission_classes = [IsAuthenticated]
    serializer_class = DocumentChunkSerializer

    def get_queryset(self):
        document = get_object_or_404(owned_documents(self.request), pk=self.kwargs["pk"])
        # The vector is not part of the response, so it is not loaded either.
        return (
            document.chunks.defer("embedding")
            .annotate(page_number=F("page__page_number"))
            .order_by("page_number", "chunk_index")
        )
