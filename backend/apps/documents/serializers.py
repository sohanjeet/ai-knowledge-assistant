import os

from django.conf import settings
from rest_framework import serializers

from .models import Document, DocumentChunk, DocumentPage

PDF_CONTENT_TYPE = "application/pdf"
PDF_MAGIC = b"%PDF-"


class DocumentSerializer(serializers.ModelSerializer):
    # write_only: accepted on upload, never returned (so no storage path/URL leaks).
    file = serializers.FileField(write_only=True)

    class Meta:
        model = Document
        fields = [
            "id",
            "title",
            "file",
            "original_filename",
            "file_size",
            "content_type",
            "processing_status",
            "processing_error",
            "created_at",
            "updated_at",
        ]
        # owner is deliberately not a field: it is set from request.user in the view.
        # Processing fields are set only by the extraction service, never by clients.
        read_only_fields = [
            "id",
            "original_filename",
            "file_size",
            "content_type",
            "processing_status",
            "processing_error",
            "created_at",
            "updated_at",
        ]
        extra_kwargs = {"title": {"required": False}}

    def validate_file(self, uploaded):
        # The extension, the client-sent Content-Type and the file's own bytes are all
        # checked, because the first two are trivially spoofed.
        if os.path.splitext(uploaded.name)[1].lower() != ".pdf":
            raise serializers.ValidationError("Only PDF files are allowed.")
        if uploaded.content_type != PDF_CONTENT_TYPE:
            raise serializers.ValidationError("Only PDF files are allowed.")
        if uploaded.size == 0:
            raise serializers.ValidationError("The uploaded file is empty.")
        if uploaded.size > settings.DOCUMENT_MAX_UPLOAD_SIZE:
            max_mb = settings.DOCUMENT_MAX_UPLOAD_SIZE // (1024 * 1024)
            raise serializers.ValidationError(f"File is too large (maximum is {max_mb} MB).")
        header = uploaded.read(len(PDF_MAGIC))
        uploaded.seek(0)
        if header != PDF_MAGIC:
            raise serializers.ValidationError("File content is not a valid PDF.")
        return uploaded

    def create(self, validated_data):
        uploaded = validated_data["file"]
        validated_data["original_filename"] = os.path.basename(uploaded.name)[:255]
        validated_data["file_size"] = uploaded.size
        validated_data["content_type"] = PDF_CONTENT_TYPE
        if not validated_data.get("title"):
            validated_data["title"] = os.path.splitext(validated_data["original_filename"])[0]
        return super().create(validated_data)


class DocumentPageSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentPage
        fields = ["page_number", "text"]
        read_only_fields = fields


class DocumentChunkSerializer(serializers.ModelSerializer):
    # Supplied by the view's queryset (annotated from the page), so no extra query per chunk.
    page_number = serializers.IntegerField(read_only=True)

    class Meta:
        model = DocumentChunk
        fields = ["id", "page_number", "chunk_index", "text", "start_offset", "end_offset"]
        read_only_fields = fields
