from django.conf import settings
from rest_framework import serializers

from .services import MAX_DOCUMENT_IDS, MAX_QUERY_LENGTH


class SearchRequestSerializer(serializers.Serializer):
    """Validates a search request body. The service re-checks the same rules for non-API callers."""

    # Surrounding whitespace is trimmed, so a whitespace-only query is rejected as blank.
    query = serializers.CharField(max_length=MAX_QUERY_LENGTH)
    limit = serializers.IntegerField(required=False, default=settings.RETRIEVAL_DEFAULT_LIMIT)
    # Optional: leave it out to search all of the user's documents. Ids are only a filter;
    # the service always restricts the search to the user's own documents.
    document_ids = serializers.ListField(
        child=serializers.IntegerField(min_value=1),
        required=False,
        allow_empty=False,
        max_length=MAX_DOCUMENT_IDS,
    )

    def validate_limit(self, value):
        # Checked here rather than with max_value=..., so the setting is read per request.
        if not 1 <= value <= settings.RETRIEVAL_MAX_LIMIT:
            raise serializers.ValidationError(f"Must be between 1 and {settings.RETRIEVAL_MAX_LIMIT}.")
        return value


class RetrievedChunkSerializer(serializers.Serializer):
    """Read-only shape of one search result (a services.RetrievedChunk). There is no vector field."""

    chunk_id = serializers.IntegerField()
    document_id = serializers.IntegerField()
    document_title = serializers.CharField()
    original_filename = serializers.CharField()
    page_number = serializers.IntegerField()
    chunk_index = serializers.IntegerField()
    text = serializers.CharField()
    start_offset = serializers.IntegerField()
    end_offset = serializers.IntegerField()
    distance = serializers.FloatField()
    similarity = serializers.FloatField()
