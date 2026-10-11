import logging

from rest_framework import status
from rest_framework.exceptions import ValidationError
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.embeddings.exceptions import EmbeddingConfigurationError, EmbeddingError

from . import services
from .serializers import RetrievedChunkSerializer, SearchRequestSerializer

logger = logging.getLogger(__name__)

SEARCH_UNAVAILABLE_ERROR = "Search is temporarily unavailable. Please try again later."


class SearchView(APIView):
    """POST a question; get back the user's most relevant document chunks. No answer is generated."""

    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = SearchRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            # The user always comes from the request, never from the body.
            results = services.retrieve_chunks(user=request.user, **serializer.validated_data)
        except services.InvalidRetrievalRequest as exc:
            raise ValidationError({"detail": str(exc)}) from exc
        except EmbeddingError as exc:
            # The query text and provider details are never logged or returned.
            log = logger.error if isinstance(exc, EmbeddingConfigurationError) else logger.warning
            log("Search failed: the query could not be embedded (%s)", type(exc).__name__)
            return Response({"detail": SEARCH_UNAVAILABLE_ERROR}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        return Response({"results": RetrievedChunkSerializer(results, many=True).data})
