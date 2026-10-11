from django.conf import settings
from django.core import checks

from .models import DocumentChunk


@checks.register(checks.Tags.models)
def check_embedding_dimensions(app_configs, **kwargs):
    """Fail `manage.py check` / runserver when the configured embedding size and the column disagree."""
    column_dimensions = DocumentChunk._meta.get_field("embedding").dimensions
    if settings.EMBEDDING_DIMENSIONS != column_dimensions:
        return [
            checks.Error(
                f"EMBEDDING_DIMENSIONS is {settings.EMBEDDING_DIMENSIONS}, but the DocumentChunk.embedding "
                f"column holds {column_dimensions}-dimensional vectors.",
                hint="Set EMBEDDING_DIMENSIONS to match, or change the column with a migration and re-embed.",
                id="documents.E001",
            )
        ]
    return []
