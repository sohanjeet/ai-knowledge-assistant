from django.apps import AppConfig


class RetrievalConfig(AppConfig):
    # No models: this app searches the DocumentChunk embeddings stored by apps.documents.
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.retrieval"
