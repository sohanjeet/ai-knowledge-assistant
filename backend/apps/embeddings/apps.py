from django.apps import AppConfig


class EmbeddingsConfig(AppConfig):
    # No models: this app only wraps the external embedding provider behind embed_texts().
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.embeddings"
