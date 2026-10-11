from django.apps import AppConfig


class DocumentsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.documents"

    def ready(self):
        # Importing the module registers its system checks.
        from . import checks  # noqa: F401
