from django.urls import path

from . import views

urlpatterns = [
    path("", views.DocumentListCreateView.as_view(), name="document-list"),
    path("<int:pk>/", views.DocumentDetailView.as_view(), name="document-detail"),
    path("<int:pk>/pages/", views.DocumentPageListView.as_view(), name="document-pages"),
    path("<int:pk>/chunks/", views.DocumentChunkListView.as_view(), name="document-chunks"),
]
