from django.urls import path
from .views import FaceRecognitionAPIView
from . import views


urlpatterns = [
    path('recognize/', FaceRecognitionAPIView.as_view(), name='recognize'),
    path('upload-face/', views.upload_face),
]
