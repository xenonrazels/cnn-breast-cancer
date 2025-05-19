from django.urls import path
from . import views

app_name = 'predictor'
urlpatterns = [
    path('', views.upload_image, name='upload_image'),
]