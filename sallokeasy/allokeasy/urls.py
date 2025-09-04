from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("upload-csv/", views.upload_csv, name="upload_csv"),
    path("produce-optmized-portfolio/", views.produce_optimized_portfolio, name="produce_optimized_portfolio"),
]
