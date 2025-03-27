from django.contrib import admin
from django.urls import path
from . import views

from django.contrib.auth import views as auth_views

urlpatterns = [
    # Trang chủ & xác thực người dùng
    path('', views.home, name="home"),
    path('index/', views.crypto_price, name="index"),
   
    path('arima-predict/', views.arima_predict, name="arima_predict"),
]