from django.contrib import admin
from django.urls import path
from . import views

from django.contrib.auth import views as auth_views

urlpatterns = [
    # Trang chủ & xác thực người dùng
    path('', views.home, name="home"),
    path('index/', views.crypto_price, name="index"),
    path('market-data/', views.market_data, name='market_data'),
    path('fundamental-analysis/', views.fundamental_analysis, name='fundamental_analysis'),
    
]