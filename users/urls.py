from django.urls import path # type: ignore
from . import views

urlpatterns = [
    path('user-login', views.loginUser, name="login"),
    path('user-logout', views.logoutUser, name="logout"),
    path('create-account', views.registerUser, name="register"),
]