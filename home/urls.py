from django.urls import path # type: ignore
from . import views

urlpatterns = [
    path('', views.Home, name="home"),
    path('ask/', views.Ask, name="ask_query"),
    path('answer/', views.answer_question, name="answer_question"),
]