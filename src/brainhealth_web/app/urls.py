from django.urls import path
from .views import home, signup, chat
from django.contrib.auth import views as auth_views

urlpatterns = [
    path("", home, name="home"),
    path("chat", chat, name="chat"),
    path("signup", signup, name="signup"),
    path("login", auth_views.LoginView.as_view(template_name='login.html'), name="login"),
    path('logout', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
]
