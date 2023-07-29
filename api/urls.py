from django.urls import path
from api import views

app_name = 'api'

urlpatterns = [
    path('singo/', views.singo, name='singo'),
    path('detailNews/', views.detailNews, name='detailNews'),
    path('thumbs/', views.thumbs, name='thumbs'),
    path('getResultUsingByHref/', views.getResultUsingByHref, name='getResultUsingByHref'),
    path('getTitleContentUsingByHref/', views.getTitleContentUsingByHref, name='getTitleContentUsingByHref'),
]