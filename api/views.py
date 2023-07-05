from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, Http404
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

import json
import requests

from bs4 import BeautifulSoup

def index(request) :
    return render(request, '')

@csrf_exempt
@require_POST
def getTitleContentUsingByHref(request) :
    
    title = list()
    content = list()

    request_data = json.loads(request.body)
    
    href_list = request_data['href']

    for href in href_list :
        
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
        web_news = requests.get(href, headers=headers).content
        source_news = BeautifulSoup(web_news, 'html.parser')
        
        title_tmp = source_news.find('h2', {'class' : 'media_end_head_headline'}).get_text()
        title.append(title_tmp)
        
        div_element  = source_news.find('div', {'id' : 'dic_area'})
        text_nodes = [node for node in div_element.contents if isinstance(node, str)]
        article = ''.join(text_nodes).strip()

        article = article.replace("\n", "")
        article = article.replace("\t", "")
        article = article.split("※ '당신의 제보가 뉴스가 됩니다'")[0]

        content.append(article)

    
    sebu1_result = sebu1_model(title, content)
    sebu2_result = sebu2_model(title, content)

    result = {}

    result.update({"sebu1_result", sebu1_result})
    result.update({"sebu2_result", sebu2_result})

    return JsonResponse({'response': result})


# 1세부 예측 모델
def sebu1_model(title, content) :
    
    result = 0.12


    return result


# 2세부 예측 모델
def sebu2_model(title, content) :
    

    result = 0.21

    return result
        

