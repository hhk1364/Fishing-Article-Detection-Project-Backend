from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, Http404
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.utils.decorators import method_decorator

import django
django.setup()

from api.models import TbFishingDataM
from api.models import TbSingoDataM

from django.core.files import File
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification, BertTokenizerFast
import re
import os

import json
import requests

from bs4 import BeautifulSoup
from datetime import datetime
import numpy as np
import time


base_dir =  settings.BASE_DIR
device=torch.device('cpu')

# 1세부 모델 초기화
HUGGINGFACE_MODEL_PATH = "klue/roberta-small"
tokenizer1 = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH)
model_file_path1 =  base_dir + "/api/fishingmodel/1model.pth"
checkpoint = torch.load(model_file_path1, map_location=device)
model1 = RobertaForSequenceClassification.from_pretrained(HUGGINGFACE_MODEL_PATH, num_labels=2)
model1.load_state_dict(checkpoint['model_state_dict'])

model1.eval()


# 2세부 모델 초기화
model2 = AutoModelForSequenceClassification.from_pretrained("klue/roberta-small", num_labels=2)
tokenizer2 = AutoTokenizer.from_pretrained("klue/roberta-small")
model_file_path2 =  base_dir + "/api/fishingmodel/2model.pt"
model2.load_state_dict(torch.load(model_file_path2, map_location=device))

model2.eval()

# 광고 모델 초기화
model3 = AutoModelForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
model_file_path3 =  base_dir + "/api/fishingmodel/gwango_model.bin"
model3.load_state_dict(torch.load(model_file_path3, map_location=device))
tokenizer3 = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

model3.eval()


def index(request):
 return HttpResponse('<h1>Hello, world!</h1>')


# 마우스 png파일
# def mouseImage(request):
    
#     image_path = os.path.join('/media', '/mouse_cursor.png')
    
#     # 파일 읽기
#     with open(image_path, 'rb') as image_file:
#         image_data = File(image_file)
    
#     # FileResponse 객체 생성
#     response = FileResponse(image_data, content_type='image/png')
#     response['Content-Disposition'] = 'inline; filename="image.png"'
    
#     return response

def tenminute():
    while True:

        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}

        web_news = requests.get("https://news.naver.com/", headers=headers).content
        source_news = BeautifulSoup(web_news, 'html.parser')

        # href_list = [a['href'] for a in source_news.find_all('a')]
        href_list = ['https://n.news.naver.com/article/088/0000826588?cds=news_media_pc&type=breakingnews', 'https://n.news.naver.com/article/448/0000419539?cds=news_media_pc&type=breakingnews', 'https://n.news.naver.com/article/296/0000067955?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/448/0000419506?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/082/0001223367?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/088/0000826585?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/657/0000016846?cds=news_media_pc&type=editn', 'https://n.news.naver.com/mnews/hotissue/article/015/0004871071?cid=2000053&type=series&cds=news_media_pc', 'https://n.news.naver.com/mnews/hotissue/article/079/0003793875?cid=1089896&type=series&cds=news_media_pc', 'https://n.news.naver.com/mnews/hotissue/article/016/0002173093?cid=2000320&type=series&cds=news_media_pc', 'https://n.news.naver.com/article/022/0003835915?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/022/0003835910?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/022/0003836073?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/022/0003835709?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/088/0000826368?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/088/0000826372?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/088/0000826373?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/657/0000016820?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/657/0000016784?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/657/0000016747?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/657/0000016734?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/119/0002732947?sid=100&type=journalists&cds=news_media_pc', 'https://n.news.naver.com/article/025/0003295523?sid=100&type=journalists&cds=news_media_pc', 'https://n.news.naver.com/article/469/0000751123?sid=110&type=journalists&cds=news_media_pc', 'https://n.news.naver.com/article/092/0002299482?sid=105&type=journalists&cds=news_media_pc']

        title = list()
        content = list()
        link = list()
        
        print(len(href_list))

        try:

            for href in href_list:
                if "https://n.news.naver.com" in href:
                    try:
                        
                        web_news = requests.get(href, headers=headers).content
                        source_news = BeautifulSoup(web_news, 'html.parser')

                        # 예외처리, 시간재기
                        title_tmp = source_news.find('h2', {'class' : 'media_end_head_headline'}).get_text()

                        div_element  = source_news.find('div', {'id' : 'dic_area'})

                        # div 자식 노드들의 텍스트를 순서대로 article 변수에 담기
                        article = ""

                        for node in div_element.children:
                            # 텍스트 노드인 경우
                            if isinstance(node, str):  
                                article += node.strip()
                            elif node.name == 'span' and 'end_photo_org' not in node.get('class', []):  # span 요소인 경우(class가 "end_photo_org"가 아닌 경우)
                                article += node.get_text(strip=True)
                            # # span 요소인 경우
                            # elif node.name == 'span':  
                            #     article += node.get_text(strip=True)


                        article = article.replace("\n", "")
                        article = article.replace("\t", "")
                        article = article.split("※ '당신의 제보가 뉴스가 됩니다'")[0]
                        article = article.split("▷ 자세한 뉴스 곧 이어집니다.")[0]

                        if count_korean_words(article) >= 0.9 :  
                            if(len(article) > 100) :
                                link.append(href) 
                                title.append(title_tmp)
                                content.append(article)

                    except Exception as e:
                        print(f"예외가 발생했습니다: {e}")
                        pass
            
            data = {
                'link' : link,
                'title': title,
                'content': content
            }

            print(len(link))

            df = pd.DataFrame(data)
            
            now = datetime.now()
            print("1세부 시작", now.strftime('%Y-%m-%d %H:%M:%S'))

            sebu1_result = sebu1_model(df)
            
            now = datetime.now()
            print("1세부 끝 2세부 시작", now.strftime('%Y-%m-%d %H:%M:%S'))

            sebu2_result, suspicious_sentences_result = sebu2_model(df)
            
            now = datetime.now()
            print("2세부 끝 광고 시작", now.strftime('%Y-%m-%d %H:%M:%S'))

            gwango_result = gwango_model(df)

            now = datetime.now()
            print("광고 끝", now.strftime('%Y-%m-%d %H:%M:%S'))

            for i in range(len(link)) :
                try : 
                    item = TbFishingDataM(link=link[i], 
                                    title=title[i],
                                    content=content[i],
                                    number_1sebu=str(sebu1_result[i]),
                                    number_2sebu=str(sebu2_result[i]),                                   
                                    suspicious_sentences=str(suspicious_sentences_result[i]),
                                    gwango=str(gwango_result[i]),
                                    gwango_like=0,
                                    gwango_hate=0,
                                    number_1sebu_like=0,
                                    number_1sebu_hate=0,
                                    number_2sebu_like=0,
                                    number_2sebu_hate=0)
                    item.save()
                except :
                    pass
        except :
                    pass
        
        time.sleep(600)  # 600초 = 10분


# 커스텀 직렬화 함수를 사용하여 set을 직렬화하는 예시
def set_serializer(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError('Object of type set is not JSON serializable')

    

@csrf_exempt
@require_POST
def getResultUsingByHref(request) :

    link = list()
    title = list()
    content = list()
    sebu1_result = list()
    sebu2_result = list()
    gwango_result = list()
    suspicious_sentences = list()

    request_data = json.loads(request.body)
    
    href_list = request_data['href']

    for href in href_list :
            try:
                # 데이터베이스에서 해당 링크에 해당하는 데이터를 조회합니다.
                result = TbFishingDataM.objects.get(link=href)
                link.append(href)
                sebu1_result.append(str(result.number_1sebu))
                sebu2_result.append(str(result.number_2sebu))
                gwango_result.append(str(result.gwango))
                suspicious_sentences.append(result.suspicious_sentences)
            except TbFishingDataM.DoesNotExist:
                link.append(href)
                sebu1_result.append("결과 없음")
                sebu2_result.append("결과 없음")
                gwango_result.append("결과 없음")
                suspicious_sentences.append("")

    data = {
        'link' : link,
        "sebu1_result" : sebu1_result, 
        "sebu2_result" : sebu2_result,
        "gwango_result" : gwango_result,
        "suspicious_sentences" : suspicious_sentences
    }

    return JsonResponse(data)






@csrf_exempt
@require_POST
def detailNews(request) :
    
    request_data = json.loads(request.body)
    
    href = request_data['href']
 
    data = {"sucess" : "sucess"}

    link = list()
    title = list()
    content = list()
    sebu1_result = list()
    sebu2_result = list()
    gwango_result = list()
    suspicious_sentences = list()


    try:
        # 데이터베이스에서 해당 링크에 해당하는 데이터를 조회합니다.
        result = TbFishingDataM.objects.get(link=href)
        link.append(href)
        sebu1_result.append(str(result.number_1sebu))
        sebu2_result.append(str(result.number_2sebu))
        gwango_result.append(str(result.gwango))
        suspicious_sentences.append(result.suspicious_sentences)
    except TbFishingDataM.DoesNotExist:
        link.append(href)
        sebu1_result.append("결과 없음")
        sebu2_result.append("결과 없음")
        gwango_result.append("결과 없음")
        suspicious_sentences.append("")

    data = {
        'link' : link,
        "sebu1_result" : sebu1_result, 
        "sebu2_result" : sebu2_result,
        "gwango_result" : gwango_result,
        "suspicious_sentences" : suspicious_sentences
    }
     
    return JsonResponse(data)

@csrf_exempt
@require_POST
def thumbs(request) :
    
    request_data = json.loads(request.body)
    
    href = request_data['link']

    data = {"sucess" : "sucess"}

    if(len(href) > 0) :
        items = TbFishingDataM.objects.get(link=href)
        if(items) :
            if 'gwango_like' in request_data :
                if(request_data['gwango_like'] == '1') :
                    items.gwango_like = str(int(items.gwango_like) + 1)
                else :
                    items.gwango_hate = str(int(items.gwango_hate) + 1)

            if '1sebu_like' in request_data :            
                if(request_data['1sebu_like'] == '1') :
                    items.number_1sebu_like = str(int(items.number_1sebu_like) + 1)
                else :
                    items.number_1sebu_hate = str(int(items.number_1sebu_hate) + 1)
            
            if '2sebu_like' in request_data :            
                if(request_data['2sebu_like'] == '1') :
                    items.number_2sebu_like = str(int(items.number_2sebu_like) + 1)
                else :
                    items.number_2sebu_hate = str(int(items.number_2sebu_hate) + 1)
        items.save()

    return JsonResponse(data)


@csrf_exempt
@require_POST
def singo(request) :

    global base_dir

    title = list()
    content = list()
    link = list()

    request_data = json.loads(request.body)
    
    href = request_data['href']
    
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}

    web_news = requests.get(href, headers=headers).content
    source_news = BeautifulSoup(web_news, 'html.parser')

    try:
        # 예외처리, 시간재기
        title_tmp = source_news.find('h2', {'class' : 'media_end_head_headline'}).get_text()

        div_element  = source_news.find('div', {'id' : 'dic_area'})

        # div 자식 노드들의 텍스트를 순서대로 article 변수에 담기
        article = ""

        for node in div_element.children:
            # 텍스트 노드인 경우
            if isinstance(node, str):  
                article += node.strip()
            elif node.name == 'span' and 'end_photo_org' not in node.get('class', []):  # span 요소인 경우(class가 "end_photo_org"가 아닌 경우)
                article += node.get_text(strip=True)
            # # span 요소인 경우
            # elif node.name == 'span':  
            #     article += node.get_text(strip=True)


        article = article.replace("\n", "")
        article = article.replace("\t", "")
        article = article.split("※ '당신의 제보가 뉴스가 됩니다'")[0]
        article = article.split("▷ 자세한 뉴스 곧 이어집니다.")[0]

        # 영어로 되어있으면
        if count_korean_words(article) < 0.9:  
            link.append(href)
            title.append("")
            content.append("")
        else :
            if(len(article) <= 100) :
                link.append(href)
                title.append("")
                content.append("")
            else :
                link.append(href) 
                title.append(title_tmp)
                content.append(article)
        
    except:
        link.append(href)
        title.append("")
        content.append("")
        print("except title : ", "except title")
        pass

    for li in request_data['list'] :
        item = TbSingoDataM(link=link[0], title=title[0],content=content[0],label=li)
        item.save()
        
    return JsonResponse({"sucess" : "sucess"})


# Define function to count Korean words
def count_korean_words(text):
    words = text.split()
    korean_words = [word for word in words if re.search(r'[가-힣]', word)]
    return len(korean_words) / len(words)


# 1세부 예측 모델
def sebu1_model(df) :

    global tokenizer1
    global model1

    df_copy = df[(df['title'] != "") & (df['content'] != "")]
    df_copy['content'] = df_copy['content'].map(make_paragraph_overlap)
    new_df = df_copy.explode('content')
    new_df.reset_index(inplace = True)

    BATCH_SIZE = 1
    test_set = TextDataset(new_df, tokenizer1)
    test_loader = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = False)

    # 모델 예측 후 predict열 수정
    predictions = []

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model1(input_ids, attention_mask=attention_mask)
            temperature = 3
            probs = temperature_scaled_softmax(outputs.logits, temperature)
            prob_class_0 = probs[:, 0]

            predictions.extend(prob_class_0)
    new_df['predict'] = predictions
    df['result'] = new_df.groupby('index')['predict'].mean()
    df['result'].fillna('내용없음', inplace = True)


    return list(df['result'])


# 2세부 예측 모델
def sebu2_model(df):
    result = []
    chosen_combos = ((2, 700), (5, 200), (5, 600), (6, 300), (6, 450))
    min_proba_groups = []
    suspicious_sentences = []
    for _, row in df.iterrows():
        title = Preprocessing(row['title'])
        content = Preprocessing(row['content'])
        ensemble_group_probas = []
        if content == '내용없음':
            result.append('내용없음')
            suspicious_sentences.append('내용없음')
            continue
        for combo in chosen_combos:
            step, max_length = combo
            groups = sliding_window_sentences(title, content, step, max_length)
            group_probas = []
            for group in groups:
                inputs = tokenizer2(group, return_tensors="pt", truncation=True, padding=True)
                output = model2(**inputs)
                probabilities = F.softmax(output.logits, dim=-1)
                group_probas.append(probabilities[0][1].item())
            min_proba = min(group_probas)
            ensemble_group_probas.append(min_proba)
            min_proba_groups.append((min_proba, group_probas.index(min_proba)))
        ensemble_proba = np.mean(ensemble_group_probas)
        value = abs(1 - ensemble_proba)
        result.append(value)
        if ensemble_proba <= 0.5:
            suspicious_sentences.append(groups[min_proba_groups[-1][1]])
        else:
            suspicious_sentences.append('정상글')
    return result, suspicious_sentences
        
# 
#  2세부 함수 시작
# 

def Preprocessing(text):
    if not isinstance(text, str) or text.strip() == "":
        return '내용없음'
    text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z\s\[\]]().?!$%&", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def sliding_window_sentences(title, content, step=1, max_length=800):
    sentences = re.split("(다\.)", content)[:-1]
    sentences = ["".join(x) for x in zip(sentences[0::2], sentences[1::2])]
    groups = []
    last_sentence_included = False
    step = min(step, len(sentences))
    for start in range(0, len(sentences), step):
        if last_sentence_included:
            break
        group = title + " [SEP] "
        for i in range(start, len(sentences)):
            if len(group + sentences[i]) > max_length:
                break
            group += sentences[i]
            if i == len(sentences) - 1:
                last_sentence_included = True
        groups.append(group)
    return groups


# 
#  1세부 함수 시작
# 

def temperature_scaled_softmax(logits, temperature):
    logits = logits / temperature
    return torch.nn.functional.softmax(logits, dim=-1)

# '본몬' 열의 길이가 700을 넘지 않도록 문단 생성
def make_paragraph_overlap(text):
    sentences = text.split('다.')[:-1]
    paragraphs = []
    current_paragraph = ''

    for sentence in sentences:
        sentence += '다.'
        if len(current_paragraph + sentence) > 350:
            paragraphs.append(current_paragraph)
            current_paragraph = sentence
        else:
            current_paragraph += sentence
    paragraphs.append(current_paragraph)

    overlapping_paragraphs = []
    overlapping_paragraphs.append(paragraphs[0])
    for i in range(len(paragraphs) - 1):
        overlapping_paragraphs.append(paragraphs[i] + paragraphs[i+1])
    overlapping_paragraphs.append(paragraphs[-1])

    return overlapping_paragraphs


# 토크나이저 적용
class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
      self.df = df
      self.tokenizer = tokenizer

    def __len__(self):
      return len(self.df)

    def __getitem__(self, idx):
        title = self.df.loc[idx, 'title']
        content = self.df.loc[idx, 'content']

        inputs = self.tokenizer(
            title,
            content,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=512,
            add_special_tokens=True
        )

        return{
          'input_ids': inputs['input_ids'][0],
          'attention_mask': inputs['attention_mask'][0]
        }

# 광고모델
def gwango_model(df) :

    global tokenizer3
    global model3

    # val_predictions_result = list()
    val_predictions = list()

    for idx in range(len(df)) :
        pre_encoding = tokenizer3(
            str(df.loc[idx]['title']),
            str(df.loc[idx]['content']),
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            output = model3(pre_encoding['input_ids'], attention_mask=pre_encoding['attention_mask'])
            predict = torch.sigmoid(output.logits)
            val_predictions.append(predict[0][0].item())

    return val_predictions


# 안씀
# @csrf_exempt
# @require_POST
def getTitleContentUsingByHref(request) :
    
    title = list()
    content = list()
    link = list()

    request_data = json.loads(request.body)
    
    href_list = request_data['href']
    # href_list = ['https://n.news.naver.com/article/088/0000826588?cds=news_media_pc&type=breakingnews', 'https://n.news.naver.com/article/448/0000419539?cds=news_media_pc&type=breakingnews', 'https://n.news.naver.com/article/296/0000067955?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/448/0000419506?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/082/0001223367?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/088/0000826585?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/657/0000016846?cds=news_media_pc&type=editn', 'https://n.news.naver.com/mnews/hotissue/article/015/0004871071?cid=2000053&type=series&cds=news_media_pc', 'https://n.news.naver.com/mnews/hotissue/article/079/0003793875?cid=1089896&type=series&cds=news_media_pc', 'https://n.news.naver.com/mnews/hotissue/article/016/0002173093?cid=2000320&type=series&cds=news_media_pc', 'https://n.news.naver.com/article/022/0003835915?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/022/0003835910?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/022/0003836073?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/022/0003835709?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/088/0000826368?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/088/0000826372?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/088/0000826373?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/657/0000016820?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/657/0000016784?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/657/0000016747?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/657/0000016734?cds=news_media_pc&type=editn', 'https://n.news.naver.com/article/119/0002732947?sid=100&type=journalists&cds=news_media_pc', 'https://n.news.naver.com/article/025/0003295523?sid=100&type=journalists&cds=news_media_pc', 'https://n.news.naver.com/article/469/0000751123?sid=110&type=journalists&cds=news_media_pc', 'https://n.news.naver.com/article/092/0002299482?sid=105&type=journalists&cds=news_media_pc']

    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}

        
    print(len(href_list))

    try:
        for href in href_list:
            if "https://n.news.naver.com" in href:
                try:
                    
                    web_news = requests.get(href, headers=headers).content
                    source_news = BeautifulSoup(web_news, 'html.parser')

                    # 예외처리, 시간재기
                    title_tmp = source_news.find('h2', {'class' : 'media_end_head_headline'}).get_text()

                    div_element  = source_news.find('div', {'id' : 'dic_area'})

                    # div 자식 노드들의 텍스트를 순서대로 article 변수에 담기
                    article = ""

                    for node in div_element.children:
                        # 텍스트 노드인 경우
                        if isinstance(node, str):  
                            article += node.strip()
                        elif node.name == 'span' and 'end_photo_org' not in node.get('class', []):  # span 요소인 경우(class가 "end_photo_org"가 아닌 경우)
                            article += node.get_text(strip=True)
                        # # span 요소인 경우
                        # elif node.name == 'span':  
                        #     article += node.get_text(strip=True)


                    article = article.replace("\n", "")
                    article = article.replace("\t", "")
                    article = article.split("※ '당신의 제보가 뉴스가 됩니다'")[0]
                    article = article.split("▷ 자세한 뉴스 곧 이어집니다.")[0]

                    if count_korean_words(article) >= 0.9 :  
                        if(len(article) > 100) :
                            link.append(href) 
                            title.append(title_tmp)
                            content.append(article)

                except Exception as e:
                    print(f"예외가 발생했습니다: {e}")
                    pass
        
        data = {
            'link' : link,
            'title': title,
            'content': content
        }

        print(len(link))

        df = pd.DataFrame(data)
        
        now = datetime.now()
        print("1세부 시작", now.strftime('%Y-%m-%d %H:%M:%S'))

        sebu1_result = sebu1_model(df)
        
        now = datetime.now()
        print("1세부 끝 2세부 시작", now.strftime('%Y-%m-%d %H:%M:%S'))

        sebu2_result, suspicious_sentences_result = sebu2_model(df)
        
        now = datetime.now()
        print("2세부 끝 광고 시작", now.strftime('%Y-%m-%d %H:%M:%S'))

        gwango_result = gwango_model(df)

        now = datetime.now()
        print("광고 끝", now.strftime('%Y-%m-%d %H:%M:%S'))

        for i in range(len(link)) :
            try : 
                item = TbFishingDataM(link=link[i], 
                                    title=title[i],
                                    content=content[i],
                                    number_1sebu=str(sebu1_result[i]),
                                    number_2sebu=str(sebu2_result[i]),                                   
                                    suspicious_sentences=str(suspicious_sentences_result[i]),
                                    gwango=str(gwango_result[i]),
                                    gwango_like=0,
                                    gwango_hate=0,
                                    number_1sebu_like=0,
                                    number_1sebu_hate=0,
                                    number_2sebu_like=0,
                                    number_2sebu_hate=0)
                item.save()
            except :
                pass
    except :
                pass
    
    data = {
        'success' : 'success'
    }

    return JsonResponse(data)

# getTitleContentUsingByHref()
# 맨뒤에 있어야함
# tenminute()
