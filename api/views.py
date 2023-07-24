from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, Http404
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification, AdamW
import re
import os

import json
import requests

from bs4 import BeautifulSoup
from datetime import datetime
import numpy as np

base_dir =  settings.BASE_DIR
device=torch.device('cpu')

# 1세부 모델 초기화
HUGGINGFACE_MODEL_PATH = "klue/roberta-base"
tokenizer1 = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH)
model_file_path1 =  base_dir + "/api/models/1model.pth"
checkpoint = torch.load(model_file_path1, map_location=device)
model1 = RobertaForSequenceClassification.from_pretrained(HUGGINGFACE_MODEL_PATH, num_labels=2)
model1.load_state_dict(checkpoint['model_state_dict'])

model1.eval()


# 2세부 모델 초기화
model2 = AutoModelForSequenceClassification.from_pretrained("klue/roberta-small", num_labels=2)
tokenizer2 = AutoTokenizer.from_pretrained("klue/roberta-small")
model_file_path2 =  base_dir + "/api/models/2model.pt"
model2.load_state_dict(torch.load(model_file_path2, map_location=device))

model2.eval()

def index(request):
 return HttpResponse('<h1>Hello, world!</h1>')

# 커스텀 직렬화 함수를 사용하여 set을 직렬화하는 예시
def set_serializer(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError('Object of type set is not JSON serializable')

@csrf_exempt
@require_POST
def getTitleContentUsingByHref(request) :
    
    title = list()
    content = list()
    link = list()

    request_data = json.loads(request.body)
    
    href_list = request_data['href']
    
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}

    for href in href_list :
        
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
       
    data = {
        'link' : link,
        'title': title,
        'content': content
    }

    df = pd.DataFrame(data)

    now = datetime.now()
    print("1세부 시작", now.strftime('%Y-%m-%d %H:%M:%S'))

    sebu1_result = sebu1_model(df)
    
    now = datetime.now()
    print("1세부 끝 2세부 시작", now.strftime('%Y-%m-%d %H:%M:%S'))

    sebu2_result = sebu2_model(df)
    
    now = datetime.now()
    print("2세부 끝", now.strftime('%Y-%m-%d %H:%M:%S'))

    result = {"link" : link, 
              "sebu1_result" : sebu1_result, 
              "sebu2_result" : sebu2_result}

    print("끝", len(df))


    return JsonResponse(result)



# 1세부 예측 모델
def sebu1_model(df) :

    global tokenizer1
    global model1

    df_copy = df[(df['title'] != "") & (df['content'] != "")]
    df_copy['previous_index'] = df_copy.index


    df_copy['content'] = df_copy['content'].map(make_paragraph)
    new_df = df_copy.explode('content')

    BATCH_SIZE = 8
    test_set = TextDataset(new_df, tokenizer1)
    test_loader = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = False)

    # 모델 예측 후 predict열 수정
    predictions = []

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            outputs = model1(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            prob_class_1 = probs[:, 1].cpu().numpy()

            predictions.extend(prob_class_1)

    new_df['predict'] = predictions
    df['predict'] = new_df.groupby('previous_index')['predict'].min()
    df['predict'].fillna('내용없음', inplace=True)
    result = list(df['predict'])

    return result


# 2세부 예측 모델
def sebu2_model(df) :
    
    global tokenizer2
    global model2

    result = []

    with torch.no_grad():
        
        for idx, row in df.iterrows():
            
            if(row['title'] == "" and row['content'] == "") :
                result.append("내용없음")
            else :
                title = Preprocessing(row['title'])
                content = Preprocessing(row['content'])
                groups = group_sentences(title, content)
                class1_probs = []

                for group in groups:
                    inputs = tokenizer2(group, return_tensors="pt", truncation=True, padding=True)
                    outputs = model2(**inputs)
                    probabilities = F.softmax(outputs.logits, dim=-1)
                    class1_probs.append(probabilities[0][1].item())  

                min_class1_prob = min(class1_probs) if class1_probs else None
                result.append(min_class1_prob)

    return result

# 
#  2세부 함수 시작
# 

# 정규식
def Preprocessing(text):
    if not isinstance(text, str):
        return str(text)

    ext = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z\s\[\].]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text
# 토큰 붙이기
def group_sentences(title, content, max_length=350):
    sentences = content.split('.')
    groups = []
    current_group = title + " [SEP] "

    for sentence in sentences:
        sentence = sentence.strip() + "다. "
        new_group = current_group + sentence
        if len(new_group) <= max_length:
            current_group = new_group
        else:
            groups.append(current_group)
            current_group = title + " [SEP] " + sentence

    if current_group:
        groups.append(current_group)

    return groups

# 
#  1세부 함수 시작
# 

# 예외처리
def process_empty_values(row):
    if row['title'] == '' or row['content'] == '':
        return "내용없음"
    else:
        return 1


# '본몬' 열의 길이가 700을 넘지 않도록 문단 생성
def make_paragraph(text):
    sentences = text.split('다.')[:-1]
    paragraphs = []
    current_paragraph = ''

    for sentence in sentences:
        sentence += '다.'
        if len(current_paragraph + sentence) > 700:
            paragraphs.append(current_paragraph)
            current_paragraph = sentence
        else:
            current_paragraph += sentence
    paragraphs.append(current_paragraph)
    return paragraphs

# 토크나이저 적용
class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
      self.df = df
      self.tokenizer = tokenizer

    def __len__(self):
      return len(self.df)

    def __getitem__(self, idx):
        title = self.df.iloc[idx, 1]
        content = self.df.iloc[idx, 2]

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