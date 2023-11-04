import gensim  # 자연어 처리 및 토픽 모델링을 위한 라이브러리입니다.
from gensim import corpora  # 텍스트 문서의 코퍼스
import pandas as pd  # Pandas 라이브러리를 'pd' 

# CSV 파일 경로 설정
csv_file_path = 'YOUR_CSV_FILE_PATH.csv'  # CSV 파일 경로를 실제 파일 경로로 변경
# CSV 파일 불러오기
df = pd.read_csv(csv_file_path)

from konlpy.tag import Okt  # Okt 형태소 분석
tokenizer = Okt()
# "POS" 열을 생성하고 형태소 분석 결과 저장
df['POS'] = df['content'].apply(lambda text: ' '.join(tokenizer.morphs(text)))

import re  # 정규 표현식
regex_pattern = r'\<[^\>]*\>|\&#8203;``&#8203;``【oaicite:0】``&#8203;``&#8203;]*\】|\[[^\)]*\]|\([^\)]*\)|[0-9]*\.[0.9]*?\.[0-9]*|[a-zA-Z]*@[a-zA-Z]*\.[a-zA-Z]*\.?[a-zA-Z]*|Copyright|ⓒ|스포츠서울&sportsseoul.com|제공'
# "RE" 열을 생성하고 정규 표현식을 적용하여 분류된 결과 저장
df['RE'] = df['POS'].apply(lambda text: re.sub(regex_pattern, ' ', text))

# 한글자 제거 = 글자 수 1 이하 제거
df['RE-1'] = df['RE'].apply(lambda text: ' '.join(word for word in text.split() if len(word) > 1))

# 불용어 처리를 위한 리스트
stop_pos = ['Noun', 'Josa', 'Alpha', 'Punctuation', 'Suffix']
stop_word = ['은', '는', '이', '가']  # 불용어 리스트에 필요한 단어 추가

# "PRO" 열을 생성하고 불용어 처리된 결과 저장
def preprocess(text):
    text = str(text).split()
    text = [i for i in text if len(i) > 1]
    text = [i for i in text if i not in stop_pos]
    text = [i for i in text if i not in stop_word]
    return text
df['PRO'] = df['RE-1'].apply(preprocess)

# 토크화 함수
def make_tokens(df):
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(i, '/', len(df))
        token = preprocess(df['PRO'][i])
        df['PRO'][i] = ' '.join(token)
    return df

df = make_tokens(df)

# 터미널 사용! 토픽 모델링을 위한 gensim 설치
# pip install gensim
from gensim.models import LdaModel, TfidfModel  # LDA 토픽 모델 구축 TF-IDF 가중치를 계산

# 토픽 모델링을 위한 데이터 준비
tokenized_docs = df['PRO'].apply(lambda x: x.split())

id2word = corpora.Dictionary(tokenized_docs)
corpus_TDM = [id2word.doc2bow(doc) for doc in tokenized_docs]
tfidf = TfidfModel(corpus_TDM)
corpus_TFIDF = tfidf[corpus_TDM]

n = 50

# LDA 모델 학습
lda = LdaModel(corpus=corpus_TFIDF, id2word=id2word, num_topics=n, random_state=100)

for t in lda.print_topics():
    print(t)

# 터미널 사용! pyLDAvis 결과 시각화
# !pip install pyLDAvis==3.4.1
import pyLDAvis  # 토픽 모델링 결과 시각화
import pyLDAvis.gensim_models as gensimvis  # yLDAvis를 사용
# 토픽 모델링 결과를 pyLDAvis 형식으로 변환
lda_display = gensimvis.prepare(lda, corpus_TFIDF, id2word)

# pyLDAvis 결과 시각화
pyLDAvis.display(lda_display)