# 2018_2023.csv 를 한나눔 토픽모델링 k fold 라쏘회귀파라미터 함
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from konlpy.tag import Hannanum

# Load the CSV file
df = pd.read_csv('2018_2023.csv')

# Text cleaning function using regular expressions
def clean_text(text):
    text = re.sub('\d+', '', text)
    text = re.sub('[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '', text)
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub('<.*?>', '', text)
    text = re.sub('[^\w\s가-힣]', '', text)
    text = re.sub('[\u4E00-\u9FFF]', '', text)
    return text

# Apply the cleaning function to the 'content' column
df['content'] = df['content'].apply(clean_text)

# Tokenize Korean text using Hannanum and remove stopwords
hannanum = Hannanum()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

def tokenize(text):
    return [word for word in hannanum.nouns(text) if word not in stopwords]

# Vectorize the content
vectorizer = CountVectorizer(tokenizer=tokenize, max_features=5000)
X = vectorizer.fit_transform(df['content'])

# Apply LDA for topic modeling
lda = LatentDirichletAllocation(n_components=10, random_state=42) 
lda.fit(X)

# Display topics 
for idx, topic in enumerate(lda.components_):
    print(f"Topic {idx + 1}: {[vectorizer.get_feature_names()[i] for i in topic.argsort()[-10:]]}")

# The Lasso regression part assumes you have a target variable related to the abnormal trading volume of `s1`.
# If you have it, uncomment the lines below:

# y = df['target_column_name']

# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     model = Lasso()
#     model.fit(X_train, y_train)
#     score = model.score(X_test, y_test)
#     print(f"R^2 Score: {score}")


_____________________________________________________________________________________________________________________
# 주가량 그래프 코드

import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 에스원의 Yahoo Finance 티커 심볼 입력
ticker = "012750.KS"

# 오늘 날짜를 기준으로 5년 전의 날짜를 계산
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)

# 에스원의 주식 데이터 가져오기 (5년치)
data = yf.download(ticker, start=start_date, end=end_date)

# 거래량 데이터 선택 및 날짜 인덱스 초기화
volume_data = data['Volume'].reset_index()

# 상위 5% 거래량 계산
threshold = volume_data['Volume'].quantile(0.95)

# 상위 5% 거래량에 해당하는 데이터만 선택
peak_data = volume_data[volume_data['Volume'] >= threshold]

# Plotly를 사용하여 5년치 거래량 그래프와 피크점 찍기
fig = go.Figure()

fig.add_trace(go.Scatter(x=volume_data['Date'], y=volume_data['Volume'],
                         mode='lines', name='거래량',
                         line=dict(color='blue')))

fig.add_trace(go.Scatter(x=peak_data['Date'], y=peak_data['Volume'],
                         mode='markers', name='피크 거래량',
                         marker=dict(color='red', size=8)))

fig.update_layout(title='에스원 5년치 거래량과 피크 거래량',
                  xaxis_title='날짜',
                  yaxis_title='거래량',
                  hovermode="x")

fig.show()

# 1년치 데이터를 연도별로 분리하여 거래량 그래프와 피크점 찍기
for year in range(start_date.year, end_date.year + 1):
    start_date_year = datetime(year, 1, 1)
    end_date_year = datetime(year, 12, 31)
    
    # 해당 연도의 데이터 선택
    mask = (volume_data['Date'] >= start_date_year) & (volume_data['Date'] <= end_date_year)
    yearly_data = volume_data[mask]
    
    # 해당 연도의 상위 5% 거래량 계산
    threshold_yearly = yearly_data['Volume'].quantile(0.95)
    
    # 해당 연도의 상위 5% 거래량에 해당하는 데이터만 선택
    peak_data_yearly = yearly_data[yearly_data['Volume'] >= threshold_yearly]
    
    # Plotly를 사용하여 거래량 그래프와 피크점 찍기
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=yearly_data['Date'], y=yearly_data['Volume'],
                             mode='lines', name='거래량',
                             line=dict(color='blue')))
    
    fig.add_trace(go.Scatter(x=peak_data_yearly['Date'], y=peak_data_yearly['Volume'],
                             mode='markers', name='피크 거래량',
                             marker=dict(color='red', size=8)))
    
    fig.update_layout(title=f'에스원 {year}년 거래량과 피크 거래량',
                      xaxis_title='날짜',
                      yaxis_title='거래량',
                      hovermode="x")
    
    fig.show()
