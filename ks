용어 (계속 업데이트 하겠습니다.)
**High Quality Topic Extraction from Business News Explains Abnormal Financial Market Volatility**

(비즈니스 뉴스에서 고품질 주제 추출이 이상한 금융 시장 변동성을 설명한다)

1. Financial market volatility - refers to the degree of variation of a financial market's price over time.
2. Topic modeling - a statistical method used to identify topics or themes in a collection of texts.
3. JSD metric - Jensen-Shannon divergence metric, a measure of similarity between two probability distributions.
4. Word distribution - the frequency of occurrence of words in a text or a collection of texts.
5. FVE - Fraction of Variance Explained, a measure of how much of the variation in a dataset is explained by a particular factor or topic.
6. Panel - a graphical representation of data or information, often used in scientific research.
7. Merger and acquisition - the process of combining two or more companies into a single entity.
8. Bond credit rating - a measure of the creditworthiness of a bond issuer, based on its financial strength and ability to repay its debts.
9. Insider trading - the illegal practice of trading on the stock market using non-public information.
10. SEC - Securities and Exchange Commission, a US government agency responsible for regulating the securities industry.
11. Recall - the process of removing a defective product from the market.
12. Natural disaster - a catastrophic event caused by natural forces, such as a hurricane, earthquake, or flood.
13. BP oil spill - a major oil spill that occurred in the Gulf of Mexico in 2010, caused by an explosion on the Deepwater Horizon oil rig.
14. Global recession - a period of economic decline affecting multiple countries or regions.
15. Earning reports - financial statements released by companies that provide information on their financial performance.
16. Retailers profits - the amount of money earned by retailers from selling goods or services.
17. New products - products that are newly introduced to the market.
18. Legislation/Regulation/Bill - laws or regulations that govern the behavior of individuals or organizations in a particular industry or sector.
19. Takeover - the acquisition of one company by another, often through the purchase of a controlling stake in the target company.

1. 금융 시장 변동성 - 금융 시장 가격의 시간에 따른 변화 정도를 나타냅니다.
2. 토픽 모델링 - 텍스트 또는 텍스트 집합에서 주제나 테마를 식별하는 통계적 방법입니다.
3. JSD 지표 - Jensen-Shannon divergence 지표로, 두 확률 분포 간의 유사성을 측정하는 방법입니다.
4. 단어 분포 - 텍스트 또는 텍스트 집합에서 단어의 출현 빈도를 나타냅니다.
5. FVE - Fraction of Variance Explained로, 특정 요인이나 토픽이 데이터 집합의 얼마나 많은 변동성을 설명하는지를 나타내는 지표입니다.
6. 패널 - 데이터나 정보를 그래픽으로 나타낸 것으로, 과학 연구에서 자주 사용됩니다.
7. 합병과 인수 - 두 개 이상의 회사를 하나의 단일 엔티티로 합치는 과정입니다.
8. 채권 신용 등급 - 채권 발행자의 신용도를 나타내는 지표로, 재무적 강도와 부채 상환 능력을 기반으로 합니다.
9. 내부자 거래 - 비공개 정보를 이용하여 주식 시장에서 거래하는 불법적인 행위입니다.
10. SEC - Securities and Exchange Commission으로, 미국 정부 기관으로서 증권 산업을 규제합니다.
11. 제품 리콜 - 결함된 제품을 시장에서 회수하는 과정입니다.
12. 자연 재해 - 허리케인, 지진, 홍수 등과 같이 자연적인 힘에 의해 발생하는 대형 재해를 말합니다.
13. BP 유출 사고 - 2010년 멕시코 만에서 발생한 대규모 유출 사고로, 딥워터 호라이즌 유전 시설에서 발생한 폭발로 인해 발생했습니다.
14. 세계적 경기 침체 - 여러 나라나 지역에 영향을 미치는 경제적 하락 기간을 말합니다.
15. 수익 보고서 - 기업이 발표하는 재무 성과에 대한 정보를 제공하는 재무 보고서입니다.
16. 소매업자 이익 - 소매업자가 상품이나 서비스를 판매하여 얻는 수익을 말합니다.
17. 새로운 제품 - 시장에 새롭게 출시된 제품을 말합니다.
18. 법규제/규정/법안 - 특정 산업이��� 분야에서 개인이나 조직의 행동을 규제하는 법률이나 규정을 말합니다.
19. 인수합병 - 한 회사가 다른 회사를 인수하는 과정으로, 대상 회사의 지배 지분을 구매하는 것이 일반적입니다.



________________________________________________________________________________________________________________________
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
