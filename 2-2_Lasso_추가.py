import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

# 데이터 파일 경로 설정 (비주얼 스튜디오 프로젝트 내의 상대 경로 또는 절대 경로 사용)
data_path = "path/to/your/datafile.csv"  # 예시 경로

# 데이터 로드
df = pd.read_csv(data_path, header=0)

# 데이터 전처리 및 스케일링
scaler = MinMaxScaler()
df['trading_volume_scaled'] = scaler.fit_transform(df['trading_volume'].values.reshape(-1, 1))
df['news_volume_scaled'] = scaler.fit_transform(df['news_volume'].values.reshape(-1, 1))

# 데이터 분할
X = df['news_volume_scaled'].values.reshape(-1, 1)
y = df['trading_volume_scaled'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 라쏘 회귀 모델 학습
cv = 100
lasso_cv = LassoCV(cv=cv, random_state=0)
lasso_cv.fit(X, y)
best_alpha = lasso_cv.alpha_

# 예측 및 평가
lasso_model = Lasso(alpha=best_alpha)
lasso_model.fit(X_train, y_train)
y_pred = lasso_model.predict(X_test)

# 평가 지표 출력
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'CV: {cv}, MSE: {mse}, MAE: {mae}, R^2: {r2}')

##
##

###### (뉴스량, 거래량) 시각화
plt.scatter(X, y, label='true data', color='blue')
plt.plot(X, y_pred, label='regression line', color='red')
plt.xlabel('news volume')
plt.ylabel('trading volume')
plt.title(f'correlationship between topic{topicNo} news volume and trading volume')
plt.legend()
plt.show()

###### 시계열로 시각화
plt.plot(date, y, label='true volume', color='blue')
plt.plot(date, y_pred, label='predicted volume', color='red')
plt.xlabel('news volume')
plt.ylabel('trading volume')
plt.title(f'topic{topicNo} predicted trading volume')
plt.legend()
plt.show()
  
##for문으로 저장
# y절편
intercept = lasso_model.intercept_
intercept
     
topic_list = [1,2,3,4,6,7,9,10,11,14,15,16,19,20,21,28,33,35,39,44,47,49]

for topicNo in topic_list :

  topic = pd.read_csv(f'/content/drive/MyDrive/SK하이닉스/07. LASSO 회귀/토픽별 일간 기사량/topic{topicNo}_newsvolume.csv')
  topic = pd.merge(stock, topic, how='left')
  topic = topic.fillna(0)

  # 정규화
  scaler = MinMaxScaler()
  topic['news_volume_scaled'] = scaler.fit_transform(topic['news_volume'].values.reshape(-1, 1))
  stock['trading_volume_scaled'] = scaler.fit_transform(stock['trading_volume'].values.reshape(-1, 1))

  # x : 기사량, y : 거래량
  X = topic['news_volume_scaled'].values.reshape(-1, 1)
  y = stock['trading_volume_scaled'].values.reshape(-1, 1)

  y_pred = lasso_model.predict(X)
  print(f'topic{topicNo} 예측된 y :', y_pred)
  topic['pred_trading_vol_scaled'] = y_pred

  # 평가 지표 계산
  mse = mean_squared_error(y, y_pred)
  mae = mean_absolute_error(y, y_pred)
  r2 = r2_score(y, y_pred)
  print('mse, mae, r2 :', mse, mae, r2)
  print()

  # 저장
  # topic.to_csv(f'/content/drive/MyDrive/SK하이닉스/07. LASSO 회귀/토픽별 일간 예측된 거래량/topic{topicNo}_pred_volume.csv', index=False)
