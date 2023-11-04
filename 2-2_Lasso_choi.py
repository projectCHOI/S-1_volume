#준비하기
import pandas as pd # pandas 가져오기
import numpy as np # numpy는 수치 계산

# CSV 파일을 불러오기
# df = pd.read_csv('your_data.csv')

# 데이터 준비
X = df[['열1', '열2', '열3']]
y = df['열2']

from sklearn.preprocessing import StandardScaler # 데이터를 표준화
# 규격화 (Min-Max Scaling)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 형태화 (로그 변환)
X_log_transformed = np.log1p(X)

# 표준화 (Standardization)
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split # 데이터를 학습
from sklearn.linear_model import LassoCV, Lasso # LassoCV회기, Lasso 회귀는 선형 회귀 모델
from sklearn.preprocessing import MinMaxScaler # 알파 값구하기 Min-Max 데이터를 [0, 1] 범위로 변환하는 데 사용

# LassoCV를 사용하여 최적 alpha 값 선택
alphas = [0.001, 0.01, 0.1, 1, 10]  # alpha 후보 값 설정
lasso_cv = LassoCV(alphas=alphas, cv=5)  # 5-폴드 교차 검증
lasso_cv.fit(X, y)
selected_alpha = lasso_cv.alpha_  # 최적 alpha 값 선택

# Lasso 회귀 모델 훈련
lasso = Lasso(alpha=selected_alpha)  # 선택한 alpha로 Lasso 모델 생성
lasso.fit(X, y)

# 결과 값 출력
print("Selected Alpha:", selected_alpha)
print("Lasso Coefficients:", lasso.coef_)

