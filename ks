


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
