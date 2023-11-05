!pip install mplcursors

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import mplcursors

# 데이터 불러오기
df = pd.read_csv('total_data_.csv')

# 날짜 형식으로 변환
df['date'] = pd.to_datetime(df['date'])

# 필터링: 2019-01-01부터 2023-10-24까지
df = df[(df['date'] >= '2019-01-01') & (df['date'] <= '2023-10-24')]

# 회사별 기사량 계산
companies = ['에스원', 'kt텔레캅', 'sk쉴더스', '아이디스', 'ITX', '인콘', '하이트론']
total_articles_by_date = pd.Series(dtype='float64')

for company in companies:
    company_df = df[df['content'].str.contains(company, case=False, na=False)]
    articles_by_date = company_df.groupby('date').size()
    total_articles_by_date = total_articles_by_date.add(articles_by_date, fill_value=0)

# 에스원의 주식 데이터 불러오기
s1 = yf.download('012750.KS', start='2019-01-01', end='2023-10-25')

# 그래프 그리기
fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Number of Articles', color=color)
line1, = ax1.plot(total_articles_by_date.index, total_articles_by_date.values, color=color, label='Number of Articles (S1)')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'black'
ax2.set_ylabel('Trading Volume', color=color)
line2, = ax2.plot(s1['Volume'], color=color, label='Trading Volume (S1)')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Total Number of Articles and S1 Trading Volume by Date')

# Hovering 기능 추가
cursor = mplcursors.cursor([line1, line2], hover=True)

@cursor.connect("add")
def on_add(sel):
    x, y = sel.target
    sel.annotation.set(text=f'Date: {x.strftime("%Y-%m-%d")}\nValue: {y:.0f}', position=(10, 10))
    sel.annotation.xy = (x, y)

plt.show()