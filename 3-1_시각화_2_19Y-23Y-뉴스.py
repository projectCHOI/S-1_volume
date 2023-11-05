import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import yfinance as yf

# Load article data (using t.csv file)
article_data = pd.read_csv('t.csv')
article_data['date'] = pd.to_datetime(article_data['date'])

# Calculate article counts per day
daily_article_counts = article_data.groupby('date').size().reset_index(name='Article Count')

# Get S1 Corporation stock data
symbol = "012750.KS"  # S1 Corporation stock symbol
start_date = "2019-01-01"
end_date = "2023-10-24"
s1_stock_data = yf.download(symbol, start=start_date, end=end_date)

# Find peak points in stock trading volume (top 5%)
threshold = s1_stock_data['Volume'].quantile(0.95)
peak_points_stock = s1_stock_data[s1_stock_data['Volume'] >= threshold]

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

# Article Count Graph (Red)
ax1.plot(daily_article_counts['date'], daily_article_counts['Article Count'], color='red', label='Article Count')
ax1.set_xlabel('Date')
ax1.set_ylabel('Article Count', color='red')

# Stock Trading Volume Graph (Black)
ax2 = ax1.twinx()
ax2.plot(s1_stock_data.index, s1_stock_data['Volume'], color='black', label='S1 Corporation Trading Volume')
ax2.scatter(peak_points_stock.index, peak_points_stock['Volume'], color='black', marker='o', s=20, label='Peak Points (Trading Volume, Top 5%)')
ax2.set_ylabel('Trading Volume', color='black')

# Adding Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc='upper left')

# Setting Title
plt.title('Article Count and S1 Corporation Trading Volume (2019-01-01 ~ 2023-10-24)')

# Display the graph
plt.grid(True)
plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
plt.tight_layout()
plt.show()
