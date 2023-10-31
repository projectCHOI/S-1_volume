#choi_LDA
## 파일 불러오기
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv('combined_data.csv')

## 전처리 1번
import re

df['regex'] = None
reg = '\[[^\]]*\]|\＜.*\＞|\<.*\>|[a-zA-Z0-9]*\@[a-zA-Z0-9]*\.[a-zA-Z0-9]*\.?[a-zA-Z0-9]*|[a-zA-Z0-9]*\.com|\※|저작권자|\ⓒ|\▶|[0-9]*\-[0-9]*\-[0-9]*|\&|△'
regex = '\<[^\>]*\>'

for i in range(len(df['content'])):
  # text = re.sub(regex1, '', df['content'][i])
  text = re.sub(reg, '', df['content'][i])

  txt = re.sub(regex, '', text).replace('\n', '').strip()
  df['regex'][i] = txt

## pip install konlpy 설치
## Okt를 사용해 형태소 분리
from konlpy.tag import Okt
okt = Okt()
df['pos'] = None

for i in range(len(df['regex'])):
  df['pos'][i] = okt.pos(df['regex'][i])
  print(df['pos'][i])
