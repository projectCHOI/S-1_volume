from wordcloud import WordCloud
import matplotlib.pyplot as plt

with open('문서.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# WordCloud 객체 생성 및 설정
wordcloud = WordCloud(
    width=800, height=400,  # 워드 클라우드 크기
    background_color='white',  # 배경색
    font_path='NanumGothic.ttf',  # 사용할 폰트 경로 (한글 폰트가 필요한 경우 설정)
    colormap='viridis',  # 색상 맵
    stopwords=None  # 워드 클라우드에서 무시할 단어 리스트 (None으로 설정하면 무시하지 않음)
)

# 워드 클라우드 생성
wordcloud.generate(text)

# 워드 클라우드 이미지 표시
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")  # 축 제거
plt.show()