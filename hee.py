import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

def get_news(url):

    headers = {"Cookie":"NNB=OURWEDJJKXKGI; nx_ssl=2; _naver_usersession_=+VdPHYAMq5W53YOSu5ttbQ==; page_uid=ieZzndprvmsssdTDFjRssssssah-332673"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    try:
      title = soup.select_one("h2#title_area").text
      date = soup.select_one("div.media_end_head_info_datestamp span")["data-date-time"].split(" ")[0]
      media = soup.select_one("a.media_end_head_top_logo img")["title"]
      content = re.sub("\<br\/\>|[a-zA-Z]*\@[a-zA-Z]*\.co.kr|\<[^\>]*\>", "", str(soup.select_one("div#newsct_article"))).replace("    ", " ").replace("\n", "").strip()

    except:
      try:
        title = soup.select_one('h2.end_tit').text
        date = soup.select_one('span.author em').text
        media = soup.select_one('div.press_logo a img')['alt']
        content = soup.select_one('div#articeBody')

      except:
        print(url)
        # title =

    return (title, date, media, content, url)


def get_news_list(keyword, start_date, end_date):

    news = []

    for date in pd.date_range(start_date, end_date):
        str_date = date.strftime("%Y.%m.%d")
        print("date: ", str_date)

        page = 1
        while True:
            strdate = str_date.replace(".", "")

            start = (page - 1) * 10 + 1

            headers = {"Cookie":"NNB=OURWEDJJKXKGI; nx_ssl=2; _naver_usersession_=+VdPHYAMq5W53YOSu5ttbQ==; page_uid=ieZzndprvmsssdTDFjRssssssah-332673"}
            # URL = f"https://search.naver.com/search.naver?where=news&sm=tab_pge&query={keyword}&sort=2&photo=0&field=0&pd=3&ds={str_date}&de={str_date}&mynews=0&office_type=0&office_section_code=102&news_office_checked=&office_category=0&service_area=0&nso=so:r,p:from{strdate}to{strdate},a:all&start={start}"
            URL = f'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=102&query={keyword}'
            res = requests.get(URL, headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")

            if soup.select_one("div.not_found02") != None:
              break

            print("page: ", page)

            news_list = soup.select("ul.list_news li")

            for list in news_list:
                if len(list.select("div.info_group a")) == 2:
                    news.append(get_news(list.select("div.info_group a")[1]["href"]))

            page += 1

        News = pd.DataFrame(news, columns=["제목", "날짜", "매체명", "본문", "URL"])

        News.to_csv("News.csv", mode="w")

    return News


print(get_news_list("에스원", "2023.10.24", "2023.10.24"))