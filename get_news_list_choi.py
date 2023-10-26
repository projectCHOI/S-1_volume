import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd

def get_news(URL):
    res = requests.get(URL)
    soup = BeautifulSoup(res.text, 'html.parser')

    title_element = soup.select_one('h2#title_area > span')
    if title_element:
        title = title_element.text
    else:
        title = "Title not found"

    content_element = soup.select_one('article#dic_area')
    if content_element:
        content = content_element.text.strip()
    else:
        content = "Content not found"

    date_element = soup.select_one('span._ARTICLE_DATE_TIME')
    if date_element and 'data-date-time' in date_element.attrs:
        date = date_element['data-date-time']
    else:
        date = "Date not found"

    return (title, date, content)

def get_news_list(keyword, startdate, enddate):
    file = open("s1-23-4.csv", mode="w", encoding="utf-8", newline="")
    writer = csv.writer(file)

    headers = {
        'Cookie': 'NNB=NOAWMWOS2LXGI; nx_ssl=2; _naver_usersession_=U/xBSxXXxIO1prOM6PxioQ==; page_uid=id0GBlp0Jywss7BI5SNssssstuN-197031',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
        'Referer': 'https://search.naver.com/search.naver?where=news&sm=tab_pge&query=%ED%85%8C%EC%8A%AC%EB%9D%BC&sort=1&photo=0&field=0&pd=3&ds=2023.09.21&de=2023.09.21&mynews=0&office_type=0&office_section_code=0&news_office_checked=&office_category=0&service_area=0&nso=so:dd,p:from20230921to20230921,a:all&start=91'
    }

    for nowdate in pd.date_range(startdate, enddate):
        nowdate = str(nowdate).replace('-', '.').split()[0]
        page = 1

        while True:
            start = (page - 1) * 30 + 1
            URL = 'https://search.naver.com/search.naver?where=news&sm=tab_pge&query={}&sort=1&photo=0&field=0&pd=3&ds={}&de={}&mynews=0&office_type=0&office_section_code=0&news_office_checked=&office_category=0&service_area=0&nso=so:dd,p:from{}to{},a:all&start={}'.format(
                keyword, nowdate, nowdate, nowdate.replace('.', ''), nowdate.replace('.', ''), start)
            res = requests.get(URL, headers=headers)
            soup = BeautifulSoup(res.text, 'html.parser')

            if not soup.select('ul.list_news'):
                break

            for li in soup.select('ul.list_news > li'):
                if len(li.select('div.info_group > a')) == 2:
                    news_url = li.select('div.info_group > a')[1]['href']
                    news_data = get_news(news_url)
                    writer.writerow(news_data)

            page += 1

    file.close()

get_news_list('에스원', '2023.04.01', '2023.04.30')