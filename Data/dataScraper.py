import requests as RQ
from bs4 import BeautifulSoup
import re
import json

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
header_needed = "The Weekly Standard"


def pre_process(content):
    content = content.split("\n")
    sentence = []
    for line in content:
        sentence.append(line.strip())
    content = "\n".join(sentence)
    content = content.replace("\xa0", " ")
    content = content.replace(" \n", "\n")
    content = re.sub(r'\n+', '\n', content)
    # content = content.replace("\n", " ")
    content = re.sub(' +', " ", content)
    return content


def washingtonpost_scraper(soup):
    try:
        title = soup.find("h1", {"data-qa": "headline"}).text
        content = soup.find("div", {"class": "article-body"}).text
        content = title + "\n" + content
    except:
        content = "-"
    return content


def politifact_scraper(soup):
    try:
        title = soup.find("div", {"class":"m-statement__quote"}).text
        content = soup.find("article", {"class": "m-textblock"}).text
        content = title + "\n" + content
    except:
        content = "-"
    return content


def factcheck_scraper(soup):
    try:
        title = soup.find("h1", {"class": "entry-title"}).text
        content = soup.find("div", {"class": "entry-content"}).text
        content = title + "\n" + content
    except:
        content = "-"
    return content

def weekly_standard(soup):
    try:
        title = soup.find("h1", {"class": "ArticlePage-headline"}).text
        content = soup.find("div", {"class": "ArticlePage-articleBody"}).text
        content = title + "\n" + content
    except:
        content = "-"
    return content


with open("jsonData_v1.json", "r") as f:
    jsonData = json.load(f)

try:
    with open("jsonData_withcontent.json", "r") as f:
        temp_dic = json.load(f)
except:
    temp_dic = []

for i in range(len(jsonData)):
    data = jsonData.pop(0)
    url = data["url"]
    try:
        if data["fact_checker"] == header_needed:
            r = RQ.get(url, headers=headers)
        else:
            r = RQ.get(url)
        html = r.text
    except:
        continue
    print(i, url)
    soup = BeautifulSoup(html,features="html.parser")
    if data["fact_checker"] == "PolitiFact":
        try:
            content = pre_process(politifact_scraper(soup))
        except:
            print("Error", url)
            content = "-"
    elif ((data["fact_checker"] == "Washington Post") or (data["fact_checker"] == "The Washington Post")):
        try:
            content = pre_process(washingtonpost_scraper(soup))
        except:
            print("Error", url)
            content = "-"
    if data["fact_checker"] == "FactCheck.org":
        try:
            content = pre_process(factcheck_scraper(soup))
        except:
            print("Error", url)
            content = "-"
    if data["fact_checker"] == "The Weekly Standard":
        try:
            content = pre_process(weekly_standard(soup))
        except:
            print("Error", url)
            content = "-"
    data["content"] = content
    temp_dic.append(data)


with open("jsonData_v1.json", "w") as f:
    json.dump(jsonData, f)

with open("jsonData_withcontent.json", "w") as f:
    json.dump(temp_dic, f)
