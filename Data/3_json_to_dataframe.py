import requests as RQ
from bs4 import BeautifulSoup
import re
import json
import pandas as pd
from langdetect import detect


with open("jsonData_withcontent.json", "r") as f:
    temp_dic = json.load(f)
print(len(temp_dic))
print(temp_dic[0])


def clean(content):
    content = content.lstrip()
    content = content.rstrip()
    content = content.replace('.\n'," ")
    content = content.replace('\n', ". ")
    content = content.replace('\u201c', '"')
    content = content.replace('\u201d', '"')
    content = content.replace('\u2019', "'")
    content = content.replace('\u2018', "'")
    content = content.replace('\u2014', "-")
    content = content.replace('\n', " ")
    content = content.replace('“', " ")
    content = content.replace('”', " ")
    content = content.replace('"', " ")
    content = re.sub(' +', " ", content)
    content = content.lstrip()
    content = content.rstrip()
    return content

df = pd.DataFrame(columns=["checker","claim","content"])
print(df.shape)
for item in temp_dic:
    checker = item["fact_checker"]
    claim = item["claim"]
    try:
        lan1 = detect(item["content"])
        lan2 = detect(item["claim"])
        if ((lan1 != "en") or (lan2 != "en")):
            continue
    except:
        continue
    claim = clean(claim)
    content = item["content"]
    content = content[:2000]
    content = clean(content)
    if((claim != "-") and (content != "-")):
        df_temp = pd.DataFrame([[checker,claim,content]], columns=["checker","claim","content"])
        #df_temp = pd.DataFrame([[checker,claim,content]], columns=["text", "label"])
        df = df.append(df_temp,ignore_index=True)
        item["claim"] = claim
        item["content"] = content
print(df.shape)
print(df.head(5))
df.to_csv("finalDataFrame.csv",sep="\t")

with open("jsonData_withcontent_preprocessed.json", "w") as f:
    json.dump(temp_dic,f)