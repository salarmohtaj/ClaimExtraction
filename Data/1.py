import requests as RQ
from bs4 import BeautifulSoup
import re
import json
import pandas as pd

with open("jsonData_withcontent.json", "r") as f:
    temp_dic = json.load(f)
print(len(temp_dic))
print(temp_dic[0])


def clean(content):
    content = content.lstrip()
    content = content.rstrip()
    content = content.replace('"'," ")
    content = re.sub(' +', " ", content)
    content = content.lstrip()
    content = content.rstrip()
    return content

df = pd.DataFrame(columns=["checker","claim","content"])
print(df.shape)
for item in temp_dic:
    checker = item["fact_checker"]
    claim = item["claim"]
    claim = clean(claim)
    content = item["content"]
    content = content[:2000]
    content = clean(content)
    if((claim != "-") and (content != "-")):
        df_temp = pd.DataFrame([[checker,claim,content]], columns=["checker","claim","content"])
        #df_temp = pd.DataFrame([[checker,claim,content]], columns=["text", "label"])
        df = df.append(df_temp,ignore_index=True)
print(df.shape)
print(df.head(5))
df.to_csv("finalDataFrame.csv",sep="\t")