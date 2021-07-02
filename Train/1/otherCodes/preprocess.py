import pandas as pd
import re

df = pd.read_csv("data/finalDataFrame.csv",sep="\t")

print(df[['claim', 'content']].head(5))

def preprocess(text):
    text = text.replace("\n"," ")
    text = text.replace('"', " ")
    text = text.replace('“', " ")
    text = text.replace('”', " ")
    text = re.sub(' +', ' ', text)
    return text
df['claim'] = df['claim'].apply(preprocess)
df['content'] = df['content'].apply(preprocess)

print(df[['claim', 'content']].head(5))

df.to_csv("data/finalDataFrame_preprocessed.csv",sep="\t",index=False)