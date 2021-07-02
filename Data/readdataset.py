import re
import collections
import random
import json
dic = []

with open("fact_checks_20190605.txt", 'r') as f:
    for index, line in enumerate(f):
        clean = re.compile('<.*?>')
        text = re.sub(clean, ' ', line.strip())
        data = json.loads(text)
        dic.append(data)

final_dic = []
for item in dic:
    try:
        url = item["url"]
    except:
        url = "-"
    try:
        fact_checker = item["author"]["name"]
    except:
        fact_checker = "-"
    try:
        date = item["datePublished"]
    except:
        date = "-"
    try:
        claim = item["claimReviewed"]
    except:
        claim = "-"
    temp_dic = {
        "url" : url,
        "fact_checker" : fact_checker,
        "date" : date,
        "claim" : claim
    }
    final_dic.append(temp_dic)
#with open("jsonData_v1.json", "w") as f:
#    json.dump(final_dic,f)


def count_fact_checkers(dic):
    c = []
    for item in dic:
        c.append(item["author"]["name"])
        # Counter({'PolitiFact': 3975, 'FactCheck.org': 907,
        # 'Washington Post': 659, 'The Weekly Standard': 132, 'The Washington Post': 91})
    return collections.Counter(c)


def list_of_urls(dic):
    c = []
    for item in dic:
        c.append(item["url"])
    return c


def list_of_claims(dic):
    c = []
    len_ = []
    for item in dic:
        claim = item["claimReviewed"]
        c.append(claim)
        len_.append(len(claim))
    return c, len_


# print(len(list_of_urls(dic)))
print(count_fact_checkers(dic))
c, len_ = list_of_claims(dic)
print(sum(len_)/len(len_))
# print(min(len_))
# print(max(len_))
# name_1 = "PolitiFact"
# name_2 = 'FactCheck.org'
# name_3 = 'Washington Post'
# name_4 = 'The Weekly Standard'
# name_5 = 'The Washington Post'
# list_1 = []
# list_2 = []
# list_3 = []
# list_4 = []
# list_5 = []
# for item in dic:
#     name = item["author"]["name"]
#     if(name == name_1):
#         list_1.append(item["url"])
#     elif (name == name_2):
#         list_2.append(item["url"])
#     elif (name == name_3):
#         list_3.append(item["url"])
#     elif (name == name_4):
#         list_4.append(item["url"])
#     elif (name == name_5):
#         list_5.append(item["url"])
# random.shuffle(list_1)
# random.shuffle(list_2)
# random.shuffle(list_3)
# random.shuffle(list_4)
# random.shuffle(list_5)
# print(list_1[0])
# print(list_2[0])
# print(list_3[0])
# print(list_4[0])
# print(list_5[0])

c1 = []
c2 = []
c3 = []
c4 = []
cc = 0
with open("jsonData_withcontent.json", "r") as f:
    temp_dic = json.load(f)
for item in temp_dic:
    if(item["content"] == "-"):
        cc+=1
    else:
        c1.append(item["fact_checker"])
        c2.append(len(item["claim"]))
        c3.append(len(item["content"]))
        c4.append(item["claim"])
print(collections.Counter(c1))
print(sum(c2)/len(c2))
print(sum(c3)/len(c3))
print(cc)
random.shuffle(c4)
print(c4[0])