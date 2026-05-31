import re

with open("../data/urls_finepdfs/filtered_urls_roh.txt","r") as f:
    roh = f.readlines()

with open("../data/urls_finepdfs/filtered_urls_fra.txt","r") as f:
    fra = f.readlines()

with open("../data/urls_finepdfs/filtered_urls_ita.txt","r") as f:
    ita = f.readlines()

with open("../data/urls_finepdfs/filtered_urls_deu.txt","r") as f:
    deu = f.readlines()


urls_roh = [url for url in roh if "www.bk.admin.ch/" in url]
urls_fra = [url for url in fra if "www.bk.admin.ch/" in url]
urls_ita = [url for url in ita if "www.bk.admin.ch/" in url]
urls_deu = [url for url in deu if "www.bk.admin.ch/" in url]


#print(urls_deu[:3])


#print("deu:",len(urls_deu))
#print("ita:",len(urls_ita))
#print("fra:",len(urls_fra))
#print("roh:",len(urls_roh))


dates_deu = set()
dates_ita = set()
dates_fra = set()
dates_roh = set()


def extract_dates(dates,urls):
    pattern = r"\d{1,2}(?:[a-zA-Z]{3}|\d{1,2})\d{2,4}"
    
    
    for url in urls:
        if "Abstimmungsbuechlein" in url:
            matches = re.findall(pattern, url, flags=re.IGNORECASE)
            dates.update(set(matches))

    return dates

extract_dates(dates_roh,urls_roh)

extract_dates(dates_deu,urls_deu)

extract_dates(dates_fra,urls_fra)

extract_dates(dates_ita,urls_ita)

#print(dates_roh)

# print the dates that are contained in every language
for d in dates_deu:
    if len(d) >4:
        if d in dates_fra:
            if d in dates_ita:
                if d in dates_roh:
                    print(d)


# to get the url to download the pdfs from the web
# replace urls_fra with the other languages to get the other languages too
for url in urls_fra:
        # the first booklet i chose is from the 01.12.1985
    if "01121985" in url:
        print(url)

    if "11032007" in url: 
        print(url)

"""
These are the dates i find in all of them:

01121985
06121987
04121988
04061989
02061991
12061994
04121994
12031995
25061995
07021999
18052003
11032007

"""

# check if there is an older scan maybe not in all languages 
# this date below is the oldest i found in the dataset

# this data does not exist for roh but for the other 3 languages
for d in dates_ita:
    if len(d) >4 and int(d[-4:])<1980:
        print(d)

for url in urls_ita:
    if "12061977" in url:
        print(url)





