import json
import pandas as pd
from nltk.corpus import stopwords
from textblob import TextBlob, Word

stopwords = stopwords.words('english')

def remove_stop_words(text):
    return " ".join([ x for x in text.split() if len(x) > 2 and x not in stopwords]).lower()

def lemmatize_words(text):
    return " ".join([Word(str(TextBlob(x).correct())).lemmatize() for x in text.split()])

data = [json.loads(line) for line in open('Cell_Phones_And_Accessories_5.json', 'r')]
df = pd.DataFrame.from_dict(data)

review_text = df[['reviewText', 'summary', 'overall']]

print(f"Score mean: {review_text['overall'].mean()}")

rt1 = review_text[review_text['overall'] == 1].sample(n=2000)
rt2 = review_text[review_text['overall'] == 2].sample(n=2000)
rt3 = review_text[review_text['overall'] == 3].sample(n=2000)
rt4 = review_text[review_text['overall'] == 4].sample(n=2000)
rt5 = review_text[review_text['overall'] == 5].sample(n=2000)

frames = [rt1, rt2, rt3, rt4, rt5]
for fr in frames:
    fr.loc[:, 'reviewText_remove'] = fr.loc[:, 'reviewText'].str.replace("[^a-zA-Z#]", " ")
    fr.loc[:, 'reviewText_remove'] = fr.loc[:, 'reviewText_remove'].apply(remove_stop_words)
    fr.loc[:, 'reviewText_remove'] = fr.loc[:, 'reviewText_remove'].apply(lemmatize_words)

rt = pd.concat(frames)

rt = rt.sample(frac=1).reset_index(drop=True)

rt.to_csv("rt2000_3.csv")

