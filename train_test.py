import pandas as pd
from textblob.classifiers import NaiveBayesClassifier
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB,ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import random


df = pd.read_csv("rt2000_3.csv")
df = df.dropna()
df1 = df.iloc[2000:3000]
df2 = df[~df.index.isin(df1.index)]

def classification_accuracy(a1, a2):
    counter = 0
    for i in range(0,len(a1)):
        if a1[i] == a2[i]:
            counter += 1
    return counter/len(a1)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

train = df[['reviewText_remove', 'overall']]
train['overall'] = pd.to_numeric(train['overall'],downcast='integer')

vectorizer = TfidfVectorizer(max_features=8000, min_df=20, max_df=0.60)
processed_features = vectorizer.fit_transform(train['reviewText_remove']).toarray()


msqe = 0
mae = 0
ca = 0

#text_classifier = KNeighborsClassifier(n_neighbors=20)
#text_classifier = svm.LinearSVC()
#text_classifier = RandomForestClassifier(n_estimators=400)
#text_classifier = MultinomialNB()
text_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,100, 50), max_iter=200)
n = 10
for i in range(0,n):
    X_test = processed_features[i*1000:(i+1)*1000]
    X_train1 = processed_features[0:i*1000]
    X_train2 = processed_features[(i + 1) * 1000:]
    X_train = []
    for ar in X_train1:
        X_train.append(ar)
    for ar in X_train2:
        X_train.append(ar)

    y_train1 = train['overall'].iloc[0:i * 1000]
    y_train2 = train['overall'].iloc[(i + 1) * 1000:]
    y_train = []
    for ar in y_train1:
        y_train.append(ar)
    for ar in y_train2:
        y_train.append(ar)


    y_test = train['overall'].iloc[i*1000:(i+1)*1000]

    text_classifier.fit(X_train, y_train)

    predictions = text_classifier.predict(X_test)

    base = [3 for i in range(0,len(predictions))]

    msqe += mean_squared_error(predictions, y_test.to_numpy())

    ca += classification_accuracy(predictions, y_test.to_numpy())

    mae += np.absolute((predictions - y_test.to_numpy())).mean()

    #rand = [random.randint(1,5) for _ in range(1000)]

print("mse:")
print(msqe/n)
print("mae:")
print(mae/n)
print("ca:")
print(ca/n)