# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 18:55:30 2022

@author: Asus
"""

# Importing Dataset
import pandas as pd


messages = pd.read_csv('./Datasets/SMSSpamCollection', sep='\t', names=['label', 'message'])

# Preprocessing the Data or Cleaning the text 

import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

corpus = []

stemmer = PorterStemmer()

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [stemmer.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=(5000))
X = cv.fit_transform(corpus).toarray()

# Encoding the labels

y = pd.get_dummies(messages['label'])

y = y.iloc[:, 1].values


# Spliting the dataset in Train and Test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# Taining the model Using Multinomial Naivebayes 
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB().fit(X_train, y_train)


# Predict value using X_test

y_pred = model.predict(X_test)


# Compairing the real test data to predict data

from sklearn.metrics import confusion_matrix
con_m = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score

score = accuracy_score(y_test, y_pred)

