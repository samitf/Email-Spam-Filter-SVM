#       C:\\Users\\samit\\OneDrive\\Documents\\SFIT\\mini-project\\mail_data.csv

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data from csv file
df = pd.read_csv('mail_data.csv')

# Encode the labels
df['Category'] = np.where(df['Category']=='spam', 1, 0)

# Vectorize the text data
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Message'])
y = df['Category']

# Train and test a Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X, y)

# Train and test a KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X, y)

# Train and test an SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X, y)

# Train and test a Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X, y)

# Take an input text and classify it using all four models
input_text = input("Enter a text message to classify: ")
input_vector = vectorizer.transform([input_text])
nb_pred = nb_model.predict(input_vector)[0]
knn_pred = knn_model.predict(input_vector)[0]
svm_pred = svm_model.predict(input_vector)[0]
lr_pred = lr_model.predict(input_vector)[0]

# Print the classification result and accuracy score for each model
print("Input Text: {}".format(input_text))
print("Naive Bayes Classification: {}".format("spam" if nb_pred == 1 else "ham"))
print("Naive Bayes Accuracy: {:.2f}%".format(accuracy_score(y, nb_model.predict(X)) * 100))
print("KNN Classification: {}".format("spam" if knn_pred == 1 else "ham"))
print("KNN Accuracy: {:.2f}%".format(accuracy_score(y, knn_model.predict(X)) * 100))
print("SVM Classification: {}".format("spam" if svm_pred == 1 else "ham"))
print("SVM Accuracy: {:.2f}%".format(accuracy_score(y, svm_model.predict(X)) * 100))
print("Logistic Regression Classification: {}".format("spam" if lr_pred == 1 else "ham"))
print("Logistic Regression Accuracy: {:.2f}%".format(accuracy_score(y, lr_model.predict(X)) * 100))
