import pandas as pd

df=pd.read_csv("D:\spam-classifier\data\spam.csv")

df = df[['Category', 'Messages']]
df.columns = ['label', 'message']

print(df.head())
print("\n\n\n")
print(df.columns)
print("\n\n\n")
df = df.dropna(subset=['message'])

#convert labels to numbers
df['label'] = df['label'].map({'ham':0,'spam':1})
print(df.head())
print("\n\n\n")

#Train test split
from sklearn.model_selection import train_test_split

x=df['message']
y=df['label']
X_train, X_test, Y_train, Y_test = train_test_split(
    x,y, test_size=0.2, random_state=42
)

print("Train size", len(X_train))
print("Test size", len(X_test))
print("\n\n\n")

"""Tf-Idf Term Frequency-Inverse Document Frequency . 
It boosts rare, significant words and penalizes common words (like "the") that appear 
across many documents.""" 
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#train_model
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_tfidf, Y_train)

#prediction
Y_pred = model.predict(X_test_tfidf)

#Accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy : ", accuracy)


def predict_spam(text):
    text_tfidf = vectorizer.transform([text])
    result = model.predict(text_tfidf)[0]

    return "spam" if result == 1 else "not spam"

print(predict_spam("you won a free lottery"))
print(predict_spam("Hey, are we meeting tomorrow?"))
print("\n\n\n")

#confusion matrx
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y_test, Y_pred)
print(cm)
print("\n\n\n")

#classification report
from sklearn.metrics import classification_report

print(classification_report(Y_test, Y_pred))

#logistic regeression model
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, Y_train)

Y_pred_lr = lr_model.predict(X_test_tfidf)

print("Logistic Regression Accuracy:", accuracy_score(Y_test, Y_pred_lr))

#Saving model
import pickle

pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

#load_model
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))