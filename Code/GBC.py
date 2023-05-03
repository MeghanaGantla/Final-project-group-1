import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import os


current_dir = os.getcwd()
path = os.path.join(current_dir, 'Data')
data_path = os.path.join(path, 'preprocessed_data.csv')
df = pd.read_csv(data_path)

df.isna().sum()

df['clean_text'] = df['clean_text'].astype(str)
X = df['clean_text'].fillna('')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
tfidf = TfidfVectorizer()
tfidf.fit(X_train)
X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train a Gradient Boosting Classifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = gbc.predict(X_test_tfidf)

# Evaluate the model performance
print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1 score', f1_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))




