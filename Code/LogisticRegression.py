from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

current_dir = os.getcwd()
path = os.path.join(current_dir, 'Data')
data_path = os.path.join(path, 'preprocessed_data.csv')
df = pd.read_csv(data_path)

# drop null values in the data
df.dropna(inplace=True)

# split data to train and test
x = df['clean_text'].values
y = df['target'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6312)

# Vectorize
tfidf = TfidfVectorizer()
tfidf.fit(x_train)
train_tfid = tfidf.transform(x_train)
test_tfid = tfidf.transform(x_test)

# Logistic regression model
clf = LogisticRegression().fit(train_tfid, y_train)
predicts = clf.predict(test_tfid)

# display results
print("====== Logistic regression results ======")
print(f'Accuracy: {accuracy_score(y_test, predicts):.4f}')
print(f'f1-score: {metrics.f1_score(y_test, predicts):.4f}')
print("="*25)
print("Confusion matrix")
cm = confusion_matrix(y_test, predicts)
print(cm)
print("="*25)
print("Classification report")
print(metrics.classification_report(y_test, predicts))

# plot confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()