import nltk
import re
import string
import pandas as pd
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os

current_dir = os.getcwd()
path = os.path.join(current_dir, 'Data')
data_path = os.path.join(path, 'Full_data.csv')

df = pd.read_csv(data_path)
print(df.head())

df['full_text'] = df['title'] + ' ' + df['text']
df.drop(['title', 'text', 'date'], axis=1, inplace=True)


# Define preprocessing functions
def urls(text):
    # Removing URLS
    return re.sub(r'http\S+', '', text)


df['clean_text'] = df['clean_text'].apply(urls)


def to_lower(text):
    # Apply lower casing
    return text.lower()


df['clean_text'] = df['clean_text'].apply(to_lower)


def remove_contractions(text):
    # removing contractions
    return ' '.join([contractions.fix(word) for word in text.split()])


df['clean_text'] = df['clean_text'].apply(remove_contractions)


def remove_punctuation(text):
    # Remove punctuations
    return text.translate(str.maketrans('', '', string.punctuation))


df['clean_text'] = df['clean_text'].apply(remove_punctuation)


def remove_characters(text):
    # remove special characters
    return re.sub('[^a-zA-Z]', ' ', text)


df['clean_text'] = df['clean_text'].apply(remove_characters)


def remove_stopwords(text):
    # Remove stopwords
    stop_words = stopwords.words('english')
    return ' '.join([word for word in nltk.word_tokenize(text) if word not in stop_words])


df['clean_text'] = df['clean_text'].apply(remove_stopwords)


def lemmatize(text):
    # Apply lemmatizing
    lem = WordNetLemmatizer()
    return ' '.join(lem.lemmatize(word) for word in text.split())


df['clean_text'] = df['clean_text'].apply(lemmatize)


data_path = os.path.join(path, 'preprocessed_data.csv')
df.to_csv(data_path, index=False)
