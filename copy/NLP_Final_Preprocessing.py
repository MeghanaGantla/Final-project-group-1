import nltk
import re
import string
#import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Define preprocessing functions
def urls(text):
    '''Remove URLs from a text string'''
    return re.sub(r'http\S+', '', text)

def punctuation(text):
    '''Remove punctuation from a text string'''
    return text.translate(str.maketrans('', '', string.punctuation))

def stopwords(text):
    '''Remove stopwords from a text string'''
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def lemmatize(text):
    '''Lemmatize words in a text string'''
    lem = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lem.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def preprocess_text(text):
    '''Apply all preprocessing steps to a text string'''
    text = urls(text)
    text = punctuation(text)
    text = stopwords(text)
    text = lemmatize(text)
    return text
