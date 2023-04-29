import nltk
import re
import string
import pandas as pd
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import os


current_dir = os.getcwd()
path = os.path.join(current_dir, 'Data')
data_path = os.path.join(path, 'Full_data.csv')

df = pd.read_csv(data_path)
print(df.head())

df['full_text'] = df['title'] + ' ' + df['text']
df.drop(['title', 'text'], axis=1, inplace=True)


# Define preprocessing functions
def urls(text):
    # Removing URLS
    return re.sub(r'http\S+', '', text)


def to_lower(text):
    # Apply lower casing
    return text.lower()


def remove_contractions(text):
    # removing contractions
    return ' '.join([contractions.fix(word) for word in text.split()])


def remove_punctuation(text):
    # Remove punctuations
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_characters(text):
    # remove special characters
    return re.sub('[^a-zA-Z]', ' ', text)


def remove_stopwords(text):
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)


def stemming_words(text):
    # Apply stemming
    stemmer = PorterStemmer()
    return ' '.join(stemmer.stem(word) for word in text.split())


def lemmatize(text):
    # Apply lemmatizing
    lem = WordNetLemmatizer()
    return ' '.join(lem.lemmatize(word) for word in text.split())

def remove_numbers(text):
    """
    Remove numbers from text using regular expressions.
    """
    return re.sub(r'\d+', '', text)

def preprocess_text(text):
    # Apply all preprocess together
    text1 = urls(text)
    text2 = to_lower(text1)
    text3 = remove_contractions(text2)
    text4 = remove_punctuation(text3)
    text5 = remove_characters(text4)
    text6 = remove_stopwords(text5)
    #text7 = stemming_words(text6)
    text8 = lemmatize(text6)
    text9 = remove_numbers(text8)
    return text9


data_path = os.path.join(path, 'preprocessed_data.csv')
df['full_text'] = df['full_text'].apply(preprocess_text)
df['full_text'] = df['full_text'].astype(str)
df.to_csv(data_path, index=False)
