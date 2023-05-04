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


df['clean_text'] = df['full_text'].apply(urls)


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

#EDA

import pandas as pd
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

current_dir = os.getcwd()
path = os.path.join(current_dir, 'Data')
data_path = os.path.join(path, 'preprocessed_data.csv')
df = pd.read_csv(data_path)

df['clean_text'] = df['clean_text'].astype(str)

# Create a bar chart to visualize the distribution of fake and real news articles
counts = df['target'].value_counts()
labels = ['Real', 'Fake']
plt.bar(labels, counts, color=['blue', 'red'])
plt.title('Distribution of Real and Fake News Articles')
plt.xlabel('Category')
plt.ylabel('Number of Articles')
plt.show()

# Create a histogram to visualize the distribution of article lengths
article_lengths = df['full_text'].str.len()
plt.hist(article_lengths, bins=20, alpha=0.5)
plt.title('Distribution of Article Lengths')
plt.xlabel('Article Length')
plt.ylabel('Frequency')
plt.show()

# Separate the real and fake news articles
fake_df = df[df['target'] == 0]
real_df = df[df['target'] == 1]

# Create a stacked bar chart to visualize the distribution of articles by category for fake and real news
fake_subject_counts = fake_df['subject'].value_counts()
real_subject_counts = real_df['subject'].value_counts()
subjects = list(set(fake_subject_counts.index) | set(real_subject_counts.index))
fake_subject_counts = fake_subject_counts.reindex(subjects, fill_value=0)
real_subject_counts = real_subject_counts.reindex(subjects, fill_value=0)

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(subjects, fake_subject_counts, label='Fake', color='red')
ax.bar(subjects, real_subject_counts, bottom=fake_subject_counts, label='Real', color='blue')
ax.set_title('Number of Articles per Category for Real and Fake News')
ax.set_xlabel('Category')
ax.set_ylabel('Number of Articles')
ax.legend()
plt.xticks(rotation=90)
plt.show()

#GradientBoostingClassifier

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


#RoBERTa

import sklearn
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

import os

current_dir = os.getcwd()
path = os.path.join(current_dir, 'Data')
data_path = os.path.join(path, 'preprocessed_data.csv')
df = pd.read_csv(data_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device {}.\n".format(device))

df['text'] = df['clean_text']
df.drop(['subject', 'full_text', 'clean_text'], axis=1, inplace=True)
# drop null values in the data
df.dropna(inplace=True)

training, sub = train_test_split(df, stratify=df.target.values, random_state=6312, test_size=0.2, shuffle=True)
validation, test = train_test_split(sub, stratify=sub.target.values, random_state=6312, test_size=0.25, shuffle=True)

training.reset_index(drop=True, inplace=True)
validation.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# distilbert
num_labels = 2
batch_size = 32
max_len = 256
num_epoch = 3
lr = 1e-5
check = "roberta-base"
tokenize = RobertaTokenizer.from_pretrained(check, do_lower_case=True)

train = tokenize(list(training.text.values), padding=True, max_length=max_len, truncation=True)
t_input = train['input_ids']
t_torch_inputs = torch.tensor(t_input)
t_mask = train['attention_mask']
t_torch_mask = torch.tensor(t_mask)
t_torch_labels = torch.tensor(training.target.values)
t_data = TensorDataset(t_torch_inputs, t_torch_mask, t_torch_labels)
t_sampler = RandomSampler(t_data)
t_dataloader = DataLoader(t_data, sampler=t_sampler, batch_size=batch_size)

validate = tokenize(list(validation.text.values), truncation=True, padding=True, max_length=max_len)
v_input = validate['input_ids']
v_torch_inputs = torch.tensor(v_input)
v_masks = validate['attention_mask']
v_torch_masks = torch.tensor(v_masks)
v_torch_labels = torch.tensor(validation.target.values)
v_data = TensorDataset(v_torch_inputs, v_torch_masks, v_torch_labels)
v_sampler = SequentialSampler(v_data)
v_dataloader = DataLoader(v_data, sampler=v_sampler, batch_size=batch_size)


def parameter_count(given_model):
    return sum(p.numel() for p in given_model.parameters() if p.requires_grad)


def metric_evaluation(predicts, labels):
    predictions = predicts.argmax(axis=1, keepdim=True)
    accuracy = accuracy_score(y_true=labels.to('cpu').tolist(), y_pred=predictions.detach().cpu().numpy())
    return accuracy


def training(model, dataloader, optimizer, device, scheduler):
    model.train()
    train_loss, train_acc = 0.0, 0.0

    for data in dataloader:
        inputs, masks, labels = [d.to(device) for d in data]
        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        train_acc += metric_evaluation(outputs.logits, labels)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def validate(model, dataloader, device):
    model.eval()
    valid_loss, valid_acc = 0.0, 0.0

    with torch.no_grad():
        for data in dataloader:
            inputs, masks, labels = [d.to(device) for d in data]
            outputs = model(inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss
            valid_loss += loss.item()
            logits = outputs.logits
            valid_acc += metric_evaluation(logits, labels)

    valid_loss /= len(dataloader)
    valid_acc /= len(dataloader)

    return valid_loss, valid_acc


model = RobertaForSequenceClassification.from_pretrained(check, num_labels=num_labels, output_hidden_states=False,
                                                            output_attentions=False)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=lr)
total_steps = len(t_dataloader) * num_epoch
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
num_params = parameter_count(model)
print("trainable parameters: {}".format(num_params))


train_losses, validation_losses, train_accuracies, validation_accuracies = [], [], [], []
best_loss = float('inf')

for epoch in range(num_epoch):
    epoch_t_loss, epoch_t_acc = training(model, t_dataloader, optimizer, device, scheduler)
    epoch_v_loss, epoch_v_acc = validate(model, v_dataloader, device)
    train_losses.append(epoch_t_loss)
    validation_losses.append(epoch_v_loss)
    train_accuracies.append(epoch_t_acc)
    validation_accuracies.append(epoch_v_acc)
    if epoch_v_loss < best_loss:
        best_loss = epoch_v_loss
        torch.save(model.state_dict(), 'model.pt')

    print(f"Epoch: {epoch}, Train Loss: {epoch_t_loss:.4f}, Train Accuracy: {epoch_t_acc:.2f}%")
    print(f"Epoch: {epoch}, Validation Loss: {epoch_v_loss:.4f}, Validation Accuracy: {epoch_v_acc:.2f}%\n")


# Testing
model.load_state_dict(torch.load('model.pt'))
test = tokenize(list(test[:1000].text.values), max_length=30, truncation=True, padding=True)
test_input = test['input_ids']
test_tensor_input = torch.tensor(test_input)
test_masks = test['attention_mask']
test_torch_masks = torch.tensor(test_masks)

with torch.no_grad():
    test_tensor_input = test_tensor_input.to(device)
    test_masks = test_masks.to(device)
    outputs = model(test_tensor_input, test_masks)
    logits = outputs.logits
    batch_logits = logits.detach().cpu().numpy()
    predict = np.argmax(batch_logits, axis=1)


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
axes[0, 0].plot(np.arange(1, 4), validation_losses)
axes[0, 0].set_title('Loss vs Epochs - Validation')
axes[0, 0].set_xlabel('Epochs')
axes[0, 0].set_ylabel('Loss')
axes[0, 1].plot(np.arange(1, 4), train_losses)
axes[0, 1].set_title('Loss vs Epochs - Train')
axes[0, 1].set_xlabel('Epochs')
axes[0, 1].set_ylabel('Loss')
axes[1, 0].plot(np.arange(1, 4), train_accuracies)
axes[1, 0].set_title('Accuracy vs Epochs - Train')
axes[1, 0].set_xlabel('Epochs')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 1].plot(np.arange(1, 4), validation_accuracies)
axes[1, 1].set_title('Accuracy vs Epochs - Validation')
axes[1, 1].set_xlabel('Epochs')
axes[1, 1].set_ylabel('Accuracy')
plt.tight_layout()
plt.show()
print("ROC AUC Score: {}".format(roc_auc_score(y_true=test[:1000].target.values, y_score=predict)))
print('f1-score: ', sklearn.metrics.f1_score(test[:1000].target.values, predict))
print(classification_report(test[:1000].target.values, predict))
