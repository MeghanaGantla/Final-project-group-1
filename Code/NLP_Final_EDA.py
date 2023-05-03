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

# Create a box plot to visualize the distribution of article lengths by target
plt.boxplot([df[df['target'] == 0]['full_text'].str.split().apply(len),
             df[df['target'] == 1]['full_text'].str.split().apply(len)],
            labels=['Fake', 'Real'])
plt.title('Distribution of Article Lengths by Target')
plt.xlabel('Target')
plt.ylabel('Number of Words')
plt.show()

# Create a scatter plot to visualize the relationship between article length and target
plt.scatter(df[df['target'] == 0]['full_text'].str.split().apply(len),
            df[df['target'] == 0]['target'],
            color='red', label='Fake')
plt.scatter(df[df['target'] == 1]['full_text'].str.split().apply(len),
            df[df['target'] == 1]['target'],
            color='blue', label='Real')
plt.title('Relationship between Article Length and Target by Full Text')
plt.xlabel('Number of Words')
plt.ylabel('Target')
plt.legend()
plt.show()
