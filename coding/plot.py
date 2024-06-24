import os
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def process_text(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return filtered_tokens


def get_text_from_files(folder_path):
    text = ""
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                text += file.read()
    return text


def get_ngram_counts(text, n, top_n=20):
    tokens = process_text(text)
    ngram_list = list(ngrams(tokens, n))
    ngram_counts = Counter(ngram_list)
    ngram_df = pd.DataFrame(ngram_counts.most_common(top_n), columns=['Ngram', 'Frequency'])
    ngram_df['Ngram'] = ngram_df['Ngram'].apply(lambda x: ' '.join(x))
    return ngram_df


def plot_ngrams(bigram_df, trigram_df, title=None, output_path=None):
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Bigram plot
    sns.barplot(ax=axes[0], x='Frequency', y='Ngram', data=bigram_df, palette='viridis')
    axes[0].set_title("Top 20 Bigrams", fontsize=16)
    axes[0].set_xlabel("Frequency", fontsize=14)
    axes[0].set_ylabel("")
    axes[0].tick_params(axis='both', which='major', labelsize=12)
    axes[0].text(0.95, 0.01, 'Bigram', verticalalignment='bottom', horizontalalignment='right',
                 transform=axes[0].transAxes, color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

    # Trigram plot
    sns.barplot(ax=axes[1], x='Frequency', y='Ngram', data=trigram_df, palette='viridis')
    axes[1].set_title("Top 20 Trigrams", fontsize=16)
    axes[1].set_xlabel("Frequency", fontsize=14)
    axes[1].set_ylabel("")
    axes[1].tick_params(axis='both', which='major', labelsize=12)
    axes[1].text(0.95, 0.01, 'Trigram', verticalalignment='bottom', horizontalalignment='right',
                 transform=axes[1].transAxes, color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.6))
    
    plt.suptitle(title if title else "Top Ngrams", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, wspace=0.3)
    if output_path:
        plt.savefig(output_path)
    plt.show()


decades = [(1950, 1959), (1960, 1969), (1970, 1979), (1980, 1989), (1990, 1999),
           (2000, 2009), (2010, 2019), (2020, 2024)]


output_directory = '/Users/zjy/Desktop/graphs'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


for decade in decades:
    decade_text = ""
    for year in range(decade[0], decade[1] + 1):
        year_folder_path = f"/Users/zjy/Desktop/subtitles/{year}"
        decade_text += get_text_from_files(year_folder_path)

    bigram_df = get_ngram_counts(decade_text, 2)
    trigram_df = get_ngram_counts(decade_text, 3)

    title = f"Top 20 Bigrams and Trigrams ({decade[0]}-{decade[1]})"
    output_path = os.path.join(output_directory, f"{decade[0]}-{decade[1]}.png")

    plot_ngrams(bigram_df, trigram_df, title=title, output_path=output_path)
