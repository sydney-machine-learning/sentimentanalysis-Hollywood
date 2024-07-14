import os
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# nltk.download('punkt')
# nltk.download('stopwords')

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

def get_ngram_counts(text, n, top_n=10):
    tokens = process_text(text)
    ngram_list = list(ngrams(tokens, n))
    ngram_counts = Counter(ngram_list)
    ngram_df = pd.DataFrame(ngram_counts.most_common(top_n), columns=['Ngram', 'Frequency'])
    ngram_df['Ngram'] = ngram_df['Ngram'].apply(lambda x: ' '.join(x))
    return ngram_df

def plot_ngrams(ngram_df, title=None, output_path=None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Frequency', y='Ngram', data=ngram_df, palette='viridis')
    plt.title(title if title else "Top Ngrams", fontsize=20)
    plt.xlabel("Frequency", fontsize=20)
    plt.ylabel("")
    plt.xticks(fontsize=20)
    plt.yticks(rotation=45, fontsize=20)  # Rotate y-ticks and adjust fontsize
    plt.tight_layout()
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

    bigram_title = f"Top 10 Bigrams ({decade[0]}-{decade[1]})"
    trigram_title = f"Top 10 Trigrams ({decade[0]}-{decade[1]})"
    
    bigram_output_path = os.path.join(output_directory, f"{decade[0]}-{decade[1]}_bigrams.png")
    trigram_output_path = os.path.join(output_directory, f"{decade[0]}-{decade[1]}_trigrams.png")

    plot_ngrams(bigram_df, title=bigram_title, output_path=bigram_output_path)
    plot_ngrams(trigram_df, title=trigram_title, output_path=trigram_output_path)
