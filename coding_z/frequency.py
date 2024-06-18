import os
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import matplotlib.pyplot as plt
from collections import Counter

def process_text(text):
    # Filter Words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return filtered_tokens

def plot_ngrams(ngram_counts, n, top_n=20, title=None):
    # High Frequency Words
    top_ngrams = ngram_counts.most_common(top_n)
    ngrams, counts = zip(*top_ngrams)
    
    # Bar Chart
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(ngrams)), counts, tick_label=[' '.join(ngram) for ngram in ngrams])
    plt.xticks(rotation=90)
    plt.title(title if title else f"Top {top_n} {n}-grams")
    plt.xlabel(f"{n}-grams")
    plt.ylabel("Frequency")
    plt.show()

# Process files by decade
decades = [(1950, 1959), (1960, 1969), (1970, 1979), (1980, 1989), (1990, 1999),
           (2000, 2009), (2010, 2019), (2020, 2024)]

for decade in decades:
    decade_text = ""
    for year in range(decade[0], decade[1] + 1):
        year_folder_path = f"/Users/zjy/Desktop/subtitles/{year}"
        for root, _, files in os.walk(year_folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    year_text = file.read()
                    decade_text += year_text

    tokens = process_text(decade_text)
    bigrams = list(nltk.ngrams(tokens, 2))
    trigrams = list(nltk.ngrams(tokens, 3))

    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)

    plot_ngrams(bigram_counts, 2, title=f"Top 20 Bigrams ({decade[0]}-{decade[1]})")
    plot_ngrams(trigram_counts, 3, title=f"Top 20 Trigrams ({decade[0]}-{decade[1]})")
