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


# Read File
file_path = '/Users/zjy/Desktop/output_subtitles/48 Hrs.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    subtitle_text = file.read()


tokens = process_text(subtitle_text)
bigrams = list(nltk.ngrams(tokens, 2))
trigrams = list(nltk.ngrams(tokens, 3))


bigram_counts = Counter(bigrams)
trigram_counts = Counter(trigrams)


plot_ngrams(bigram_counts, 2, title="Top 20 Bigrams")
plot_ngrams(trigram_counts, 3, title="Top 20 Trigrams")

