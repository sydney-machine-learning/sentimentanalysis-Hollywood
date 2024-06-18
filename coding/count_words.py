import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import csv


def count_words(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        words = nltk.word_tokenize(text)
        return len(words)
    except Exception as e:
        print(f"Error {file_path}: {e}")
        return 0

def count_words_directory(directory_path, output_csv_path):
    total_words = 0
    results = []

    for root, _, files in os.walk(directory_path):
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                word_count = count_words(file_path)
                total_words += word_count
                results.append([file_name, word_count])
                print(f"{file_name}: {word_count} words")

    #store data to csv
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['File Name', 'Word Count'])
            writer.writerows(results)
        print(f"Word counts are stored to {output_csv_path}")
    except Exception as e:
        print(f"Error {output_csv_path}: {e}")

    return total_words

directory_path = '/Users/zjy/Desktop/subtitles'
output_csv_path = '/Users/zjy/Desktop/coding/words_counted.csv'
total_words = count_words_directory(directory_path, output_csv_path)

print(f"Total number of words in all subtitle files: {total_words}")
