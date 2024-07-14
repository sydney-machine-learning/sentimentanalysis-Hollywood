import pandas as pd
import re
import os
from collections import Counter

# Define the list of common words to exclude
common_exclusions = {'-','♪','i', 'you', 'to', 'the', 'a', 'and', 'it', 'is', 'that', 'of','s', 't', 'what', 'in', 'me', 'this', 'on', 'sir', 'get','for', 'she', 'be', 'eve', 'not', 'have', 'all', 'her', 'was', 'my','can', 'oh', 'no', 'we', 'well', 'annie', 'be', 'he', 'like', 'don'}

def parse_srt_excluding_common(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Extract all timestamps
    timestamps = re.findall(r'\d{2}:\d{2}:\d{2},\d{3}', content)
    if timestamps:
        last_timestamp = timestamps[-1]
        hours, minutes, seconds_milliseconds = last_timestamp.split(':')
        seconds, milliseconds = seconds_milliseconds.split(',')
        total_minutes = int(hours) * 60 + int(minutes) + int(seconds) / 60 + int(milliseconds) / (60 * 1000)
    else:
        total_minutes = 0

    # Remove timestamps and numbers
    lines = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', content)
    lines = re.sub(r'\d+', '', lines)
    lines = lines.replace('\n', '')
    lines = re.sub(r'\n\s*\n', '\n', lines).strip()
    lines = lines.replace('-', '')
    lines = lines.replace('♪', '')
    lines = re.sub(r'</?i>', '', lines)
    lines = re.sub(r'</?b>', '', lines)

    # Extract words
    words = re.findall(r'\b\w+\b', lines.lower())
    word_count = len(words)

    # Filter out common exclusions
    filtered_words = [word for word in words if word not in common_exclusions]

    # Get the most common words
    common_words = Counter(filtered_words).most_common(10)
    top_ten_words = [word for word, _ in common_words]

    return word_count, total_minutes, top_ten_words,lines

def process_srt_files_excluding_common(directory):
    data = []
    for year_folder in sorted(os.listdir(directory)):
        year_path = os.path.join(directory, year_folder)
        if os.path.isdir(year_path):
            for srt_file in sorted(os.listdir(year_path)):
                if srt_file.endswith('.srt'):
                    file_path = os.path.join(year_path, srt_file)
                    movie_name = re.sub(r'[^\w\s]', '', srt_file).replace(' ', '')
                    word_count, total_minutes, top_ten_words , content = parse_srt_excluding_common(file_path)
                    data.append({
                        'movie': movie_name,
                        'year': year_folder,
                        'numberofwords': word_count,
                        'time': f"{total_minutes:.2f}mins",
                        'toptenwords': top_ten_words,
                        'bodyContent': content
                    })
    return data

# Process the SRT files and gather the data
srt_data_excluding_common = process_srt_files_excluding_common(os.path.join('C:/Users/Admin/Desktop/sentimentanalysis-Hollywood/data'))

# Convert to DataFrame and save as CSV
df = pd.DataFrame(srt_data_excluding_common)
df.to_csv("C:/Users/Admin/Desktop/data_csv.csv", index=False)


