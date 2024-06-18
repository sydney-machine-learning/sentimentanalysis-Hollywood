import os
import re

def remove_hash_tags(content):
    return re.sub(r'\#.*?\#', '', content)

def remove_brackets(content):
    return re.sub(r'\([^()]*\)', '', content)

def remove_music_tags(content):
    return re.sub(r'♪ .*? ♪', '', content)

def remove_html_tags(content):
    return re.sub(r'<i>.*?</i>', '', content)

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    cleaned_txt = remove_hash_tags(txt)
    cleaned_txt = remove_brackets(cleaned_txt)
    cleaned_txt = remove_music_tags(cleaned_txt)
    cleaned_txt = remove_html_tags(cleaned_txt)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_txt)

def process_files(directory):
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            process_file(file_path)
            print(f"Processed file: {file_path}")

output_directory = '/Users/zjy/Desktop/subtitles'

if __name__ == "__main__":
    process_files(output_directory)
    print("All files processed.")

