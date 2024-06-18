import os
import re

def remove_brackets(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(file_path, 'w', encoding='utf-8') as f:
        for line in lines:
            cleaned_line = re.sub(r'\#.*?\#', '', line)
            f.write(cleaned_line)

output_directory = '/Users/zjy/Desktop/output_subtitles'

for root, _, files in os.walk(output_directory):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        remove_brackets(file_path)
