import os
import re

def read_srt_file(file_path):
    subtitles = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        subtitle = None
        for line in lines:
            line = line.strip()
            if line.isdigit():  
                if subtitle is not None:
                    subtitles.append(subtitle)
                subtitle = {'index': int(line), 'text': ''}
            elif '-->' in line:  
                pass
            elif line: 
                if subtitle is not None:
                    subtitle['text'] += line + ' '
    
    if subtitle is not None:
        subtitles.append(subtitle)
    
    return subtitles

def new_directory(directory_path, output_directory_path):
    all_subtitles = []
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.srt'):
                file_path = os.path.join(root, file)
                
                # Extract year from file path
                year_match = re.search(r'(\d{4})', file_path)
                if year_match:
                    year = year_match.group(1)
                else:
                    year = 'Unknown'
                
                # Create year folder if it doesn't exist
                year_folder = os.path.join(output_directory_path, year)
                if not os.path.exists(year_folder):
                    os.makedirs(year_folder)
                
                # Read and process srt file
                subtitles = read_srt_file(file_path)
                all_subtitles.extend(subtitles)
                
                # Save the output of each file into a new file respectively
                output_file_path = os.path.join(year_folder, f"{os.path.splitext(file)[0]}_processed.txt")
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    for subtitle in subtitles:
                        f.write(subtitle['text'] + '\n')
                print(f"Processed and saved: {output_file_path}")
    
    return all_subtitles

directory_path = '/Users/zjy/Desktop/1950-2024'
output_directory_path = os.path.expanduser('~/Desktop/output_subtitles')

if not os.path.exists(output_directory_path):
    os.makedirs(output_directory_path)

all_subtitles = new_directory(directory_path, output_directory_path)

print(f"Subtitles are processed and saved to {output_directory_path}")
