import os

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
    
    for year_folder in os.listdir(directory_path):
        year_folder_path = os.path.join(directory_path, year_folder)
        if not os.path.isdir(year_folder_path):
            continue
        
        for file_name in os.listdir(year_folder_path):
            file_path = os.path.join(year_folder_path, file_name)
            if not file_name.endswith('.srt'):
                continue
            
            subtitles = read_srt_file(file_path)
            all_subtitles.extend(subtitles)
            
            # Extract year from file path
            year = int(year_folder)
            
            # Create year folder if it doesn't exist
            year_folder_output = os.path.join(output_directory_path, str(year))
            if not os.path.exists(year_folder_output):
                os.makedirs(year_folder_output)
            
            # Save the output of each file into a new file respectively
            output_file_path = os.path.join(year_folder_output, f"{os.path.splitext(file_name)[0]}.txt")
            with open(output_file_path, 'w', encoding='utf-8') as f:
                for subtitle in subtitles:
                    f.write(subtitle['text'] + '\n')
            print(f"Processed and saved: {output_file_path}")
    
    return all_subtitles

directory_path = '/Users/zjy/Desktop/1950-2024'
output_directory_path = os.path.expanduser('~/Desktop/subtitles')

if not os.path.exists(output_directory_path):
    os.makedirs(output_directory_path)

all_subtitles = new_directory(directory_path, output_directory_path)

print(f"Subtitles are processed and saved to {output_directory_path}")
