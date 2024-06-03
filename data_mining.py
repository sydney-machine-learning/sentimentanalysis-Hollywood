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
    
    return subtitles

file_path = '2000-2024/2000/Cast Away.srt'  
subtitles = read_srt_file(file_path)

for subtitle in subtitles:
    print(subtitle['text'])

