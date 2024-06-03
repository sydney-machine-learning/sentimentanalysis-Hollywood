def read_srt_file(file_path):
    subtitles = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        subtitle = None
        for line in lines:
            line = line.strip()
            if line.isdigit():  # 判断是否为字幕序号
                if subtitle is not None:
                    subtitles.append(subtitle)
                subtitle = {'index': int(line), 'text': ''}
            elif '-->' in line:  # 判断是否为时间轴
                pass  # 在这个简单示例中，我们不需要时间轴
            elif line:  # 判断是否为字幕文本
                if subtitle is not None:
                    subtitle['text'] += line + ' '
    
    return subtitles

# 示例用法
file_path = '2000-2024/2000/Cast Away.srt'  # 替换为你的SRT文件路径
subtitles = read_srt_file(file_path)

# 打印所有字幕文本
for subtitle in subtitles:
    print(subtitle['text'])

