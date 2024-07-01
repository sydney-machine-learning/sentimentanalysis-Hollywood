import os
import re
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

# 检查和设置设备为CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载预训练的模型和分词器
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10)  # 假设有10个情感标签
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.to(device)  # 将模型移动到CPU或GPU上

# 加载字幕数据并进行预测
def analyze_sentiment(model, tokenizer, text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
    logits = outputs[0]
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()[0]
    return probabilities

def get_text_from_files(folder_path):
    text = ""
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text += file.read()
    return text

def process_movies(base_directory, model, tokenizer):
    results = []
    for root, _, files in os.walk(base_directory):
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                text = get_text_from_files(file_path)
                sentiment_scores = analyze_sentiment(model, tokenizer, text)
                movie_name = os.path.basename(file_path)
                results.append({
                    'Movie': movie_name,
                    'Sentiments': sentiment_scores.tolist()  # 转换为列表以便保存为CSV
                })
    return results

def save_results_to_csv(results, output_file):
    # 假设有10个情感标签
    sentiment_labels = ['Optimistic', 'Thankful', 'Empathetic', 'Pessimistic', 'Anxious', 'Sad', 'Annoyed', 'Denial', 'Official report', 'Joking']
    data = []
    for result in results:
        row = [result['Movie']] + result['Sentiments']
        data.append(row)
    
    df = pd.DataFrame(data, columns=['Movie'] + sentiment_labels)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# 设置字幕文件目录路径
base_directory = '/Users/zjy/Desktop/subtitles'
output_csv = 'sentiment_analysis_results.csv'

results = process_movies(base_directory, model, tokenizer)
print("Movies processed.")
print("Results:", results)  # 打印结果以便调试

save_results_to_csv(results, output_csv)
