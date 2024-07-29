import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = BertTokenizer.from_pretrained("GroNLP/hateBERT")
model = BertForSequenceClassification.from_pretrained("GroNLP/hateBERT").to(device)

class SubtitleDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def predict_abusive_language_batch(texts, batch_size=8):
    dataset = SubtitleDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    results = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = [str(text) for text in batch]  
            inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            results.extend(probabilities.cpu().numpy())

    return results


def classify_abusive_language_batch(texts, threshold=0.5, batch_size=8):
    probabilities = predict_abusive_language_batch(texts, batch_size=batch_size)
    classifications = []
    for prob in probabilities:
        abusive_probability = prob[1]
        if abusive_probability > threshold:
            classifications.append(("Abusive Language", abusive_probability))
        else:
            classifications.append(("Non-Abusive Language", abusive_probability))
    return classifications

# Example usage
# text = "Your example text here"
# probabilities = predict_abusive_language(text)
# print(probabilities)

data=pd.read_csv('C:/Users/Admin/Desktop/sentimentanalysis-Hollywood/coding/data_csv.csv')

texts = data['bodyContent'].tolist()
results = classify_abusive_language_batch(texts, batch_size=8)

data['classification'], data['probability'] = zip(*results)

output_path = r'C:\Users\Admin\Desktop\sentimentanalysis-Hollywood\coding\hate_data.csv'
data.to_csv(output_path, index=False)


plt.figure(figsize=(10, 5))


data.groupby('year')['probability'].mean().plot(kind='bar')
plt.title('Average Probability of Abusive Language by Year')
plt.xlabel('Year')
plt.ylabel('Average Probability')
plt.show()


data['is_abusive'] = data['classification'] == 'Abusive Language'
data.groupby('year')['is_abusive'].sum().plot(kind='bar')
plt.title('Number of Abusive Language Instances by Year')
plt.xlabel('Year')
plt.ylabel('Number of Instances')
plt.show()
