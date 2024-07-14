from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="1Torch was not compiled with flash attention")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_data = pd.read_table("C:/Users/Admin/Desktop/sentimentanalysis-Hollywood/train_data/cad_v1_1_train.tsv")
test_data = pd.read_table("C:/Users/Admin/Desktop/sentimentanalysis-Hollywood/train_data/cad_v1_1_test.tsv")

# remove Na
train_data = train_data.dropna(subset=['text', 'labels'])
test_data = test_data.dropna(subset=['text', 'labels'])

# get unique label
train_unique_labels = train_data['labels'].unique()
test_unique_labels = test_data['labels'].unique()
unique_labels = list(set(train_unique_labels) | set(test_unique_labels))

# map label to id
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}


# transform label to id
train_labels = [label_to_id[label] for label in train_data['labels']]
test_labels = [label_to_id[label] for label in test_data['labels']]

def tokenize_data(texts, labels, tokenizer, max_length=128):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels


# Tokenize train and test data
train_inputs, train_masks, train_labels = tokenize_data(train_data['text'].values, train_labels, tokenizer)
test_inputs, test_masks, test_labels = tokenize_data(test_data['text'].values, test_labels, tokenizer)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)


def evaluate_model(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []

    for batch in dataloader:
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device)
        b_labels = b_labels.to(device)

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
        logits = outputs.logits
        predictions.extend(np.argmax(logits.detach().cpu().numpy(), axis=1))
        true_labels.extend(b_labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1, predictions, true_labels

def analyze_errors(predictions, true_labels, texts):
    errors = []
    for i in range(len(predictions)):
        if predictions[i] != true_labels[i]:
            errors.append((texts[i], true_labels[i], predictions[i]))
    return errors



loaded_model = BertForSequenceClassification.from_pretrained('C:/Users/Admin/Desktop/sentimentanalysis-Hollywood/coding/model/newBERTa_model')
loaded_tokenizer = BertTokenizer.from_pretrained('C:/Users/Admin/Desktop/sentimentanalysis-Hollywood/coding/model/newBERTa_tokenizer')
loaded_model.to(device)

accuracy, precision, recall, f1, predictions, true_labels = evaluate_model(loaded_model, test_dataloader, device)
print(f'Test Accuracy: {accuracy}')
print(f'Test Precision: {precision}')
print(f'Test Recall: {recall}')
print(f'Test F1 Score: {f1}')

texts = test_data['text'].values  # original Pandas DataFrame
errors = analyze_errors(predictions, true_labels, texts)
print(f'Number of errors: {len(errors)}')
for error in errors[:10]:  # print 10 errors
    print(f'Text: {error[0]}, True Label: {error[1]}, Predicted Label: {error[2]}')