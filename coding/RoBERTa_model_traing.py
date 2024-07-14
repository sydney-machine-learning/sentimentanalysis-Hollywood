from transformers import BertTokenizer
import torch
import numpy as np
import pandas as pd
import os
os.environ["PYTORCH_ENABLE_FLASH_ATTN"] = "1"

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_data=pd.read_table("C:/Users/Admin/Desktop/sentimentanalysis-Hollywood/train_data/cad_v1_1_train.tsv")
test_data=pd.read_table("C:/Users/Admin/Desktop/sentimentanalysis-Hollywood/train_data/cad_v1_1_test.tsv")

print(train_data.head)

train_data = train_data.dropna(subset=['text', 'labels'])
test_data = test_data.dropna(subset=['text', 'labels'])

train_unique_labels = train_data['labels'].unique()
test_unique_labels = test_data['labels'].unique()


unique_labels = list(set(train_unique_labels) | set(test_unique_labels))

label_to_id = {label: idx for idx, label in enumerate(unique_labels)}

train_labels = [label_to_id[label] for label in train_data['labels']]
test_labels = [label_to_id[label] for label in test_data['labels']]



# Function to tokenize the text data
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


# Tokenize training and testing data
train_inputs, train_masks, train_labels = tokenize_data(train_data['text'].values, train_labels, tokenizer)
test_inputs, test_masks, test_labels = tokenize_data(test_data['text'].values, test_labels, tokenizer)


from transformers import BertForSequenceClassification, AdamW

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(unique_labels),  # 
    output_attentions=False,
    output_hidden_states=False
)


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

# Create the DataLoader for our validation set
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)


from transformers import get_linear_schedule_with_warmup
import torch

# Set the seed value all over the place to make this reproducible
seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 4
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
for epoch_i in range(0, epochs):
    print(f'Epoch {epoch_i + 1}/{epochs}')
    print('Training...')

    total_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device)
        b_labels = b_labels.to(device)

        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Average training loss: {avg_train_loss}')


print('Evaluating...')

model.eval()
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

for batch in test_dataloader:
    b_input_ids, b_input_mask, b_labels = batch
    b_input_ids = b_input_ids.to(device)
    b_input_mask = b_input_mask.to(device)
    b_labels = b_labels.to(device)

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs.logits
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    eval_accuracy += np.sum(np.argmax(logits, axis=1) == label_ids)
    nb_eval_examples += b_input_ids.size(0)
    nb_eval_steps += 1

print(f'Accuracy: {eval_accuracy / nb_eval_examples}')
