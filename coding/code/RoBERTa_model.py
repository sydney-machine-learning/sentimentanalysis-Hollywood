import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np
import torch
from scipy.special import softmax
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import RobertaModel
import torch.nn as nn
import roberta_new
from transformers import RobertaModel

MAX_LEN = 200  # Based on the length of movies
TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 1
EPOCHS = 4
LEARNING_RATE = 2e-05
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')  # Use RoBERTa tokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RoBERTaCustom(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.movie = dataframe['movie']
        self.targets = self.dataframe.list
        self.max_len = max_len

    def __len__(self):
        return len(self.movie)

    def __getitem__(self, index):
        movie = str(self.movie[index])
        movie = " ".join(movie.split())

        inputs = self.tokenizer.encode_plus(
            movie,
            None,
            add_special_tokens=True,  # Add special tokens for RoBERTa
            max_length=self.max_len,
            padding='max_length',  # Pad to max_length
            return_token_type_ids=True,
            return_tensors='pt',  # Return PyTorch tensors
            truncation=True  # Truncate sequences longer than max_length
        )
        input_ids = inputs['input_ids'].squeeze(0)  # Remove the added batch dimension
        attention_mask = inputs['attention_mask'].squeeze(0)  # Remove the added batch dimension

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


roberta = torch.load("C:/Users/Admin/Desktop/sentimentanalysis-Hollywood/.venv/Lib/site-packages/torch/roberta-finetuned.pth")
roberta 

mdf=pd.read_csv("C:/Users/Admin/Desktop/data_csv.csv")


roberta_df = pd.DataFrame()
roberta_df['movie'] = mdf['bodyContent']

values = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 1027
roberta_df['list'] = values


test_dataset = CustomDataset(roberta_df, tokenizer, MAX_LEN)

roberta_test_params = {'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 0
                    }

test_loader = DataLoader(test_dataset, **roberta_test_params)

def test():
    roberta.eval()
    roberta_outputs = []

    with torch.no_grad():
        for unw, data in enumerate(test_loader, 0):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            # targets = data['targets'].to(device, dtype=torch.float)

            outputs = roberta(input_ids=input_ids, attention_mask=attention_mask)

            roberta_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    return roberta_outputs

test_outputs = test()
test_outputs = np.array(test_outputs)

for i in range(test_outputs.shape[0]):
    for j in range(test_outputs.shape[1]):
        if test_outputs[i][j] >= 0.5: test_outputs[i][j] = 1
        else: test_outputs[i][j] = 0

for i in range(len(test_outputs)):
    roberta_df.at[i, 'Optimistic'] = test_outputs[i][0]
    roberta_df.at[i, 'Thankful'] = test_outputs[i][1]
    roberta_df.at[i, 'Empathetic'] = test_outputs[i][2]
    roberta_df.at[i, 'Pessimistic'] = test_outputs[i][3]
    roberta_df.at[i, 'Anxious'] = test_outputs[i][4]
    roberta_df.at[i, 'Sad'] = test_outputs[i][5]
    roberta_df.at[i, 'Annoyed'] = test_outputs[i][6]
    roberta_df.at[i, 'Denial'] = test_outputs[i][7]
    roberta_df.at[i, 'Official report'] = test_outputs[i][8]
    roberta_df.at[i, 'Joking'] = test_outputs[i][9]

roberta_df.to_csv("C:/Users/Admin/Desktop/roberta_df.csv", index=False)
