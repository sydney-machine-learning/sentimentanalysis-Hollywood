import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from pathlib import Path

# --- Paths relative to the repo (never hardcode C:/Users/Admin/...) ---
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = REPO_ROOT / "coding" / "data_csv.csv"                        # input
MODEL_PATH = REPO_ROOT / "coding" / "model" / "roberta-finetuned.pth"  # trained weights
OUTPUT_CSV = REPO_ROOT / "coding" / "roberta_df.csv"                    # output

MAX_LEN = 200          # tokens per movie (NOTE: this truncates long movies)
INFER_BATCH_SIZE = 32  # was 1 -- batching makes inference far faster
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
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


# Load the fine-tuned model. map_location=device lets a model saved on GPU load
# on CPU (and vice versa); .to(device) then puts it where the data will be.
roberta = torch.load(MODEL_PATH, map_location=device)
roberta.to(device)

mdf = pd.read_csv(DATA_CSV)

roberta_df = pd.DataFrame()
roberta_df['movie'] = mdf['bodyContent']

# One [0]*10 placeholder target per movie (unused at inference). Using a list
# comprehension over len(mdf) avoids both the magic number 1027 AND a subtle bug:
# [[0]*10] * n makes n references to the SAME list, not n independent lists.
roberta_df['list'] = [[0] * 10 for _ in range(len(mdf))]


test_dataset = CustomDataset(roberta_df, tokenizer, MAX_LEN)

roberta_test_params = {'batch_size': INFER_BATCH_SIZE,  # was 1 -- the key speedup
                       'shuffle': False,
                       'num_workers': 0  # keep 0 on Windows to avoid multiprocessing issues
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

test_outputs = np.array(test())

# Threshold at 0.5 -> 0/1, vectorized (replaces a slow nested Python loop).
test_outputs = (test_outputs >= 0.5).astype(int)

# Assign all 10 emotion columns at once (replaces the per-cell .at[] loop).
EMOTIONS = ['Optimistic', 'Thankful', 'Empathetic', 'Pessimistic', 'Anxious',
            'Sad', 'Annoyed', 'Denial', 'Official report', 'Joking']
roberta_df[EMOTIONS] = test_outputs

roberta_df.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote {len(roberta_df)} rows to {OUTPUT_CSV}")
