import torch
import torch.nn as nn
from transformers import RobertaModel

class RoBERTaCustom(nn.Module):
    def __init__(self, num_labels=10):
        super(RoBERTaCustom, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits