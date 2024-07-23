import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

# pip3 install transformers datasets scikit-learn torch

# Load datasets
train_df = pd.read_csv('/Users/zjy/Desktop/coding/abuse/train.csv')
test_df = pd.read_csv('/Users/zjy/Desktop/coding/abuse/test.csv')
test_labels_df = pd.read_csv('/Users/zjy/Desktop/coding/abuse/test_labels.csv')
sample_submission_df = pd.read_csv('/Users/zjy/Desktop/coding/abuse/sample_submission.csv')

# Initialize the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["comment_text"], padding="max_length", truncation=True)

# Preprocess the training data
train_df['label'] = train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].max(axis=1)
train_df = train_df[['comment_text', 'label']]
train_df, val_df = train_test_split(train_df, test_size=0.1)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Tokenize datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# Remove columns not needed for training
tokenized_train_dataset = tokenized_train_dataset.remove_columns(['comment_text', '__index_level_0__'])
tokenized_val_dataset = tokenized_val_dataset.remove_columns(['comment_text', '__index_level_0__'])

# Set format for PyTorch
tokenized_train_dataset.set_format("torch")
tokenized_val_dataset.set_format("torch")

# Load the model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./fine_tuned_roberta')

# Tokenize and prepare test dataset
test_df = test_df[['id', 'comment_text']]
test_dataset = Dataset.from_pandas(test_df)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = tokenized_test_dataset.remove_columns(['comment_text', '__index_level_0__'])
tokenized_test_dataset.set_format("torch")

# Make predictions on the test set
predictions = trainer.predict(tokenized_test_dataset)

# Prepare submission
sample_submission_df['toxic'] = predictions[0][:, 1]
sample_submission_df.to_csv('submission.csv', index=False)
