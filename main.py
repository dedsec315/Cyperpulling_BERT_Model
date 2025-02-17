!pip install transformers
'''import pandas as pd
import random
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

random_seed = 42
random.seed(random_seed)


# Function to load and preprocess the dataset

def loadAndPreProcess(path):

  df = pd.read_json(path, lines = True) # Read the data

  df["label"] = df.annotation.apply(lambda x: x.get('label'))  # Extract label list
  df["label"] = df.label.apply(lambda x: x[0])  # Get first label from the list

  X = df.content.values  # Extract text content
  y = df.label.values  # Extract labels

  return X, y


# Load the data

X, y = loadAndPreProcess('/content/Dataset.json')

# Load the pretrained Transformer (BERT) and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
Bert_model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize our database using Bert tokenizer
encoding = tokenizer.batch_encode_plus(
    pd.Series(X).dropna().tolist(),                	
    padding=True,          	
    truncation=True,       	
    return_tensors='pt',  	
    add_special_tokens=True
)
 
token_ids = encoding['input_ids']  
print(f"Token ID: {token_ids}")
attentionMask = encoding['attention_mask']  
print(f"Attention mask: {attentionMask}")

#print(tokenized_data) '''

# Install required libraries (uncomment if not installed)
# !pip install transformers torch pandas scikit-learn

import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)

def load_and_preprocess(path):
    """Loads dataset and extracts text and labels."""
    df = pd.read_json(path, lines=True)  # Read JSON file
    df["label"] = df.annotation.apply(lambda x: x.get('label')[0])  # Extract first label
    X = df.content.values  # Extract text content
    y = df.label.values  # Extract labels
    return X, y

# Load dataset
X, y = load_and_preprocess('/content/Dataset.json')

# Convert labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Custom Dataset class
class CyberTrollDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create DataLoaders
train_dataset = CyberTrollDataset(X_train, y_train, tokenizer)
test_dataset = CyberTrollDataset(X_test, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define BERT Classifier Model
class BertClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output  # Get [CLS] token representation
        dropped_output = self.dropout(pooled_output)
        return self.fc(dropped_output)

# Initialize model
num_classes = len(set(y))  # Get number of unique labels
model = BertClassifier(num_classes)
device = torch.device("cuda")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=3)

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

# Evaluate the model
evaluate_model(model, test_loader)






# Test our model
def predict_label(model, text, tokenizer, label_encoder, device):
    model.eval()
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        _, predicted_label = torch.max(output, dim=1)
    
    return predicted_label.item()


Test_case = ""

predicted_label = predict_label(model, Test_case, tokenizer, label_encoder, device)
print(f"Predicted Label: {predicted_label}")
  
