import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import pandas as pd
import gensim
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense

# Load Data
path = "dataset"

df_fake = pd.read_csv('dataset/Fake.csv')
df_true = pd.read_csv('dataset/True.csv')

# Preprocess Data
unknown_publishers_fake = []
for index, row in enumerate(df_fake.text.values):
    try:
        record = row.split(' - ', maxsplit=1)
        record[1]
        assert (len(record[0]) < 260)
    except:
        unknown_publishers_fake.append(index)

publisher_fake = []
tmp_text_fake = []
for index, row in enumerate(df_fake.text.values):
    if index in unknown_publishers_fake:
        tmp_text_fake.append(row)
        publisher_fake.append('unknown')
    else:
        record = row.split(' - ', maxsplit=1)
        publisher_fake.append(record[0])
        tmp_text_fake.append(record[1])

df_fake['publisher'] = publisher_fake
df_fake['text'] = tmp_text_fake
df_fake['text'] = df_fake['title'] + " " + df_fake['text']

unknown_publishers_true = []
for index, row in enumerate(df_true.text.values):
    try:
        record = row.split(' - ', maxsplit=1)
        record[1]
        assert (len(record[0]) < 260)
    except:
        unknown_publishers_true.append(index)

publisher_true = []
tmp_text_true = []
for index, row in enumerate(df_true.text.values):
    if index in unknown_publishers_true:
        tmp_text_true.append(row)
        publisher_true.append('unknown')
    else:
        record = row.split(' - ', maxsplit=1)
        publisher_true.append(record[0])
        tmp_text_true.append(record[1])

df_true['publisher'] = publisher_true
df_true['text'] = tmp_text_true
df_true['text'] = df_true['title'] + " " + df_true['text']

# Combine Fake and True DataFrames
df_fake['class'] = 1
df_true['class'] = 0
data = pd.concat([df_fake, df_true], ignore_index=True)

# Tokenize and Pad Sequences using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

X_bert = [" ".join(sent) for sent in data['text']]
tokenized_inputs = tokenizer(X_bert, padding=True, truncation=True, return_tensors='pt', max_length=1000, add_special_tokens=True)
X_bert_ids = tokenized_inputs['input_ids']
X_bert_masks = tokenized_inputs['attention_mask']

# Convert data to PyTorch tensors
y_tensor = torch.tensor(data['class'].values, dtype=torch.float32)

# Split data into train and test sets
X_train_ids, X_test_ids, X_train_masks, X_test_masks, y_train, y_test = train_test_split(
    X_bert_ids, X_bert_masks, y_tensor, test_size=0.2, random_state=42
)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=1)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training the model
num_epochs = 3
batch_size = 8  # Adjust as needed
train_dataset = TensorDataset(X_train_ids, X_train_masks, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    test_inputs = (X_test_ids.to(device), X_test_masks.to(device))
    predictions = model(*test_inputs).logits
    y_pred = (torch.sigmoid(predictions) >= 0.5).cpu().numpy().astype(int)

acc = accuracy_score(y_test.numpy(), y_pred)
print(f'Accuracy: {acc:.4f}')

class_report = classification_report(y_test.numpy(), y_pred)
print(class_report)
