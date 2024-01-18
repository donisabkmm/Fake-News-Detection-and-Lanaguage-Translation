import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
# from keras.utils import np_utils
import numpy as np

from main_tensorflow import history

# Load Data
path = "dataset"

df_fake = pd.read_csv('dataset/Fake.csv')
df_true = pd.read_csv('dataset/True.csv')

# Word Cloud for Fake News
text_fake = ' '.join(df_fake['text'].tolist())
wordcloud_fake = WordCloud(width=1920, height=1080).generate(text_fake)
fig_fake = plt.figure(figsize=(10, 10))
plt.imshow(wordcloud_fake)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

# Word Cloud for True News
text_true = ' '.join(df_true['text'].tolist())
wordcloud_true = WordCloud(width=1920, height=1080).generate(text_true)
fig_true = plt.figure(figsize=(10, 10))
plt.imshow(wordcloud_true)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

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

# Tokenize and Pad Sequences
X = [d.split() for d in data['text'].tolist()]
DIM = 100
w2v_model = gensim.models.Word2Vec(sentences=X, vector_size=DIM, window=10, min_count=1)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

nos = np.array([len(x) for x in X])
maxlen = 1000
X = pad_sequences(X, maxlen=maxlen)
vocab_size = len(tokenizer.word_index) + 1
vocab = tokenizer.word_index


def get_weight_matrix(model):
    weight_matrix = np.zeros((vocab_size, DIM))
    for word, i in vocab.items():
        weight_matrix[i] = model.wv[word]
    return weight_matrix


embedding_vectors = get_weight_matrix(w2v_model)

# Define PyTorch model
class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.FloatTensor(embedding_vectors))
        self.embedding.weight.requires_grad = False  # Freeze the embedding layer
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.bilstm(embedded)
        output = self.fc(output[:, -1, :])
        return self.sigmoid(output)


# Convert data to PyTorch tensors
y_tensor = torch.tensor(data['class'].values, dtype=torch.float32)
X_padded = [torch.LongTensor(seq) for seq in X]
X_padded = pad_sequence(X_padded, batch_first=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_tensor, test_size=0.2)

# Instantiate the model
embedding_dim = DIM
hidden_dim = 100
model = BiLSTM(embedding_dim, hidden_dim)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    predictions = (model(X_test) >= 0.5).float()
    acc = accuracy_score(y_test.numpy(), predictions.numpy())
    print(f'Accuracy: {acc:.4f}')

    class_report = classification_report(y_test.numpy(), predictions.numpy())
    print(class_report)

# Plotting accuracy
plt.plot(history.history['accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['Train'])
plt.show()
