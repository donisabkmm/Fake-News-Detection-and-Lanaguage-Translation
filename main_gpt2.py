import matplotlib.pyplot as plt
from wordcloud import WordCloud
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from transformers import GPT2Tokenizer, GPT2Model
import torch

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2Model.from_pretrained("gpt2")

path = "dataset"

df = pd.read_csv('dataset/Fake.csv')
df.head(3)

text = ' '.join(df['text'].tolist())
wordcloud = WordCloud(width=1920, height=1080).generate(text)
fig = plt.figure(figsize=(10, 10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

dr = pd.read_csv('dataset/True.csv')
dr.head(1)

text = ' '.join(dr['text'].tolist())
wordcloud = WordCloud(width=1920, height=1080).generate(text)
fig = plt.figure(figsize=(10, 10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

unknown_publishers = []
for index, row in enumerate(dr.text.values):
    try:
        record = row.split(' - ', maxsplit=1)
        record[1]
        assert (len(record[0]) < 260)
    except:
        unknown_publishers.append(index)

publisher = []
tmp_text = []
for index, row in enumerate(dr.text.values):
    if index in unknown_publishers:
        tmp_text.append(row)
        publisher.append('unknown')
    else:
        record = row.split(' - ', maxsplit=1)
        publisher.append(record[0])
        tmp_text.append(record[1])

dr['publisher'] = publisher
dr['text'] = tmp_text
dr['text'] = dr['title'] + " " + dr['text']
df['text'] = df['title'] + " " + df['text']
dr['text'] = dr['text'].apply(lambda x: str(x).lower())
df['text'] = df['text'].apply(lambda x: str(x).lower())
dr['class'] = 1
df['class'] = 0

data = pd.concat([dr, df], ignore_index=True)

y = data['class'].values
X = [d.split() for d in data['text'].tolist()]
DIM = 768  # GPT-2 hidden size

# Tokenize and Pad Sequences using GPT-2 tokenizer
X_gpt2 = [" ".join(sent) for sent in X]
tokenized_inputs = tokenizer(X_gpt2, padding=True, truncation=True, return_tensors='pt', max_length=1000)
input_ids = tokenized_inputs['input_ids']

# Get GPT-2 embeddings
with torch.no_grad():
    gpt2_embeddings = gpt2_model(input_ids).last_hidden_state.mean(dim=1)

# Convert data to numpy array
X_gpt2 = gpt2_embeddings.numpy()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_gpt2, y, test_size=0.2)

# Build the model
model = Sequential()
model.add(Dense(256, input_dim=DIM, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
y_pred = (model.predict(X_test) >= 0.5).astype(int)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'])
plt.show()
