import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, RepeatVector, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Sample English to Malayalam translation dataset
english_texts = ["Hello", "How are you?", "What is your name?", "I love programming"]
malayalam_texts = ["ഹലോ", "സുഖമാണോ?", "നിനക്ക് പേരാണോ?", "ഞാൻ പ്രോഗ്രാമിംഗ് സ്നേഹിക്കുന്നു"]

tokenizer_en = Tokenizer(filters='')
tokenizer_en.fit_on_texts(english_texts)
tokenizer_ml = Tokenizer(filters='')
tokenizer_ml.fit_on_texts(malayalam_texts)

vocab_size_en = len(tokenizer_en.word_index) + 1
vocab_size_ml = len(tokenizer_ml.word_index) + 1

input_sequences = tokenizer_en.texts_to_sequences(english_texts)
target_sequences = tokenizer_ml.texts_to_sequences(malayalam_texts)

input_sequences = pad_sequences(input_sequences)
target_sequences = pad_sequences(target_sequences)

model = Sequential([
    Embedding(vocab_size_en, 256, input_length=input_sequences.shape[1]),
    Bidirectional(LSTM(512, return_sequences=True)),
    Bidirectional(LSTM(512)),
    RepeatVector(target_sequences.shape[1]),
    Bidirectional(LSTM(512, return_sequences=True)),
    Bidirectional(LSTM(512, return_sequences=True)),
    Dense(vocab_size_ml, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(input_sequences, target_sequences, epochs=50, verbose=2)
def translate_text(model, input_text):
    input_seq = tokenizer_en.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=input_sequences.shape[1])
    predicted_sequence = model.predict(input_seq)
    malayalam_sequence = tf.argmax(predicted_sequence, axis=-1).numpy()
    malayalam_text = tokenizer_ml.sequences_to_texts(malayalam_sequence)
    return malayalam_text[0]

# Test the translator
input_text = "How are you?"
translated_text = translate_text(model, input_text)
print(f"Input: {input_text}")
print(f"Translated: {translated_text}")


