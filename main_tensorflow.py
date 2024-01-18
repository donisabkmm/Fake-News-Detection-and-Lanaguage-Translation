
import matplotlib.pyplot as plt #MATLAB-like, way of plotting
from wordcloud import WordCloud #Generating Word Cloud in Python
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Embedding,Bidirectional,LSTM,Conv1D,MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
import pandas as pd
import gensim

path = "dataset"

df=pd.read_csv ('dataset/Fake.csv')
df.head(3)

text= ' '.join(df['text'].tolist())
wordcloud = WordCloud(width=1920,height=1080).generate(text)
fig =plt.figure(figsize=(10, 10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

dr=pd.read_csv ('dataset/True.csv')
dr.head(1)

text= ' '.join(dr['text'].tolist())
wordcloud = WordCloud(width=1920,height=1080).generate(text)
fig =plt.figure(figsize=(10, 10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


unknown_publishers = []
for index,row in enumerate(dr.text.values):
  try:
    record=row.split(' - ',maxsplit=1)
    record[1]
    assert (len(record[0])<260)
  except:
    unknown_publishers.append(index)
publisher = []
tmp_text =[]
for index,row in enumerate(dr.text.values):
  if index in unknown_publishers:
    tmp_text.append(row)
    publisher.append('unknown')

  else:
      record = row.split(' - ',maxsplit=1)
      publisher.append(record[0])
      tmp_text.append(record[1])

dr['publisher']=publisher
dr['text']=tmp_text
dr['text']=dr['title']+" "+dr['text']
df['text']=df['title']+" "+df['text']
dr['text']=dr['text'].apply(lambda x:str(x).lower())
df['text']=df['text'].apply(lambda x:str(x).lower())
dr['class']=1
df['class']=0

data = pd.concat([dr, df], ignore_index=True)

y=data['class'].values
X=[d.split() for d in data['text'].tolist()]
DIM=100
w2v_model = gensim.models.Word2Vec(sentences=X, vector_size=DIM, window=10, min_count=1)
from keras.preprocessing.text import Tokenizer
tokenizer =Tokenizer()
tokenizer.fit_on_texts(X)
X =tokenizer.texts_to_sequences(X)
import numpy as np
nos = np.array([len(x) for x in X])
len(nos[nos>1000])
maxlen =1000
X =pad_sequences(X,maxlen=maxlen)
vocab_size=len(tokenizer.word_index)+1
vocab=tokenizer.word_index
def get_weight_matrix(model):
  weight_matrix=np.zeros((vocab_size,DIM))
  for word, i in vocab.items():
    weight_matrix[i]=model.wv[word]
  return weight_matrix
embedding_vectors= get_weight_matrix(w2v_model)

model=Sequential()
model.add(Embedding(vocab_size,output_dim=DIM, weights= [embedding_vectors],input_length=maxlen, trainable=False))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
history=model.fit(X_train,y_train,epochs=10,batch_size=64,validation_data=(X_test,y_test))
y_pred = (model.predict(X_test) >=0.5).astype(int)
accuracy_score(y_test,y_pred)

print(classification_report(y_test,y_pred))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['Train','Test'])
plt.show()