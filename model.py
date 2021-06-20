import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

df = pd.read_csv('financialNews.csv')
df.columns = ['Sentiment', 'News']
mapper = {'negative': 0, 'neutral': 1, 'positive': 2}
df.Sentiment = df.Sentiment.map(mapper)
train, test = train_test_split(df, test_size=0.2)
train_news = np.array(train['News'].tolist().copy())
labels = keras.utils.to_categorical(train['Sentiment'].astype('int64'))
test_news = np.array(test['News'].tolist().copy())
labels_test = keras.utils.to_categorical(test['Sentiment'].astype('int64'))

# Set up variables for preprocessing and learning
vocab_size = 10000
embedding_dim = 50
max_length = 142
trunc_type = 'post'
padding_type = 'post'
oov_token = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_news)

# tokenizer.word_index

sequences = tokenizer.texts_to_sequences(train_news)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(test_news)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
num_epochs = 30
history = model.fit(padded, labels, epochs=num_epochs, validation_data=(testing_padded, labels_test))
model.save('sentimentmodel.h5')
phrase = ['TSLA looks bullish']

testing_sequences = tokenizer.texts_to_sequences(phrase)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length,padding=padding_type,truncating=trunc_type)

pred = model.predict(testing_padded)
classes = np.argmax(pred, axis=-1)
dict_sentiment = {0:'Negative', 1:'Neutral', 2: 'Positive'}
print(f'{phrase} : {dict_sentiment[int(classes)]}')
