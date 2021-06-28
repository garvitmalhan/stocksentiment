import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
model = tf.keras.models.load_model('sentimentmodel.h5')

#Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)



#Set up variables for preprocessing and learning
vocab_size = 10000
embedding_dim = 50
max_length = 142
trunc_type='post'
padding_type='post'
oov_token = '<OOV>'

phrase = ['TSLA has decreased their sales']

testing_sequences = tokenizer.texts_to_sequences(phrase)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length,padding=padding_type,truncating=trunc_type)

pred = model.predict(testing_padded)
classes = np.argmax(pred, axis=-1)
dict_sentiment = {0:'Negative', 1:'Neutral', 2: 'Positive'}
print(f'{phrase} : {dict_sentiment[int(classes)]}')