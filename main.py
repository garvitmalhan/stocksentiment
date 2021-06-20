from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import requests


url = "https://www.google.com/finance/quote/AXISBANK:NSE"
symbols=['ICICIBANK:NSE', 'HDFCBANK:NSE', 'AXISBANK:NSE', 'KOTAKBANK:NSE', 'SBIN:NSE', 'INDUSINDBK:NSE', 'AUBANK:NSE', 'BANDHANBNK:NSE', 'FEDERALBNK:NSE', 'IDFCFIRSTB:NSE', 'PNB:NSE', 'RBLBANK:NSE']
req = Request(url=url, headers={'user-agent': 'my-app'})
response = urlopen(req)
html = BeautifulSoup(response, features='html.parser')
news = html.find("div", attrs={"class": "AoCdqe"}).text
news_data=html.findAll("div", attrs={"class": "yY3Lee"})

#print(news_data)
df = pd.DataFrame(columns=['News', 'Time'])

for index, row in enumerate(news_data):
    stocknews = row.find("div", attrs={"class": "AoCdqe"}).text
    times=row.find("div", attrs={"class": "Adak"}).text
    df = df.append({'News': stocknews, 'Time': times}, ignore_index=True)


model = tf.keras.models.load_model('sentimentmodel.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

oov_tok = '<OOV>'
trunc_type = 'post'
padding_type='post'
vocab_size =10000
max_length = 142


def preprocessText(text):
    sequences = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    return padded


prep = preprocessText(df.News)
prep = model.predict(prep)
df['Sentiment'] = np.argmax(prep, axis=-1)

print(df)







