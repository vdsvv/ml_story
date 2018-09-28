#! /usr/bin/env python
# -*- coding: utf-8 -*-

#https://towardsdatascience.com/a-beginners-guide-on-sentiment-analysis-with-rnn-9e100627c02e
#https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py
#https://m.habr.com/company/dca/blog/274027/

#https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/
#https://natural-language-understanding-demo.ng.bluemix.net/?cm_mc_uid=07180841621515339722626&cm_mc_sid_50200000=21935401537303374005&cm_mc_sid_52640000=99867961537303374019
#https://www.repustate.com/sentiment-analysis-api-demo/
#https://www.kaggle.com/c/sentiment-analysis-in-russian

#http://help.sentiment140.com/for-students/
#http://neuro.compute.dtu.dk/wiki/Sentiment_analysis#Corpora
#!!! http://study.mokoron.com

#!!! http://snowball.tartarus.org/algorithms/russian/stemmer.html
#https://github.com/ckoepp/TwitterSearch

#https://machinelearningmastery.com/start-here/
#https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/

#http://www.tutorialspoint.com/python/python_stemming_algorithms.htm

#https://github.com/plotti/keras_sentiment/blob/master/Imdb%20Sentiment.ipynb

import numpy as np
from keras.preprocessing import sequence
from dataset_rus_twitter import DatasetRusTwitter
from keras import Sequential
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import LSTM
#from keras.layers import GRU
#from keras.layers import CuDNNGRU
#from keras.layers import CuDNNLSTM
from keras.preprocessing.sequence import pad_sequences

class SentimentNN:
    def __init__(self):
        self.vocabulary_size = 75175
        self.max_words = 100
        self.dp = DatasetRusTwitter(self.vocabulary_size, index_from=3)
        pass

    def generateRusTweetData(self):
        (X_train, y_train), (X_test, y_test) = self.dp.loadData()
        self.max_words = 500
        self.X_train = sequence.pad_sequences(X_train, maxlen = self.max_words)
        self.X_test = sequence.pad_sequences(X_test, maxlen = self.max_words)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)

    def createModel(self):
        embedding_size = 32
        model = Sequential()
        model.add(Embedding(self.vocabulary_size, embedding_size, input_length = self.max_words))
        #Training mode: CuDNNLSTM
        #Inference mode: CuDNNLSTM or LSTM
        #model.add(CuDNNLSTM(100))
        #model.add(LSTM(100, recurrent_dropout=0.4, dropout=0.4))
        model.add(LSTM(100))
        model.add(Dense(1, activation='sigmoid'))
        #print(model.summary())
        model.compile(loss='binary_crossentropy', 
                    optimizer='adam', 
                    metrics=['accuracy'])
        self.model = model

    def saveWeights(self, file_path):
        self.model.save_weights(file_path)
        pass

    def loadWeights(self, file_path):
        self.model.load_weights(file_path)
        pass
        
    def predict(self, text):
        tokens = self.dp.textToDigitTokens(text)
        batch = np.expand_dims(tokens, axis = 0)
        batch = sequence.pad_sequences(batch, maxlen = self.max_words)
        prediction = self.model.predict(batch)
        print(text , prediction)
        return prediction
    
    def train(self, epochs=1, batch_size=64):
        X_valid, y_valid = self.X_train[:batch_size], self.y_train[:batch_size]
        X_train2, y_train2 = self.X_train[batch_size:], self.y_train[batch_size:]
        self.model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=epochs)
    
    def evaluate(self):
        scores = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        print('Test accuracy:', scores[1])