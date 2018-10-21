#http://adventuresinmachinelearning.com/keras-lstm-tutorial/
#http://corochann.com/penn-tree-bank-ptb-dataset-introduction-1456.html
#http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

#from __future__ import self.__print_function
import os
import numpy as np
import random
import time

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import CuDNNGRU
from keras.layers import Bidirectional
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import Callback

from dataset import Dataset

class LiveSamplerCallback(Callback):
    def __init__(self, model, sb):
        super(LiveSamplerCallback, self).__init__()
        self.model = model
        self.sb = sb
    
    def on_epoch_begin(self, batch, logs):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs):
        dtime = time.time() - self.epoch_time_start
        self.sb.send_message('✌️ <b>Epoch:</b> <code>{0:03}/{1:03}</code> <b>time:</b> <code>{2:0.0f}s</code> <b>acc:</b> <code>{3:0.4f}</code> <b>val-acc:</b> <code>{4:0.4f}</code>'.
                             format(epoch, self.params['epochs'], dtime, logs['acc'], logs['val_acc']))
            
class SequenceGeneratorNN:
    def __init__(self, params, service_bot = None):
        params = params.copy()
        self.sb = service_bot
        self.__print('👉 sgnn::__init__')
        self.num_steps = params['model_steps_num']
        self.path_models = params['path_models']
        self.default_model_name = params['default_model_name']
        self.default_model_ext = params['default_model_ext']
        self.default_model_path = self.path_models + self.default_model_name + self.default_model_ext
        self.skip_step = params['skip_step']
        self.batch_size = params['batch_size']
        self.save_model_auto = params['save_model_auto']
        self.epochs = params['epochs']
        self.rnn_units = params['rnn_units']
        self.rnn_bidirectional = params['rnn_bidirectional']
        self.rnn_layers = params['rnn_layers']
        self.rnn_gpu_unit = params['rnn_gpu_unit']
        self.embeding_dim = params['embeding_dim']
        self.stateful = params['stateful']
        self.temperature = params['pred_temperature']
        self.pred_words_num = params['pred_words_num']
        self.pred_begin_words_num = params['pred_begin_words_num']
        self.dropout = params['dropout']
        self.save_model_auto_period = params['save_model_auto_period']
        self.use_tensor_board = params['use_tensor_board']
        self.validation_split = params['validation_split']
        self.dp = Dataset(params)
        self.model = None
        self.__print_params(params)
        self.__print('👈 sgnn::__init__')

    def __print(self, text):
        if self.sb != None:
            self.sb.print(text)
        print(text)

    def __print_params(self, ps):
        del ps['service_bot_token']
        del ps['work_bot_token']
        del ps['bot_proxy_url']

        if self.sb != None:
            self.sb.print_params(ps)

        text = '🍀Parameters:\n'
        for par in ps.items():
            text += '{}: {}\n'.format(par[0], par[1])
        print(text)

    def generateData(self):
        self.__print('👉 sgnn::generateData')
        self.loadData()
        self.loadVocab()
        self.__print('👈 sgnn::generateData')

    def loadData(self):
        self.__print('👉 sgnn::loadData')
        self.train_data_X, self.train_data_Y = self.dp.loadData()
        self.__print('👈 sgnn::loadData')

    def loadVocab(self):
        self.__print('👉 sgnn::loadVocab')
        self.word_2_id, self.id_2_word = self.dp.loadVocab()
        self.vocab_size = len(self.word_2_id)
        #self.default_model_path = self.path_models + self.default_model_name + '-VOCAB' + str(self.vocab_size) + self.default_model_ext
        self.__print('👈 sgnn::loadVocab')

    #I:  [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    #S1: [ 10.  20.  30.] 🖤 [20, 30, 40]
    #S2: [ 40.  50.  60.] 🖤 [50, 60, 70]
    #S3: [ 70.  80.  90.] 🖤 [80, 90, 100]
    #S4: [ 10.  20.  30.] 🖤 [20, 30, 40]
    def train(self):
        self.__print('👉 sgnn::trainModel')
        callbacks = []
        if self.save_model_auto:
            callbacks.append(ModelCheckpoint(filepath=self.default_model_path, verbose=1, 
                             period=self.save_model_auto_period))
        if self.use_tensor_board:
            callbacks.append(TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True))
        
        if self.sb != None:
            callbacks.append(LiveSamplerCallback(self.model, self.sb))

        self.model.fit(x=self.train_data_X, y=self.train_data_Y, validation_split=self.validation_split,
                       epochs=self.epochs, batch_size=self.batch_size, 
                       shuffle=True, callbacks=callbacks)
        self.__print('👈 sgnn::trainModel')

    # early_stopping = EarlyStopping(monitor='val_acc', patience=5)
    #def train(self, epochs, batch_size, save_model_auto = False, use_tensor_board = False):
    #    callbacks = []
    #    if save_model_auto:
    #        callbacks.append(ModelCheckpoint(filepath=self.models_path + 'model-{epoch:02d}.hdf5', verbose=1))
    #    if use_tensor_board:
    #        callbacks.append(TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True))
    #    
    #    self.train_data_generator = BatchGenerator(self.train_data, self.num_steps, batch_size, 
    #                                            self.vocab_size, skip_step=self.skip_step)
    #    steps_per_epoch = len(self.train_data)//(batch_size*self.num_steps)
    #    self.model.fit_generator(generator=self.train_data_generator.generate(), steps_per_epoch=steps_per_epoch, epochs=epochs,
    #                             callbacks=callbacks)
    
    def saveWeights(self, file_path):
        self.__print('👉 sgnn::saveWeights')
        if file_path == None:
            file_path = self.default_model_path
        self.__print('🤙 sgnn::saveWeights: ' + file_path)
        self.model.save_weights(file_path)
        self.__print('👈 sgnn::saveWeights')

    def loadWeights(self, file_path = None):
        self.__print('👉 sgnn::loadWeights')
        if file_path == None:
            file_path = self.default_model_path
        self.__print('🤙 sgnn::loadWeights: ' + file_path)
        self.model.load_weights(file_path)
        self.__print('👈 sgnn::loadWeights')

    def createModel(self):
        self.__print('👉 sgnn::createModel')
        model = Sequential()
        
        if self.stateful:
            model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embeding_dim, batch_input_shape=(self.batch_size, None)))
        else:
            model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embeding_dim, input_length=self.num_steps))
        
        for _ in range(self.rnn_layers):
            rnn_unit = None
            if self.rnn_gpu_unit:
                rnn_unit = CuDNNLSTM(self.rnn_units, stateful=self.stateful, return_sequences=True) 
            else:
                rnn_unit = LSTM(self.rnn_units, stateful=self.stateful, return_sequences=True)
            if self.rnn_bidirectional:
                model.add(Bidirectional(rnn_unit))
            else:
                model.add(rnn_unit)
        
        model.add(Dropout(self.dropout))
        model.add(Dense(self.vocab_size, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy',
                        optimizer='rmsprop',
                        metrics=['accuracy'])
        #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        self.model = model
        self.__print('👈 sgnn::createModel')

    def createModel_v4(self):
        self.__print('👉 sgnn::createModel')
        model = Sequential()
        if self.stateful:
            model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embeding_dim, batch_input_shape=(self.batch_size, None)))
        else:
            model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embeding_dim, input_length=self.num_steps))
        model.add(CuDNNLSTM(self.rnn_units, stateful=self.stateful, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(CuDNNLSTM(self.rnn_units, stateful=self.stateful, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(TimeDistributed(Dense(self.vocab_size, activation='softmax')))
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        self.model = model
        self.__print('👈 sgnn::createModel')

    def createModel_v3(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embeding_dim, batch_input_shape=(self.batch_size, None)))
        model.add(CuDNNLSTM(self.rnn_units, stateful=True, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(CuDNNLSTM(self.rnn_units, stateful=True, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(self.vocab_size, activation='softmax')))
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        self.model = model

    def createModel_v2(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=64, batch_input_shape=(self.batch_size, None)))
        model.add(CuDNNLSTM(128, stateful=True, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(CuDNNLSTM(128, stateful=True, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(Dense(self.vocab_size))
        model.add(Activation('softmax'))
        #opt = SGD(lr=0.005, momentum=0.0, decay=0.0, nesterov=False)
        #model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        self.model = model

    def createModel_v1(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=64, input_length=self.num_steps))
        model.add(CuDNNLSTM(128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(CuDNNLSTM(128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(Dense(self.vocab_size))
        model.add(Activation('softmax'))
        #opt = SGD(lr=0.005, momentum=0.0, decay=0.0, nesterov=False)
        #model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        self.model = model

    def setTemperature(self, temp):
        if temp >= 0.0 and temp <= 1.0:
            self.temperature = temp
    
    def setPredictWordsNumber(self, number):
        if number >= 1 and number <= 500:
            self.pred_words_num = number
    
    def setPredictBeginWordsNumber(self, number):
        if number >= 1 and number <= 200:
            self.pred_begin_words_num = number

    #I:  ['Солнце', 'просунуло', 'огненные']
    #O:  ['просунуло', 'огненные', 'щупальца🔹']
    #I:  ['Солнце', 'просунуло', 'огненные', 'щупальца']
    #O:  ['просунуло', 'огненные', 'щупальца', 'сквозь🔹']
    #I:  ['Солнце', 'просунуло', 'огненные', 'щупальца', 'сквозь']
    #O:  ['просунуло', 'огненные', 'щупальца', 'сквозь', 'щели🔹']
    #F:  ['Солнце', 'просунуло', 'огненные', 'щупальца', 'сквозь', 'щели']
    def predict(self, text, pred_words_num=None, pred_begin_words_num=None):
        print('👉 sgnn::predict')
        if pred_words_num == None:
            pred_words_num = self.pred_words_num
        if pred_begin_words_num == None:
            pred_begin_words_num = self.pred_begin_words_num
        full_text_sequence = ''
        digit_tokens_src = self.dp.textToDigitTokens(text)
        if len(digit_tokens_src) == 0:
            print('👈 sgnn::predict')
            return '💔 Введённых слов нет словаре 💔'
        text_tokens_src = [self.id_2_word[token] for token in digit_tokens_src]
        full_text_sequence = '<b>' + ' '.join(text_tokens_src) + '</b>' 
        for _ in range(pred_words_num):
            data_src = np.reshape(digit_tokens_src, (1, -1))
            data_pred = self.model.predict(data_src)
            digit_tokens_pred = data_pred[0]
            last_pred_digit_token_t, prob = self.__applyTemperature(digit_tokens_pred[-1], self.temperature)
            last_pred_text_token_t = self.id_2_word[last_pred_digit_token_t]
            digit_tokens_pred = np.argmax(digit_tokens_pred, axis=1)
            last_pred_digit_token = digit_tokens_pred[-1]
            last_pred_text_token = self.id_2_word[last_pred_digit_token]
            digit_tokens_pred[-1] = last_pred_digit_token_t
            text_tokens_pred = [self.id_2_word[token] for token in digit_tokens_pred]
            #self.__print('I: ', text_tokens_src)
            #self.__print('O: ', text_tokens_pred)
            #☀⭐🍭🍋👉
            if last_pred_digit_token_t != last_pred_digit_token:
                #full_text_sequence += ' ' + '️<code>' + last_pred_text_token_t + '</code>' # + '[' + str(prob) + ']' + '👈🏽'
                #full_text_sequence += ' ' + last_pred_text_token_t + '<code>[' + last_pred_text_token + ']</code>' # + '[' + str(prob) + ']' + '👈🏽'
                full_text_sequence += ' ' + '<b>' + last_pred_text_token_t + '</b>'
            else:    
                full_text_sequence += ' ' + last_pred_text_token_t# + '[' + str(prob) + ']'
            
            digit_tokens_src.append(digit_tokens_pred[-1])
            dtsl = len(digit_tokens_src)
            if pred_begin_words_num != None and dtsl > pred_begin_words_num:
                digit_tokens_src = digit_tokens_src[1:]
            text_tokens_src = [self.id_2_word[token] for token in digit_tokens_src]
        #self.__print('F: ', full_text_sequence)
        print('👈 sgnn::predict')
        return full_text_sequence

    def __applyTemperature(self, word_probs, temperature):
        if temperature is None or temperature == 1.0:
            index = np.argmax(word_probs)
            prob = word_probs[index]
            return (index, prob)
        else:
            sorted_indexes = (-word_probs).argsort()
            arr = []
            index = sorted_indexes[0]
            prob = word_probs[index]
            arr.append((index, prob))
            index_alt_1 = sorted_indexes[1]
            prob_alt_1 = word_probs[index_alt_1]
            #index_alt_2 = sorted_indexes[2]
            #prob_alt_2 = word_probs[index_alt_2]
            #if prob_alt_2 >= 0.1:
            #    return index_alt_2, prob_alt_2
            if prob_alt_1 >= temperature:
                arr.append((index_alt_1, prob_alt_1))
            return random.choice(arr)

    def __applyTemperature_v1(self, word_probs, temperature):
        if temperature is not None and temperature != 0.0:
            word_probs = np.asarray(word_probs).astype('float64')
            preds = np.log(word_probs + K.epsilon()) / temperature
            #preds = np.log(word_probs) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            #self.__print(sum(preds))
            #self.__print('👉 sgnn::multinomial')
            word_probs = np.random.multinomial(1, preds, 1)
            #self.__print('👈 sgnn::multinomial')
        return np.argmax(word_probs)