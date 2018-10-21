from data_parser_x import DataParser
import random
import collections
import os
import numpy as np
from keras.utils import to_categorical
from collections import Counter

class Dataset:
    def __init__(self, params, vocabulary_size=None, index_from=None):
        self.__ps = params
        self.vocabulary_size = vocabulary_size
        self.index_from = index_from
        self.__dp = DataParser()
        self.vocab_spec = None        
        #self.vocab_spec = [('<PAD>',0), ('<EOS>',0), ('<UNK>',0), ('<GO>',0)]        
        
    def loadData(self):
        self.sorted_vocab = self.__dp.loadVocab(self.__ps['path_vocab'])
        self.train_data_X = self.__dp.loadDigitTokenSequences(self.__ps['path_input_digit_tokens'])
        self.train_data_Y = self.__dp.loadDigitTokenSequences(self.__ps['path_target_digit_tokens'])
        self.train_data_X = np.array(self.train_data_X)
        #self.train_data_X =  self.train_data_X[:, :, np.newaxis]
        self.train_data_Y = np.array(self.train_data_Y)
        self.train_data_Y =  self.train_data_Y[:, :, np.newaxis]
        #self.train_data_Y =  to_categorical(self.train_data_Y, num_classes=len(self.sorted_vocab))
        return self.train_data_X, self.train_data_Y

    def loadVocab(self):
        self.sorted_vocab = self.__dp.loadVocab(self.__ps['path_vocab'])
        self.word_2_id = self.__dp.getWord2Index(self.sorted_vocab)
        self.id_2_word = self.__dp.getIndex2Word(self.sorted_vocab)
        return self.word_2_id, self.id_2_word

    def textToDigitTokens(self, text):
        textTokens = self.__dp.getTextTokenSequenceWithPunctuation(text)
        digitTokens = self.__dp.getDigitTokenSequence(self.word_2_id, textTokens)
        #self.__dp.modifyDigitTokenSequence(digitTokens, self.vocabulary_size, self.index_from)
        return digitTokens

    def prepareData(self, params_list):
        self.__pss = params_list
        self.generateTextSequences()
        self.generateTextTokenSequence()
        self.generateVocabulary()
        self.generateTextTokenXY()
        self.generateDigitTokenXY()
    
    def generateTextSequences(self):
        if self.__pss == None:
            self.__dp.cropTextLines(self.__ps['path_raw_text'], self.__ps['path_crop_text'], self.__ps['crop_lines'])
            tss = self.__dp.getTextSentenceSequencesByFile(self.__ps['path_crop_text'])
            tss = self.__dp.removeExtraWhitespaces(tss)
            self.__dp.saveTextSequences(self.__ps['path_refined_text'], tss)
        else:
            for ps in self.__pss:
                self.__dp.cropTextLines(ps['path_raw_text'], ps['path_crop_text'], self.__ps['crop_lines'])
                tss = self.__dp.getTextSentenceSequencesByFile(ps['path_crop_text'])
                tss = self.__dp.removeExtraWhitespaces(tss)
                self.__dp.saveTextSequences(ps['path_refined_text'], tss)

    def generateTextTokenSequence(self):
        if self.__pss == None:
            tss = self.__dp.loadTextSequences(self.__ps['path_refined_text'])
            ttss = self.__dp.getTextTokenSequencesWithPunctuation(tss)
            tts = self.__dp.concatSentences(ttss)
            tts = self.__dp.removeComment(tts)
            self.__dp.saveTextTokenSequence(self.__ps['path_text_tokens'], tts)
        else:
            for ps in self.__pss:
                tss = self.__dp.loadTextSequences(ps['path_refined_text'])
                ttss = self.__dp.getTextTokenSequencesWithPunctuation(tss)
                tts = self.__dp.concatSentences(ttss)
                tts = self.__dp.removeComment(tts)
                self.__dp.saveTextTokenSequence(ps['path_text_tokens'], tts)
        
    def generateTextTokenXY(self):
        if self.__pss == None:
            tts = self.__dp.loadTextTokenSequence(self.__ps['path_text_tokens'])
            input_tts = tts[:-1]
            target_tts = tts[1:]
            input_tts = self.__dp.selectTokenSequences(input_tts, self.__ps['model_steps_num'], self.__ps['skip_step'])
            target_tts = self.__dp.selectTokenSequences(target_tts, self.__ps['model_steps_num'], self.__ps['skip_step'])
            self.__dp.saveTextTokenSequences(self.__ps['path_input_text_tokens'], input_tts)
            self.__dp.saveTextTokenSequences(self.__ps['path_target_text_tokens'], target_tts)
        else:
            input_tts_x = []
            target_tts_x = []
            for ps in self.__pss:
                tts = self.__dp.loadTextTokenSequence(ps['path_text_tokens'])
                input_tts = tts[:-1]
                target_tts = tts[1:]
                input_tts_x.extend(self.__dp.selectTokenSequences(input_tts, self.__ps['model_steps_num'], self.__ps['skip_step']))
                target_tts_x.extend(self.__dp.selectTokenSequences(target_tts, self.__ps['model_steps_num'], self.__ps['skip_step']))
            p = np.random.permutation(len(input_tts_x))
            input_tts_x = np.array(input_tts_x)
            target_tts_x = np.array(target_tts_x)
            input_tts_x = input_tts_x[p]
            target_tts_x = target_tts_x[p]
            self.__dp.saveTextTokenSequences(self.__ps['path_input_text_tokens'], input_tts_x)
            self.__dp.saveTextTokenSequences(self.__ps['path_target_text_tokens'], target_tts_x)
        pass

    def generateVocabulary(self):
        if self.__pss == None:
            tts = self.__dp.loadTextTokenSequence(self.__ps['path_text_tokens'])
            _, sorted_vocab = self.__dp.buildVocab(tts, self.vocab_spec)
            self.__dp.saveVocab(self.__ps['path_vocab'], sorted_vocab)
        else:
            tts = []
            for ps in self.__pss:
                tts.extend(self.__dp.loadTextTokenSequence(ps['path_text_tokens']))
            _, sorted_vocab = self.__dp.buildVocab(tts)
            self.__dp.saveVocab(self.__ps['path_vocab'], sorted_vocab)

    def generateDigitTokenSequences(self):
        sorted_vocab = self.__dp.loadVocab(self.__ps['path_vocab'])
        w2i = self.__dp.getWord2Index(sorted_vocab)
        tts = self.__dp.loadTextTokenSequence(self.__ps['path_text_tokens'])
        dts = self.__dp.getDigitTokenSequence(w2i, tts)
        self.__dp.saveDigitTokenSequence(self.__ps['path_digit_tokens'], dts)
        pass
    
    def generateDigitTokenXY(self):
        sorted_vocab = self.__dp.loadVocab(self.__ps['path_vocab'])
        w2i = self.__dp.getWord2Index(sorted_vocab)
        tts = self.__dp.loadTextTokenSequences(self.__ps['path_input_text_tokens'])
        dts = self.__dp.getDigitTokenSequences(w2i, tts)
        if self.__ps['stateful']:
            dts = self._batch_sort_for_stateful_rnn(np.array(dts), self.__ps['batch_size'])
        self.__dp.saveDigitTokenSequences(self.__ps['path_input_digit_tokens'], dts)
        tts = self.__dp.loadTextTokenSequences(self.__ps['path_target_text_tokens'])
        dts = self.__dp.getDigitTokenSequences(w2i, tts)
        if self.__ps['stateful']:
            dts = self._batch_sort_for_stateful_rnn(np.array(dts), self.__ps['batch_size'])
        self.__dp.saveDigitTokenSequences(self.__ps['path_target_digit_tokens'], dts)
        pass
    
    def _batch_sort_for_stateful_rnn(self, sequences, batch_size):
        # Now the tricky part, we need to reformat our data so the first
        # sequence in the nth batch picks up exactly where the first sequence
        # in the (n - 1)th batch left off, as the RNN cell state will not be
        # reset between batches in the stateful model.
        num_batches = sequences.shape[0] // batch_size
        num_samples = num_batches * batch_size
        reshuffled = np.zeros((num_samples, sequences.shape[1]), dtype=np.int32)
        for batch_index in range(batch_size):
            # Take a slice of num_batches consecutive samples
            slice_start = batch_index * num_batches
            slice_end = slice_start + num_batches
            index_slice = sequences[slice_start:slice_end, :]
            # Spread it across each of our batches in the same index position
            reshuffled[batch_index::batch_size, :] = index_slice
        return reshuffled