#https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/
#https://azuliadesigns.com/html-character-codes-ascii-entity-unicode-symbols/

import csv
import numpy as np
import string
from collections import Counter
from os import path
import random

class DataParser:
    def __init__(self):
        self.allowed_chars = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
        self.transform_table = str.maketrans('', '', string.punctuation)
        self.vocab = Counter()
        self.__word2index = None
        self.UNK = 2
        pass

    def updateVocab(self):
        [self.vocab.update(sequence) for sequence in self.textTokenSequences]
        self.sorted_vocab = sorted(self.vocab.items(), key=lambda x: (-x[1], x[0]))
        return self.sorted_vocab

    def saveVocab(self, filePath):
        with open(filePath, mode='w', encoding='utf-8') as file_obj:
            for item in self.sorted_vocab:
                str_item = '{0:06}\t{1}\n'.format(item[1], item[0])
                file_obj.write(str_item)
        pass

    def loadVocab(self, filePath):
        self.sorted_vocab = []
        lines = []
        self.vocab.clear()
        with open(filePath, mode='r', encoding='utf-8') as file_obj:
            lines = file_obj.read().split('\n')
        for i, line in enumerate(lines):
            if line == '':
                continue
            count, word = line.split('\t')
            self.sorted_vocab.append((word,count))
        pass

    def saveTextSequences(self, filePath):
        with open(filePath, mode='w', encoding='utf-8') as file_obj:
            for sequence in self.textSequences:
                text = sequence + '\n'
                file_obj.write(text)

    def saveTextTokenSequences(self, filePath):
        with open(filePath, mode='w', encoding='utf-8') as file_obj:
            for sequence in self.textTokenSequences:
                text = '💛'.join(sequence) + '\n'
                file_obj.write(text)
    
    def saveDigitTokenSequences(self, filePath):
        with open(filePath, mode='w', encoding='utf-8') as file_obj:
            for sequence in self.digitTokenSequences:
                str_sequence = [str(token) for token in sequence]
                text = '💛'.join(str_sequence) + '\n'
                file_obj.write(text)
    
    def loadTextSequences(self, filePath):
        with open(filePath, mode='r', encoding='utf-8') as file_obj:
            self.textSequences = file_obj.read().split('\n')
        return self.textSequences

    def loadTextTokenSequences(self, filePath):
        self.textTokenSequences = []
        sequences = []
        with open(filePath, mode='r', encoding='utf-8') as file_obj:
            sequences = file_obj.read().split('\n')
        [self.textTokenSequences.append(sequence.split('💛')) for sequence in sequences if sequence != '']
        return self.textTokenSequences
    
    def loadDigitTokenSequences(self, filePath):
        self.digitTokenSequences = []
        sequences = []
        with open(filePath, mode='r', encoding='utf-8') as file_obj:
            sequences = file_obj.read().split('\n')
        for sequence in sequences:
            if len(sequence) == 0:
                continue
            digitTokenSequence = []
            for token in sequence.split('💛'):
                i_token = int(token)
                digitTokenSequence.append(i_token)
            self.digitTokenSequences.append(digitTokenSequence)       
        return self.digitTokenSequences

    # index_from==3 vocabulary_size==20 : [0 20 1 40 30 17] => [3 23 4 43 33 20] => [3 4]
    def modifyDigitTokenSequences(self, vocabulary_size = None, index_from = None):
        for digitTokenSequence in self.digitTokenSequences:
            self.modifyDigitTokenSequence(digitTokenSequence, vocabulary_size, index_from)
        return self.digitTokenSequences

    def modifyDigitTokenSequence(self, digitTokenSequence, vocabulary_size = None, index_from = None):
        for i, _ in enumerate(digitTokenSequence):
            if index_from != None:
                digitTokenSequence[i] += index_from
            if vocabulary_size != None and digitTokenSequence[i] >= vocabulary_size:
                digitTokenSequence[i] = self.UNK
        return digitTokenSequence    

    def getTextSequences(self, filePath, lines_num=None):
        self.textSequences = []
        with open(filePath, encoding='utf-8') as file_obj:
            reader = csv.reader(file_obj, delimiter=";")
            for i, row in enumerate(reader):
                if (lines_num != None) and (i == lines_num):
                    break
                text = row[3].replace('\n', ' ')
                if(text != ''):
                    self.textSequences.append(text)
        return self.textSequences

    def getTextTokenSequences(self):
        self.textTokenSequences = []
        for textSequence in self.textSequences:
            textTokenSequence = self.getTextTokenSequence(textSequence)
            if len(textTokenSequence) != 0:
                self.textTokenSequences.append(textTokenSequence)
        return self.textTokenSequences
    
    def getTextTokenSequence(self, textSequence):
        tokens = self.splitIntoTokens(textSequence)
        tokens = self.removePunctuations(tokens)
        tokens = self.removeEmpty(tokens)
        tokens = self.removeNonAlphabetic(tokens)
        tokens = self.filterStopWords(tokens)
        tokens = self.filterByLength(tokens, 1, 20)
        tokens = self.toLowerCase(tokens)
        tokens = self.removeNonCyryllic(tokens)
        return tokens
    
    def splitIntoTokens(self, text):
        return text.split()

    def removePunctuations(self, tokens):
        return [w.translate(self.transform_table) for w in tokens]
    
    def removeEmpty(self, tokens):
        return [w for w in tokens if w != '']

    def removeNonAlphabetic(self, tokens):
        return [word for word in tokens if word.isalpha()]
        
    def filterStopWords(self, tokens):
        #stop_words = set(stopwords.words('english'))
        #tokens = [w for w in tokens if not w in stop_words]
        return tokens

    def filterByLength(self, tokens, min_len, max_len):
        return [word for word in tokens if (len(word) >= min_len and len(word) <= max_len)]
        
    def toLowerCase(self, tokens):
        return [word.lower() for word in tokens]

    def removeNonCyryllic(self, tokens):
        return [token for token in tokens if self.isCyryllic(token)]

    def getDigitTokenSequences(self):
        self.digitTokenSequences = []
        for textTokenSequence in self.textTokenSequences:
            sequence = self.getDigitTokenSequence(textTokenSequence)
            self.digitTokenSequences.append(sequence)
        pass

    def getDigitTokenSequence(self, textTokenSequence):
        w2i = self.getWord2Index()
        sequence = [self.UNK if w2i.get(token) == None else w2i.get(token)  for token in textTokenSequence]
        return sequence

    def getWord2Index(self):
        if self.__word2index == None:
            word2index = {}
            for i,word in enumerate(self.sorted_vocab):
                word2index[word[0]] = i
            self.__word2index = word2index
        return self.__word2index

    def getIndex2Word(self):
        index2word = {}
        for i,word in enumerate(self.sorted_vocab):
            index2word[i] = word[0]
        return index2word

    def isCyryllic(self, token):
        for c in token:
            if c not in self.allowed_chars:
                return False
        return True 

    def getFileName(self, filePath):
        return path.splitext(path.basename(filePath))[0]