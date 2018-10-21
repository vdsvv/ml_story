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
        self.end_sentence_symbols = '.?!'
        self.transform_table = str.maketrans('', '', string.punctuation)
        self.bad_symbol_table = str.maketrans('', '', '*')
        self.vocab = Counter()
        self.__word2index = None
        self.UNK = 2
        pass

    def buildVocab(self, textTokenSequence, vocab_spec = None):
        vocab = Counter()
        vocab.update(textTokenSequence)
        sorted_vocab = sorted(vocab.items(), key=lambda x: (-x[1], x[0]))
        if vocab_spec != None:
            sorted_vocab = vocab_spec + sorted_vocab
        return vocab, sorted_vocab

    def updateVocab(self):
        [self.vocab.update(sequence) for sequence in self.textTokenSequences]
        self.sorted_vocab = sorted(self.vocab.items(), key=lambda x: (-x[1], x[0]))
        return self.sorted_vocab

    def saveVocab(self, filePath, sorted_vocab):
        with open(filePath, mode='w', encoding='utf-8') as file_obj:
            str_item = ''
            for i, item in enumerate(sorted_vocab):
                if i == 0:
                    str_item = '{0:06}\t{1}'.format(item[1], item[0])
                else:    
                    str_item = '\n{0:06}\t{1}'.format(item[1], item[0])
                file_obj.write(str_item)
        pass

    def loadVocab(self, filePath):
        sorted_vocab = []
        lines = []
        with open(filePath, mode='r', encoding='utf-8') as file_obj:
            lines = file_obj.read().split('\n')
        for line in lines:
            if line == '':
                continue
            count, word = line.split('\t')
            sorted_vocab.append((word,count))
        return sorted_vocab

    def saveTextSequences(self, filePath, textSequences):
        with open(filePath, mode='w', encoding='utf-8') as file_obj:
            for i, sequence in enumerate(textSequences):
                if i == 0:
                    text = sequence
                else:    
                    text = '\n' + sequence
                file_obj.write(text)

    def saveTextTokenSequences(self, filePath, textTokenSequences):
        with open(filePath, mode='w', encoding='utf-8') as file_obj:
            for i, sequence in enumerate(textTokenSequences):
                if i == 0:
                    text = '💛'.join(sequence)
                else:
                    text = '\n' +'💛'.join(sequence)
                file_obj.write(text)

    def saveTextTokenSequence(self, filePath, textTokenSequence):
        with open(filePath, mode='w', encoding='utf-8') as file_obj:
            for i, token in enumerate(textTokenSequence):
                if i == 0:
                    text = token
                else:    
                    text = '💛' + token
                file_obj.write(text)
    
    def loadTextTokenSequence(self, filePath):
        textTokenSequence = []
        sequence = []
        with open(filePath, mode='r', encoding='utf-8') as file_obj:
            sequence = file_obj.read()
        textTokenSequence = sequence.split('💛')
        return textTokenSequence
    
    def saveDigitTokenSequences(self, filePath, digitTokenSequences):
        with open(filePath, mode='w', encoding='utf-8') as file_obj:
            for i, sequence in enumerate(digitTokenSequences):
                str_sequence = [str(token) for token in sequence]
                if i == 0:
                    text = '💛'.join(str_sequence)
                else:    
                    text = '\n' + '💛'.join(str_sequence)
                file_obj.write(text)
    
    def saveDigitTokenSequence(self, filePath, digitTokenSequence):
        with open(filePath, mode='w', encoding='utf-8') as file_obj:
            for i, token in enumerate(digitTokenSequence):
                    if i == 0:
                        text = str(token)
                    else:    
                        text = '💛' + str(token)
                    file_obj.write(text)

    def loadTextSequences(self, filePath):
        with open(filePath, mode='r', encoding='utf-8') as file_obj:
            self.textSequences = file_obj.read().split('\n')
        return self.textSequences

    def loadTextTokenSequences(self, filePath):
        textTokenSequences = []
        sequences = []
        with open(filePath, mode='r', encoding='utf-8') as file_obj:
            sequences = file_obj.read().split('\n')
        [textTokenSequences.append(sequence.split('💛')) for sequence in sequences if sequence != '']
        return textTokenSequences

    def loadDigitTokenSequences(self, filePath):
        digitTokenSequences = []
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
            digitTokenSequences.append(digitTokenSequence)       
        return digitTokenSequences

    def loadDigitTokenSequence(self, filePath):
        digitTokenSequence = []
        sequence = []
        with open(filePath, mode='r', encoding='utf-8') as file_obj:
            sequence = file_obj.read()
        for token in sequence.split('💛'):
            i_token = int(token)
            digitTokenSequence.append(i_token)
        return digitTokenSequence

    def selectTokenSequences(self, tokenSequence, sequenceLength, sequenceStep):
        passes = []
        for offset in range(0, sequenceLength, sequenceStep):
            pass_samples = tokenSequence[offset:]
            num_pass_samples = len(pass_samples) // sequenceLength
            pass_samples = np.resize(pass_samples, (num_pass_samples, sequenceLength))
            passes.append(pass_samples)
        return np.concatenate(passes)

    # index_from==3 vocabulary_size==20 : [0 20 1 40 30 17] => [3 23 4 43 33 20] => [3 4]
    #def modifyDigitTokenSequences(self, vocabulary_size = None, index_from = None):
    #    for digitTokenSequence in self.digitTokenSequences:
    #        self.modifyDigitTokenSequence(digitTokenSequence, vocabulary_size, index_from)
    #    return self.digitTokenSequences

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

    def cropTextLines(self, filePath_in, filePath_out, top_n=None):
        textLines = []
        file_in = open(filePath_in, mode='r', encoding='utf-8-sig')
        file_out = open(filePath_out, mode='w', encoding='utf-8')
        if(top_n == None):
            text = file_in.read()
            file_out.write(text)
        else:
            textLines = file_in.readlines()[0:top_n]
            file_out.writelines(textLines)
        file_in.close()
        file_out.close()
        
    def getTextSentenceSequencesByFile(self, filePath):
        raw_text = ''
        with open(filePath, mode='r', encoding='utf-8-sig') as file_obj:
            raw_text = file_obj.read()
        #raw_text = self.removeBadSymbolsInText(raw_text)
        textSequences = self.getTextSentenceSequencesByText(raw_text)
        return textSequences

    def removeBadSymbolsInText(self, text):
        return text.translate(self.bad_symbol_table)

    def getTextSentenceSequencesByText(self, text):
        sentences = []
        sentence = ''
        previous = ' '
        for current in text:
            if current not in self.end_sentence_symbols and previous in self.end_sentence_symbols:
                sentence = sentence.strip(' ')
                sentences.append(sentence)
                sentence = ''
            if current == '\n':
                sentence += ' '
            elif current == '\r':
                sentence += ''
            else:
                sentence += current
            previous = current
        sentence = sentence.strip(' ')
        if sentence != '':
            sentences.append(sentence)
        return sentences    

    def removeExtraWhitespaces(self, textSequences):
        sequences = []
        for textSequence in textSequences:
            sequence = self.removeExtraWhitespacesFromText(textSequence)
            sequences.append(sequence)
        return sequences
        
    def removeExtraWhitespacesFromText(self, text):
        sequence = ''
        previous = ''
        for current in text:
            if current == ' ' and previous == ' ':
                continue
            sequence += current
            previous = current
        return sequence
    
    def getTextTokenSequencesWithPunctuation(self, textSequences):
        textTokenSequences = []
        for textSequence in textSequences:
            textTokenSequence = self.getTextTokenSequenceWithPunctuation(textSequence)
            if len(textTokenSequence) != 0:
                textTokenSequences.append(textTokenSequence)
        return textTokenSequences

    def getTextTokenSequenceWithPunctuation(self, textSequence):
        tokens = []
        token = ''
        #isPreviousAlpha = False
        #isPreviousDigit = False
        #isPreviousPunct = False
        for current in textSequence:
            isCurrentAlpha = current.isalpha() or current == '-'
            isCurrentDigit = current.isdigit()
            isCurrentPunct = not isCurrentDigit and not isCurrentAlpha
            if not isCurrentPunct:
                token += current
            else:
                if token != '':
                    tokens.append(token)
                    token = ''
                if current != ' ':
                    tokens.append(current)
        if token != '':
            tokens.append(token)
        return tokens

    # I: [Полвека][поэзии][Глава][2][У][самого][развилка][,][там][,]
    # O: [У][самого][развилка][,][там][,]
    def removeChapterName(self, textTokenSequences):
        textTokenSequences_new = []
        for textTokenSequence in textTokenSequences:
            needRemove = False
            for i, token in enumerate(textTokenSequence):
                if token == 'Глава' or token == 'ГЛАВА':
                    needRemove = True
                    break
            if needRemove:
                textTokenSequences_new.append(textTokenSequence[i+2:])
            else:
                textTokenSequences_new.append(textTokenSequence)        
        return textTokenSequences_new
    
    
    
    # I: [датурового][эликсира]🌳[<][от][лат][.][datura][-][дурман][.][>]🌳[,][но][девочка]
    # O: [датурового][эликсира][,][но][девочка]
    def removeComment(self, textTokenSequence):
        textTokenSequence_new = []
        openComment_1 = False
        for token in textTokenSequence:
            if token == '<':
                openComment_1 = True
            if not openComment_1:
                textTokenSequence_new.append(token)
            if token == '>':
                openComment_1 = False
        return textTokenSequence_new

    # I: [🌳[[-][Нет][.]]🌳[[-][Хорошо][.][Знаешь][что][?]]🌳]
    # O: [[-][Нет][.][-][Хорошо][.][Знаешь][что][?]]
    def concatSentences(self, textTokenSequences):
        sequence_new = []
        for textTokenSequence in textTokenSequences:
            for token in textTokenSequence:
                sequence_new.append(token)
        return sequence_new


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

    def getDigitTokenSequences(self, word2index, textTokenSequences):
        digitTokenSequences = []
        for textTokenSequence in textTokenSequences:
            sequence = self.getDigitTokenSequence(word2index, textTokenSequence)
            digitTokenSequences.append(sequence)
        return digitTokenSequences

    def getDigitTokenSequence(self, word2index, textTokenSequence):
        #sequence = [self.UNK if word2index.get(token) == None else word2index.get(token) for token in textTokenSequence]
        #sequence = [word2index.get(token) for token in textTokenSequence]
        sequence = [word2index.get(token) for token in textTokenSequence if word2index.get(token) != None]
        return sequence

    def getWord2Index(self, sorted_vocab):
        word2index = {}
        for i,word in enumerate(sorted_vocab):
            word2index[word[0]] = i
        return word2index

    def getIndex2Word(self, sorted_vocab):
        index2word = {}
        for i,word in enumerate(sorted_vocab):
            index2word[i] = word[0]
        return index2word

    def isCyryllic(self, token):
        for c in token:
            if c not in self.allowed_chars:
                return False
        return True 

    def getFileName(self, filePath):
        return path.splitext(path.basename(filePath))[0]