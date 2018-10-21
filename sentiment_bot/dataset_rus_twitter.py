from data_parser_x import DataParser
import random

class DatasetRusTwitter:
    def __init__(self, vocabulary_size=None, index_from=None):
        self.path_raw_neg = './data/negative.csv'
        self.path_raw_pos = './data/positive.csv'
        self.path_tss_neg = './data/negative_tss.txt'
        self.path_tss_pos = './data/positive_tss.txt'
        self.path_ttss_neg = './data/negative_ttss.txt'
        self.path_ttss_pos = './data/positive_ttss.txt'
        self.path_vocab = './data/vocab.txt'
        self.path_dtss_neg = './data/negative_dtss.txt'
        self.path_dtss_pos = './data/positive_dtss.txt'
        self.vocabulary_size = vocabulary_size
        self.index_from = index_from
        self.__dp = DataParser()
    
    def loadData(self):
        self.__dp.loadVocab(self.path_vocab)
        lv = len(self.__dp.sorted_vocab)
        self.__dp.loadDigitTokenSequences(self.path_dtss_neg)
        neg_sequences = self.__dp.modifyDigitTokenSequences(self.vocabulary_size, self.index_from)
        self.__dp.loadDigitTokenSequences(self.path_dtss_pos)
        pos_sequences = self.__dp.modifyDigitTokenSequences(self.vocabulary_size, self.index_from)
        neg_sequences = [(sequence, 0) for sequence in neg_sequences]
        pos_sequences = [(sequence, 1) for sequence in pos_sequences]
        l1 = len(neg_sequences)
        l2 = len(pos_sequences)
        sequences = pos_sequences + neg_sequences
        l3 = len(sequences)
        random.shuffle(sequences)
        X, Y = list(zip(*sequences))
        return (X, Y), (X, Y)
    
    def prepareData(self):
        self.generateTextSequences()
        self.generateTextTokenSequences()
        self.generateVocabulary()
        self.generateDigitTokenSequences()
        
    def textToDigitTokens(self, text):
        self.__dp.loadVocab(self.path_vocab)
        textTokens = self.__dp.getTextTokenSequence(text)
        digitTokens = self.__dp.getDigitTokenSequence(textTokens)
        self.__dp.modifyDigitTokenSequence(digitTokens, self.vocabulary_size, self.index_from)
        return digitTokens

    """
    def getWord2Index(self):
        word2id = self.__dp.getWord2Index()
        if self.index_from != None:
            word2id = {k:(v+self.index_from) for k,v in word2id.items()}
            word2id["<PAD>"] = 0
            word2id["<START>"] = 1
            word2id["<UNK>"] = 2
        if self.vocabulary_size != None:
            self.__dp
        return word2id
    """

    def generateTextSequences(self):
        self.__dp.getTextSequences(self.path_raw_neg)
        self.__dp.saveTextSequences(self.path_tss_neg)
        self.__dp.getTextSequences(self.path_raw_pos)
        self.__dp.saveTextSequences(self.path_tss_pos)

    def generateTextTokenSequences(self):
        self.__dp.loadTextSequences(self.path_tss_neg)
        self.__dp.getTextTokenSequences()
        self.__dp.saveTextTokenSequences(self.path_ttss_neg)
        self.__dp.loadTextSequences(self.path_tss_pos)
        self.__dp.getTextTokenSequences()
        self.__dp.saveTextTokenSequences(self.path_ttss_pos)
    
    def generateVocabulary(self):
        self.__dp.loadTextTokenSequences(self.path_ttss_neg)
        self.__dp.updateVocab()
        self.__dp.loadTextTokenSequences(self.path_ttss_pos)
        self.__dp.updateVocab()
        self.__dp.saveVocab(self.path_vocab)

    def generateDigitTokenSequences(self):
        self.__dp.loadVocab(self.path_vocab)
        self.__dp.loadTextTokenSequences(self.path_ttss_neg)
        self.__dp.getDigitTokenSequences()
        self.__dp.saveDigitTokenSequences(self.path_dtss_neg)
        self.__dp.loadTextTokenSequences(self.path_ttss_pos)
        self.__dp.getDigitTokenSequences()
        self.__dp.saveDigitTokenSequences(self.path_dtss_pos)