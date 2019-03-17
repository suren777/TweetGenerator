import numpy as np
from CODE.Config.config import  *

class Tokenizer():
    def __init__(self, dataset, type='char'):
        self.pad = '<pad>'
        self.unk = '<unk>'
        self.eos = '</s>'
        self.items = [self.pad, self.eos, self.unk]
        self.type = type
        if type == 'char':
            self.items += list(set(unique for charlist in dataset for unique in str(charlist)))
        else:
            from collections import Counter
            cnt = Counter(list(unique for charlist in dataset for unique in str(charlist).split(' ')))
            self.items += list(w for w, _ in  cnt.most_common(MAX_VOCAB_SIZE))

        self.size = len(self.items)
        self.encode_dict = dict()
        self.decode_dict = dict()
        for i, w in enumerate(self.items):
            self.encode_dict[w] = i
            self.decode_dict[i] = w

    def encode_input(self, sentence):
        if self.type  == 'word':
            sentence = str(sentence).split()
            max_len = min(len(sentence), MAX_QUESTION_SIZE - 1)
        else:
            max_len = min(len(str(sentence)), MAX_QUESTION_SIZE - 1)
        res = np.ones((MAX_QUESTION_SIZE))*self.encode_dict[self.pad]
        res[-1] = self.encode_dict[self.eos]

        res[-max_len-1:-1] = [self.encode_dict.get(s, self.encode_dict[self.unk]) for s in str(sentence)[:max_len]]
        return res

    def encode_input_sequences(self, sequences):
        res = np.array([self.encode_input(a) for a in sequences])
        return res

    def encode_output(self, sentence, pad=True):
        if self.type == 'word':
            sentence = str(sentence).split()
            max_len = min(len(sentence), MAX_QUESTION_SIZE - 1)
        else:
            max_len = min(len(str(sentence)), MAX_QUESTION_SIZE - 1)
        res = np.ones((MAX_QUESTION_SIZE)) * self.encode_dict[self.pad]
        if pad:
            res[0] = self.encode_dict[self.eos]
            res[1:max_len] = [self.encode_dict.get(s, self.encode_dict[self.unk]) for s in str(sentence)[0:max_len-1]]
        else:
            res[0:max_len] = [self.encode_dict.get(s, self.encode_dict[self.unk]) for s in str(sentence)[0:max_len]]
        return res

    def encode_output_sequences(self, sequences, pad=True):
        res = np.array([self.encode_output(a, pad) for a in sequences])
        return res

    def decode(self, sentence):
        return [self.decode_dict[a] for a in sentence]

    def to_categorical(self, sentence):
        res = np.zeros((sentence.shape[0], self.size))
        for i, k in enumerate(sentence):
            if k!=0:
                res[i, int(k)] = 1
        return res

    def to_categorical_sequences(self, sequences):
        return np.array([self.to_categorical(s) for s in sequences])

    def create_empty_input(self, size):
        res = np.zeros((1, 1, size))
        res[0,0,self.encode_dict[self.eos]]=1
        return res

    def create_empty_input_ch(self):
        return self.encode_dict[self.eos]

    def from_categorical(self, word):
        return np.argmax(word)

    def sentence_to_categorical(self,sentence):
        self.encode_input(sentence)
        return self.to_categorical(self.encode_input(sentence))

    def sentences_to_categorical_seq(self, sequences):
        return np.array([self.sentence_to_categorical(s) for s in sequences])