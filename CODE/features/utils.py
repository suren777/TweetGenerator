from pickle import load
from numpy import array
import numpy as np
import pickle
from FILES.Config.config import  *

dataFolder = r"FILES/Datasets/"
# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))



def get_char_tokenizer(dataset=None, tokenizerFile='tokenizer_char.pkl'):
    try:
        with open(dataFolder+tokenizerFile, 'rb') as f:
            tokenizer = pickle.load(f)
    except:
        print("No tokenizer found, creating a new one")
        if dataset is not None:
            tokenizer = Tokenizer(dataset)
            with open(dataFolder + tokenizerFile, 'wb') as f:
                pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            tokenizer = None
            print("No dataset found.")
    return tokenizer

# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)






# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    pickle.dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

class Tokenizer():
    def __init__(self, dataset):
        self.characters = list(set(unique for charlist in dataset for unique in str(charlist)))
        self.pad = '<pad>'
        self.eos = '</s>'
        self.characters = [self.pad, self.eos]
        self.characters += list(set(unique for charlist in dataset for unique in str(charlist)))
        self.size = len(self.characters)
        self.encode_dict = dict()
        self.decode_dict = dict()
        for i, w in enumerate(self.characters):
            self.encode_dict[w] = i
            self.decode_dict[i] = w

    def encode_input(self, sentence):
        res = np.ones((MAX_QUESTION_SIZE))*self.encode_dict[self.pad]
        res[-1] = self.encode_dict[self.eos]
        max_len = min(len(str(sentence)), MAX_QUESTION_SIZE-1)
        res[-max_len-1:-1] = [self.encode_dict[s] for s in str(sentence)[:max_len]]
        return res

    def encode_input_sequences(self, sequences):
        res = np.array([self.encode_input(a) for a in sequences])
        return res

    def encode_output(self, sentence, pad=True):
        res = np.ones((MAX_QUESTION_SIZE)) * self.encode_dict[self.pad]
        l = min(len(str(sentence)), MAX_QUESTION_SIZE)
        if pad:
            res[0] = self.encode_dict[self.eos]
            res[1:l] = [self.encode_dict[s] for s in str(sentence)[0:l-1]]
        else:
            res[0:l] = [self.encode_dict[s] for s in str(sentence)[0:l]]
        return res

    def encode_output_sequences(self, sequences, pad=True):
        res = np.array([self.encode_output(a,pad) for a in sequences])
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
        res = np.zeros((1,1,size))
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