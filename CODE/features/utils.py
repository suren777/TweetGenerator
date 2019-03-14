from pickle import load
from numpy import array
from numpy import argmax
import pickle
from CODE.features.utils import  *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

dataFolder = r"FILES/Datasets/"
# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def get_tokenizer(dataset=None, max_vocabulary=2000, tokenizerFile='tokenizer.pkl'):
	try:
		with open(dataFolder+tokenizerFile, 'rb') as f:
			tokenizer = pickle.load(f)
	except:
		print("No tokenizer found, creating a new one")
		if dataset is not None:
			tokenizer = Tokenizer(num_words=max_vocabulary)
			tokenizer.fit_on_texts(dataset)
			with open(dataFolder + tokenizerFile, 'wb') as f:
				pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
		else:
			tokenizer = None
			print("No dataset found.")
	return tokenizer

# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y




# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None