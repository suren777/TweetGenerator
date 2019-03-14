import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
import  pandas as pd


dataFolder = r"FILES/Datasets/"

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# split a loaded document into sentences
def to_pairs(doc,splitToken='\t'):
	lines = doc.strip().split('\n')
	pairs = [line.split(splitToken) for line in lines]
	return pairs

# clean a list of lines
def clean_pairs(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for i, line in enumerate(pair):
			# normalize unicode characters
			line = normalize('NFD', str(line)).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			# tokenize on white space
			line = line.split()
			# convert to lowercase
			line = [word.lower() for word in line]
			# remove punctuation from each token
			line = [word.translate(table) for word in line]
			# remove non-printable chars form each token
			line = [re_print.sub('', w) for w in line]
			# remove tokens with numbers in them
			line = [word for word in line if word.isalpha()]
			# store as string
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return array(cleaned)

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

if __name__ == '__main__':
	# load dataset
	filename = "DialogueFrame.csv"
	doc = pd.read_csv(dataFolder+filename)
	# split into english-german pairs
	#pairs = to_pairs(list(doc.values[:,1:]))
	# clean sentences
	clean_pairs = clean_pairs(list(doc.values[:,1:]))
	# save clean pairs to file
	save_clean_data(clean_pairs, dataFolder+'question-answer.pkl')
	# spot check
	for i in range(100):
		print('[%s] => [%s]' % (clean_pairs[i,0][:], clean_pairs[i,1][:]))