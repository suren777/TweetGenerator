from pickle import dump
from numpy.random import shuffle
from CODE.features.utils import  load_clean_sentences
dataFolder = r"FILES/Datasets/"


# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)


# load dataset
rawDataset = load_clean_sentences(dataFolder+'english-german.pkl')

# reduce dataset size
n_sentences = 10000
dataset = rawDataset[:n_sentences, :]
# random shuffle
shuffle(dataset)
# split into train/test
train, test = dataset[:9000], dataset[9000:]
# save
save_clean_data(dataset, dataFolder + 'english-german-both.pkl')
save_clean_data(train, dataFolder + 'english-german-train.pkl')
save_clean_data(test, dataFolder + 'english-german-test.pkl')