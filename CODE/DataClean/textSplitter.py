import  pandas as pd
from sklearn.utils import shuffle
dataFolder = r"FILES/Datasets/"

raw_data_set = r"question-answer{0}.csv"

# load dataset
rawDataset = pd.read_csv(dataFolder+raw_data_set.format(''))

# reduce dataset size
shuffle(rawDataset)
# split into train/test
train, test = rawDataset[:-1000], rawDataset[-1000:]
# save

rawDataset.to_csv(dataFolder + raw_data_set.format('-both'), index=False)
test.to_csv(dataFolder + raw_data_set.format('-test'), index=False)
train.to_csv(dataFolder + raw_data_set.format('-train'), index=False)