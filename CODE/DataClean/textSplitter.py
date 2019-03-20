import  pandas as pd
from sklearn.utils import shuffle
from CODE.Config.config import *

dataFolder = r"FILES/Datasets/"
raw_data_set = r"question-answer{0}.csv"

# load dataset
rawDataset = pd.read_csv(dataFolder+raw_data_set.format(''))

# reduce dataset size
shuffle(rawDataset)
# split into train/test
# train, test = rawDataset[:-1000], rawDataset[-1000:]
# save
index = list()
for row in rawDataset.iterrows():
    if row[1][1]==row[1][1] and row[1][0]==row[1][0]:
        if len(row[1][1].split()) < MAX_ANSWER_SIZE:
            index.append(row[0])
    else:
        index.append(row[0])

rawDataset=rawDataset.drop(index)

answers = list()
for answer in rawDataset.values[:,1]:
    arr = answer.split(' ')[:MAX_ANSWER_SIZE]
    answers.append(' '.join(arr))
rawDataset.values[:,1] = answers
rawDataset.to_csv(dataFolder + raw_data_set.format('-both'), index=False)
