import pandas as pd
import os
from CODE.Config.config import *

class DataFeeder():
    def __init__(self, fileLoc, batchSize, tokenizer, max_freq = 3):
        self.dataLocation = fileLoc
        self.cacheLocation = fileLoc.split('.')[0]+'-cache.csv'
        self.batchSize = batchSize
        self.tokenizer = tokenizer
        self.max_freq = max_freq
        self.nskips = 0
        if self.cacheLocation.split('/')[-1] not in os.listdir('FILES/Datasets'):
            self.clean_unk()
        self.datasetSize = pd.read_csv(self.cacheLocation).shape[0]
        self.maxskips = self.datasetSize // self.batchSize

    def clean_unk(self):
        df = pd.read_csv(self.dataLocation)
        from collections import Counter
        index = set()
        for row in df.iterrows():
            if Counter(self.tokenizer.encode_input(row[1][1])).get(2.0,0)>=self.max_freq :
                index.add(row[0])


        df.drop(list(index)).to_csv(self.cacheLocation, index=False)


    def preprocess_data(self, data):
        Xi = self.tokenizer.encode_input_sequences(data.values[:, 0])
        Yi = self.tokenizer.encode_output_sequences(data.values[:, 1])
        Yo = self.tokenizer.encode_output_sequences(data.values[:, 1], pad=False)
        Yo = self.tokenizer.to_categorical_sequences(Yo)
        return [Xi, Yi], Yo

    def genTrainBatch(self):
        data = pd.read_csv(self.cacheLocation, skiprows=self.batchSize*self.nskips, nrows=self.batchSize)
        if self.nskips < self.maxskips-1:
            self.nskips+=1
        else:
            self.nskips = 0

        return self.preprocess_data(data)

    def genValBatch(self, size):
        return self.preprocess_data(pd.read_csv(self.cacheLocation,
                                                skiprows=self.batchSize*(self.maxskips-1),
                                                nrows=size))
