import pandas as pd

class DataFeeder():
    def __init__(self, fileLoc, batchSize, tokenizer):
        self.dataLocation=fileLoc
        self.batchSize = batchSize
        self.tokenizer = tokenizer
        self.nskips = 0
        self.datasetSize = pd.read_csv(self.dataLocation).shape[0]
        self.maxskips = self.datasetSize // self.batchSize


    def genTrainBatch(self):
        data = pd.read_csv(self.dataLocation, skiprows=self.batchSize*self.nskips, nrows=self.batchSize)
        if self.nskips < self.maxskips-1:
            self.nskips+=1
        else:
            self.nskips = 0

        Xi = self.tokenizer.sentences_to_categorical_seq(data.values[:, 0])
        Yi = self.tokenizer.encode_output_sequences(data.values[:, 1])
        Yi = self.tokenizer.to_categorical_sequences(Yi)
        Yo = self.tokenizer.encode_output_sequences(data.values[:, 1], pad=False)
        Yo =self.tokenizer.to_categorical_sequences(Yo)
        return [Xi, Yi], Yo

    def genValBatch(self,size):
        data = pd.read_csv(self.dataLocation, skiprows=self.batchSize*(self.maxskips-1), nrows=size)

        Xi = self.tokenizer.sentences_to_categorical_seq(data.values[:, 0])
        Yi = self.tokenizer.encode_output_sequences(data.values[:, 1])
        Yi = self.tokenizer.to_categorical_sequences(Yi)
        Yo = self.tokenizer.encode_output_sequences(data.values[:, 1], pad=False)
        Yo =self.tokenizer.to_categorical_sequences(Yo)
        return [Xi, Yi], Yo

