import pandas as pd

class DataFeeder():
    def __init__(self, fileLoc, batchSize, tokenizer):
        self.dataLocation=fileLoc
        self.batchSize = batchSize
        self.tokenizer = tokenizer
        self.nskips = 0
        self.datasetSize = pd.read_csv(self.dataLocation).shape[0]
        self.maxskips = self.datasetSize // self.batchSize

    def preprocess_data(self, data):
        Xi = self.tokenizer.encode_input_sequences(data.values[:, 0])
        Yi = self.tokenizer.encode_output_sequences(data.values[:, 1])
        Yo = self.tokenizer.encode_output_sequences(data.values[:, 1], pad=False)
        Yo = self.tokenizer.to_categorical_sequences(Yo)
        return [Xi, Yi], Yo

    def genTrainBatch(self):
        data = pd.read_csv(self.dataLocation, skiprows=self.batchSize*self.nskips, nrows=self.batchSize)
        if self.nskips < self.maxskips-1:
            self.nskips+=1
        else:
            self.nskips = 0

        return self.preprocess_data(data)

    def genValBatch(self, size):
        return self.preprocess_data(pd.read_csv(self.dataLocation,
                                                skiprows=self.batchSize*(self.maxskips-1),
                                                nrows=size))
