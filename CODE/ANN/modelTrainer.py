from CODE.features.utils import  *
from CODE.ANN.model import *
import pandas as pd
from CODE.ANN.dataFeed import DataFeeder

# load datasets
raw_data_set = r"question-answer{0}.csv"

dataset = pd.read_csv(dataFolder + raw_data_set.format('-both'))
# prepare tokenizer
tokenizer = get_char_tokenizer(dataset=dataset.values.flatten())
del dataset


# define model
model = DialogueModel(tokenizer.size)

# fit model
filename = 'FILES/SavedModels/model-{}.hdf5'
# checkpoint = kf.callbacks.ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# earlyStop = kf.callbacks.EarlyStopping(monitor='loss', patience=100)
epochs = 1000
batch_size=1024
dataFeed = DataFeeder(dataFolder + raw_data_set.format('-both'), batch_size, tokenizer)
validation_set = dataFeed.genValBatch()
for i in range(epochs):
    X,Y = dataFeed.genTrainBatch()
    hist = model.model.fit(x=X,
              y=Y,
              epochs=1,
              batch_size=batch_size,
              validation_data = validation_set,
              verbose=0)

    if i > 0:
        if hist.history['val_loss'][0] < val_loss:
            val_loss = hist.history['val_loss'][0]
            model.model.save(filename.format('train'))
            model.decoder_model_inf.save(filename.format('decode'))
            model.encoder_model_inf.save(filename.format('encode'))
            print("New best val_loss:{0} \t on epoch: {1}".format(val_loss,i))
    else:
        val_loss = hist.history['val_loss'][0]