from CODE.features.utils import *
import pandas as pd
from CODE.features.dataFeed import DataFeeder
from CODE.ANN.model import new_candidate_model
import numpy as np
from tqdm import tqdm
from CODE.Config.config import *


raw_data_set = r"question-answer{0}.csv"
dataFolder = r"FILES/Datasets/"

dataset = pd.read_csv(dataFolder + raw_data_set.format('-both'))
tokenizer = get_char_tokenizer(dataset=dataset.values.flatten(),tokenType='word')
del dataset
filename = 'FILES/SavedModels/model-{}.hdf5'

epochs = 100000
internal_epochs = 10
batch_size = 512
dataFeed = DataFeeder(dataFolder + raw_data_set.format('-both'), batch_size*4, tokenizer)
dataFeed.nskips = np.random.randint(low=0, high=dataFeed.maxskips)
validation_set = dataFeed.genValBatch(batch_size//10)
count = 0
model, encoder, decoder = new_candidate_model(tokenizer, filename, lstm_hidden=LSTM_HIDDEN_SIZE)

for i in tqdm(range(epochs//internal_epochs)):

    X,Y = dataFeed.genTrainBatch()
    hist = model.fit(x=X,
              y=Y,
              epochs=internal_epochs,
              batch_size=batch_size,
              validation_data = validation_set,
              verbose=2)

    if i > 0:
        if hist.history['val_loss'][-1] <= 1e-2:
            print("Overflow issue -- Terminating")
            break
        if np.isnan( hist.history['val_loss'][-1] ):
            print('Loss is NaN: breaking')
            break
        avg_val_loss = sum(hist.history['val_loss']) / float(len(hist.history['val_loss']))
        if avg_val_loss < val_loss and hist.history['val_loss'][-1] > 0:
            val_loss = avg_val_loss
            model.save_weights(filename.format('train'))
            print("\n\tNew best avg_val_loss:{0} \t on epoch: {1} ".format(avg_val_loss, i*internal_epochs))
            count+=1
            if count == 1:
                id = np.random.randint(0, 99)
                test_seq = pd.read_csv(dataFolder + raw_data_set.format('-both-cache'))[:100]
                enc_sentence = tokenizer.encode_input_sequences([test_seq.values[id, 0]])
                states_val = encoder.predict(enc_sentence)
                target_seq = tokenizer.create_empty_input_ch()
                result = []
                for _ in range(MAX_ANSWER_SIZE):
                    decoder_out, decoder_h, decoder_c = decoder.predict(x=[[target_seq]] + states_val)
                    target_seq = np.argmax(decoder_out[0, -1, 2:])+2
                    if target_seq == tokenizer.encode_dict[tokenizer.esc]:
                        break
                    result += [tokenizer.decode_dict[target_seq]]
                    states_val = [decoder_h, decoder_c]


                print('\n'+" ".join([' - + '] * 10))
                print("\tInput: {}".format(test_seq.values[id, 0]))
                print("\tOutput: {}".format((' ').join(result)))
                print("\tActual: {}".format(test_seq.values[id, 1]))
                print('\n' + " ".join([' - + '] * 10))
                count = 0
    else:
        val_loss = sum(hist.history['val_loss']) / float(len(hist.history['val_loss']))

