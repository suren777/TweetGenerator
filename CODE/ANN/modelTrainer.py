from CODE.features.utils import  *
import pandas as pd
from CODE.ANN.dataFeed import DataFeeder
from CODE.ANN.model import new_candidate_model

raw_data_set = r"question-answer{0}.csv"

dataset = pd.read_csv(dataFolder + raw_data_set.format('-both'))
# prepare tokenizer
tokenizer = get_char_tokenizer(dataset=dataset.values.flatten())
del dataset
filename = 'FILES/SavedModels/model-{}.hdf5'

model, encoder, decoder = new_candidate_model(tokenizer, filename)
epochs = 10000
internal_epochs = 1
batch_size = 1024
dataFeed = DataFeeder(dataFolder + raw_data_set.format('-both'), batch_size*internal_epochs, tokenizer)
dataFeed.nskips = np.random.randint(low=0, high=dataFeed.maxskips)
validation_set = dataFeed.genValBatch(batch_size//10)
count = 0
for i in range(epochs//internal_epochs):
    X,Y = dataFeed.genTrainBatch()
    hist = model.fit(x=X,
              y=Y,
              epochs=internal_epochs,
              batch_size=batch_size,
              validation_data = validation_set,
              verbose=0)

    if i > 0:
        if hist.history['val_loss'][0] <= 0:
            print("Overflow issue -- Terminating")
            break
        if hist.history['val_loss'][0] < val_loss and hist.history['val_loss'][0] > 0:
            val_loss = hist.history['val_loss'][0]
            model.save_weights(filename.format('train'))
            print("New best val_loss:{0} \t on epoch: {1}".format(val_loss, i*internal_epochs))
            count+=1
            if count == 10:
                test_seq = pd.read_csv(dataFolder + raw_data_set.format('-both'))[:1]
                enc_sentence = tokenizer.encode_input_sequences(test_seq)
                states_val = encoder.predict(enc_sentence)
                target_seq = tokenizer.create_empty_input_ch()
                result = ''
                for _ in range(MAX_ANSWER_SIZE):
                    decoder_out, decoder_h, decoder_c = decoder.predict(x=[[target_seq]] + states_val)
                    target_seq = np.argmax(decoder_out[0, -1, :])
                    result += tokenizer.decode_dict[target_seq]
                    states_val = [decoder_h, decoder_c]

                print("Input: {}".format(test_seq.values[0, 0]))
                print("Output: {}".format(result))
                print("Actual: {}".format(test_seq.values[0, 1]))
                count = 0
    else:
        val_loss = hist.history['val_loss'][0]