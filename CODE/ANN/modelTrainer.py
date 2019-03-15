from CODE.features.utils import  *
import tensorflow.keras as kf
import pandas as pd
from CODE.ANN.dataFeed import DataFeeder
from CODE.ANN.model import create_embedding_layer
from CODE.ANN.modelInference import decode_seq

raw_data_set = r"question-answer{0}.csv"

dataset = pd.read_csv(dataFolder + raw_data_set.format('-both'))
# prepare tokenizer
tokenizer = get_char_tokenizer(dataset=dataset.values.flatten())
del dataset
filename = 'FILES/SavedModels/model-{}.hdf5'

lstm_hidden = 512

encoder_input = kf.layers.Input(shape=(None,))
encoder_masking = kf.layers.Masking(mask_value=0.0)(encoder_input)
encoder_embed_input = create_embedding_layer(tokenizer)(encoder_masking)
encoder_LSTM = kf.layers.LSTM(lstm_hidden, return_state=True)
encoder_outputs, encoder_h, encoder_c = encoder_LSTM(encoder_embed_input)
encoder_states = [encoder_h, encoder_c]
decoder_input = kf.layers.Input(shape=(None, ))
decoder_masking = kf.layers.Masking(mask_value=0.0)(decoder_input)
decoder_embed_input = create_embedding_layer(tokenizer)(decoder_masking)
decoder_LSTM = kf.layers.LSTM(lstm_hidden, return_sequences=True, return_state=True)
# decoder model
decoder_out, _, _ = decoder_LSTM(decoder_embed_input, initial_state=encoder_states)
decoder_dense = kf.layers.Dense(tokenizer.size, activation='softmax',  kernel_regularizer=kf.regularizers.l2(0.01),
                activity_regularizer=kf.regularizers.l1(0.01))
decoder_out = decoder_dense(decoder_out)
model = kf.models.Model(inputs=[encoder_input, decoder_input], outputs=[decoder_out])
optimizer = kf.optimizers.RMSprop(lr=0.001, clipnorm=5)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
model.summary()


encoder_model_inf = kf.models.Model(encoder_input, encoder_states)

decoder_state_input_h = kf.layers.Input(shape=(lstm_hidden,))
decoder_state_input_c = kf.layers.Input(shape=(lstm_hidden,))
decoder_input_states = [decoder_state_input_h, decoder_state_input_c]

decoder_out, decoder_h, decoder_c = decoder_LSTM(decoder_embed_input,
                                                 initial_state=decoder_input_states)

decoder_states = [decoder_h , decoder_c]
decoder_out = decoder_dense(decoder_out)
decoder_model_inf = kf.models.Model(inputs=[decoder_input] + decoder_input_states,
                          outputs=[decoder_out] + decoder_states )



epochs = 1000
internal_epochs = 1
batch_size = 512
dataFeed = DataFeeder(dataFolder + raw_data_set.format('-both'), batch_size*internal_epochs, tokenizer)
validation_set = dataFeed.genValBatch(batch_size//10)
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
            model.save(filename.format('train'))
            encoder_model_inf.save(filename.format('encode'))
            decoder_model_inf.save(filename.format('decode'))
            print("New best val_loss:{0} \t on epoch: {1}".format(val_loss, i*internal_epochs))
            if i%10 == 0:
                test_seq = pd.read_csv(dataFolder + raw_data_set.format('-train'))[:1]
                print("Input: {}".format(test_seq.values[0, 0]))
                print("Output: {}".format(decode_seq(test_seq)))
                print("Actual: {}".format(test_seq.values[0, 1]))
    else:
        val_loss = hist.history['val_loss'][0]