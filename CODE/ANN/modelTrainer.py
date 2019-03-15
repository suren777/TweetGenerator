from CODE.features.utils import  *
import tensorflow.keras as kf
import pandas as pd
from CODE.ANN.dataFeed import DataFeeder

raw_data_set = r"question-answer{0}.csv"

dataset = pd.read_csv(dataFolder + raw_data_set.format('-both'))
# prepare tokenizer
tokenizer = get_char_tokenizer(dataset=dataset.values.flatten())
del dataset
filename = 'FILES/SavedModels/model-{}.hdf5'

encoder_input = kf.layers.Input(shape=(None, tokenizer.size))
encoder_LSTM = kf.layers.LSTM(256, return_state=True, kernel_regularizer=kf.regularizers.l2(0.01),
                activity_regularizer=kf.regularizers.l1(0.01))
encoder_outputs, encoder_h, encoder_c = encoder_LSTM(encoder_input)
encoder_states = [encoder_h, encoder_c]
decoder_input = kf.layers.Input(shape=(None, tokenizer.size))
decoder_LSTM = kf.layers.LSTM(256, return_sequences=True, return_state=True, kernel_regularizer=kf.regularizers.l2(0.01),
                activity_regularizer=kf.regularizers.l1(0.01))
# decoder model
decoder_out, _, _ = decoder_LSTM(decoder_input, initial_state=encoder_states)
decoder_dense = kf.layers.Dense(tokenizer.size, activation='softmax',  kernel_regularizer=kf.regularizers.l2(0.01),
                activity_regularizer=kf.regularizers.l1(0.01))
decoder_out = decoder_dense(decoder_out)
model = kf.models.Model(inputs=[encoder_input, decoder_input], outputs=[decoder_out])
optimeizer = kf.optimizers.Adam(lr=0.01, clipnorm=5)
model.compile(optimizer=optimeizer, loss='categorical_crossentropy')
model.summary()


encoder_model_inf = kf.models.Model(encoder_input, encoder_states)

decoder_state_input_h = kf.layers.Input(shape=(256,))
decoder_state_input_c = kf.layers.Input(shape=(256,))
decoder_input_states = [decoder_state_input_h, decoder_state_input_c]

decoder_out, decoder_h, decoder_c = decoder_LSTM(decoder_input,
                                                 initial_state=decoder_input_states)

decoder_states = [decoder_h , decoder_c]
decoder_out = decoder_dense(decoder_out)
decoder_model_inf = kf.models.Model(inputs=[decoder_input] + decoder_input_states,
                          outputs=[decoder_out] + decoder_states )



epochs = 10000
batch_size=2048
dataFeed = DataFeeder(dataFolder + raw_data_set.format('-both'), batch_size, tokenizer)
validation_set = dataFeed.genValBatch()
for i in range(epochs):
    X,Y = dataFeed.genTrainBatch()
    hist = model.fit(x=X,
              y=Y,
              epochs=1,
              batch_size=batch_size,
              validation_data = validation_set,
              verbose=0)

    if i > 0:
        if hist.history['val_loss'][0] < val_loss and hist.history['val_loss'][0] > 0:
            val_loss = hist.history['val_loss'][0]
            model.save(filename.format('train'))
            encoder_model_inf.save(filename.format('encode'))
            decoder_model_inf.save(filename.format('decode'))
            print("New best val_loss:{0} \t on epoch: {1}".format(val_loss,i))
    else:
        val_loss = hist.history['val_loss'][0]