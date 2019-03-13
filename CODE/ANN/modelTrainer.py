from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from CODE.features.utils import  *
from FILES.Config.config import *
# define NMT model
def define_model(src_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(src_vocab, activation='softmax')))
	return model

# load datasets
raw_data_set = r"question-answer{0}.pkl"

dataset = load_clean_sentences(dataFolder + raw_data_set.format('-both'))
train = load_clean_sentences(dataFolder+raw_data_set.format('-train'))
test = load_clean_sentences(dataFolder+raw_data_set.format('-test'))

# prepare tokenizer
tokenizer = get_tokenizer(dataset=dataset.flatten(), max_vocabulary=2000)
vocab_size = tokenizer.num_words

print('Question Vocabulary Size: %d' % vocab_size)



# prepare training data
trainX = encode_sequences(tokenizer, MAX_QUESTION_SIZE, train[:, 1])
trainY = encode_sequences(tokenizer, MAX_ANSWER_SIZE, train[:, 0])
trainY = encode_output(trainY, vocab_size)
# prepare validation data
testX = encode_sequences(tokenizer, MAX_QUESTION_SIZE, test[:, 1])
testY = encode_sequences(tokenizer, MAX_ANSWER_SIZE, test[:, 0])
testY = encode_output(testY, vocab_size)

# define model
model = define_model(vocab_size, MAX_QUESTION_SIZE, MAX_ANSWER_SIZE, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
print(model.summary())
#plot_model(model, to_file='model.png', show_shapes=True)
# fit model
filename = 'FILES/SavedModels/model.{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlyStop = EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint, reduce_lr, earlyStop], verbose=2)