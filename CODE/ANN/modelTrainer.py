from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint
from CODE.features.utils import  *

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
tokenizer = create_tokenizer(dataset.flatten())
vocab_size = len(tokenizer.word_index) + 1
questionLength = max_length(dataset[:, 0])
print('Question Vocabulary Size: %d' % vocab_size)
print('Question Max Length: %d' % (questionLength))
# prepare german tokenizer
answerLength = max_length(dataset[:, 1])
print('Answer Max Length: %d' % (answerLength))

# prepare training data
trainX = encode_sequences(tokenizer, questionLength, train[:, 1])
trainY = encode_sequences(tokenizer, answerLength, train[:, 0])
trainY = encode_output(trainY, vocab_size)
# prepare validation data
testX = encode_sequences(tokenizer, questionLength, test[:, 1])
testY = encode_sequences(tokenizer, answerLength, test[:, 0])
testY = encode_output(testY, vocab_size)

# define model
model = define_model(vocab_size, vocab_size, questionLength, answerLength, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
print(model.summary())
#plot_model(model, to_file='model.png', show_shapes=True)
# fit model
filename = 'FILES/SavedModels/model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)