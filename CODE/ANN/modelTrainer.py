import tensorflow.keras as kf
from CODE.features.utils import  *
from FILES.Config.config import *
# define NMT model
def define_model(src_vocab, src_timesteps, tar_timesteps, n_units):
	model = kf.Sequential()
	model.add(kf.layers.Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(kf.layers.Bidirectional(kf.layers.LSTM(n_units)))
	model.add(kf.layers.RepeatVector(tar_timesteps))
	model.add(kf.layers.LSTM(n_units, return_sequences=True))
	model.add(kf.layers.TimeDistributed(kf.layers.Dense(src_vocab, activation='softmax')))
	return model

# load datasets
raw_data_set = r"question-answer{0}.pkl"

dataset = load_clean_sentences(dataFolder + raw_data_set.format('-both'))
train = load_clean_sentences(dataFolder+raw_data_set.format('-train'))
test = load_clean_sentences(dataFolder+raw_data_set.format('-test'))

# prepare tokenizer
tokenizer = get_tokenizer(dataset=dataset.flatten(), max_vocabulary=MAX_VOCAB_SIZE)

print('Question Vocabulary Size: %d' % MAX_VOCAB_SIZE)



# prepare training data
trainX = encode_sequences(tokenizer, MAX_QUESTION_SIZE, train[:, 1])
trainY = encode_sequences(tokenizer, MAX_ANSWER_SIZE, train[:, 0])
trainY = encode_output(trainY, MAX_VOCAB_SIZE)
# prepare validation data
testX = encode_sequences(tokenizer, MAX_QUESTION_SIZE, test[:, 1])
testY = encode_sequences(tokenizer, MAX_ANSWER_SIZE, test[:, 0])
testY = encode_output(testY, MAX_VOCAB_SIZE)

# define model
model = define_model(MAX_VOCAB_SIZE, MAX_QUESTION_SIZE, MAX_ANSWER_SIZE, 256)
sgd = kf.optimizers.RMSprop(lr=0.01, clipnorm=5)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
# summarize defined model
print(model.summary())
#plot_model(model, to_file='model.png', show_shapes=True)
# fit model
filename = 'FILES/SavedModels/model.{epoch:02d}-{loss:.2f}.hdf5'
checkpoint = kf.callbacks.ModelCheckpoint(filename, monitor='loss', verbose=1, save_best_only=True, mode='min')
earlyStop = kf.callbacks.EarlyStopping(monitor='loss', patience=100)


model.fit(trainX, trainY, epochs=10000, batch_size=1024, validation_data=(testX, testY), callbacks=[checkpoint,  earlyStop], verbose=2)