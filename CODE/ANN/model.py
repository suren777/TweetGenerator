import tensorflow.keras as kf
from CODE.Config.config import *
import numpy as np


# def custom_loss(yTrue,yPred):
#
# 	def

def create_embedding_layer(tokenizer):
	if tokenizer.type == 'char':
		import numpy as np
		embedding_index = tokenizer.encode_dict
		embedding_matrix = np.zeros((tokenizer.size + 1, tokenizer.size), dtype='float32')
		for w, i in embedding_index.items():
			embedding_matrix[i,embedding_index[w]]=1.0
		return kf.layers.Embedding(tokenizer.size + 1,
									tokenizer.size,
									weights=[embedding_matrix],
									input_length=MAX_QUESTION_SIZE,
									trainable=False)
	else:
		return kf.layers.Embedding(tokenizer.size + 1,
								   EMBEDDING_DIM_SIZE,
								   input_length=MAX_QUESTION_SIZE,
								   trainable=True)

def vanilla_model(src_vocab, src_timesteps, tar_timesteps, n_units):
	model = kf.Sequential()
	model.add(kf.layers.Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(kf.layers.Bidirectional(kf.layers.LSTM(n_units)))
	model.add(kf.layers.RepeatVector(tar_timesteps))
	model.add(kf.layers.LSTM(n_units, return_sequences=True))
	model.add(kf.layers.TimeDistributed(kf.layers.Dense(src_vocab, activation='softmax')))
	return model

def adv_model_train(vocab_size):
	#encoder model
	encoder_input = kf.layers.Input(shape=(MAX_QUESTION_SIZE, vocab_size))
	encoder_LSTM = kf.layers.LSTM(256, return_state=True)
	encoder_outputs, encoder_h, encoder_c = encoder_LSTM(encoder_input)
	encoder_states = [encoder_h, encoder_c]
	decoder_input = kf.layers.Input(shape=(MAX_QUESTION_SIZE, vocab_size))
	decoder_LSTM = kf.layers.LSTM(256, return_sequences=True, return_state=True)
	#decoder model
	decoder_out, _, _ = decoder_LSTM(decoder_input, initial_state=encoder_states)
	decoder_dense = kf.layers.Dense(vocab_size, activation='softmax')
	decoder_out = decoder_dense(decoder_out)
	model = kf.models.Model(inputs=[encoder_input, decoder_input], outputs=[decoder_out])
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
	model.summary()
	return model

def adv_model_inf(vocab_size):
	#encoder model
	encoder_input = kf.layers.Input(shape=(MAX_QUESTION_SIZE, vocab_size))
	encoder_LSTM = kf.layers.LSTM(256, return_state=True)
	encoder_outputs, encoder_h, encoder_c = encoder_LSTM(encoder_input)
	encoder_states = [encoder_h, encoder_c]
	decoder_input = kf.layers.Input(shape=(MAX_QUESTION_SIZE, vocab_size))
	decoder_LSTM = kf.layers.LSTM(256, return_sequences=True, return_state=True)
	#decoder model
	decoder_out, _, _ = decoder_LSTM(decoder_input, initial_state=encoder_states)
	decoder_dense = kf.layers.Dense(vocab_size, activation='softmax')
	decoder_out = decoder_dense(decoder_out)
	model = kf.models.Model(inputs=[encoder_input, decoder_input], outputs=[decoder_out])
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
	model.summary()
	return model


def weighted_categorical_crossentropy(weights):

	weights = kf.backend.variable(weights)

	def loss(y_true, y_pred):
		# scale predictions so that the class probas of each sample sum to 1
		y_pred /= kf.backend.sum(y_pred, axis=-1, keepdims=True)
		# clip to prevent NaN's and Inf's
		y_pred = kf.backend.clip(y_pred, kf.backend.epsilon(), 1 - kf.backend.epsilon())
		# calc
		loss = y_true * kf.backend.log(y_pred) * weights
		loss = -kf.backend.sum(loss, -1)
		return loss

	return loss


def new_candidate_model(tokenizer, filename, lstm_hidden=512):

	encoder_input = kf.layers.Input(shape=(None,))
	encoder_masking = kf.layers.Masking(mask_value=0.0)(encoder_input)
	encoder_embed_input = create_embedding_layer(tokenizer)(encoder_masking)
	encoder_LSTM1 = kf.layers.GRU(lstm_hidden,  return_sequences=True)(encoder_embed_input)
	encoder_LSTM = kf.layers.LSTM(lstm_hidden, return_state=True)
	encoder_outputs, encoder_h, encoder_c = encoder_LSTM(encoder_LSTM1)
	encoder_states = [encoder_h, encoder_c]
	decoder_input = kf.layers.Input(shape=(None,))
	decoder_masking = kf.layers.Masking(mask_value=0.0)(decoder_input)
	decoder_embed_input = create_embedding_layer(tokenizer)(decoder_masking)
	decoder_LSTM = kf.layers.LSTM(lstm_hidden, return_sequences=True, return_state=True)
	# decoder model
	decoder_out, _, _ = decoder_LSTM(decoder_embed_input, initial_state=encoder_states)
	decoder_dense = kf.layers.Dense(MAX_VOCAB_SIZE+4, activation='softmax', kernel_regularizer=kf.regularizers.l2(0.01),
									activity_regularizer=kf.regularizers.l1(0.01))

	# decoder_dense = kf.layers.Dense(MAX_VOCAB_SIZE + 4, use_bias=False)
	# normalize = kf.layers.BatchNormalization()(decoder_dense(decoder_out))
	# final = kf.layers.Activation('softmax')(normalize)
	# model = kf.models.Model(inputs=[encoder_input, decoder_input], outputs=[final])

	model = kf.models.Model(inputs=[encoder_input, decoder_input], outputs=[decoder_dense(decoder_out)])

	optimizer = kf.optimizers.RMSprop(lr=0.001, clipnorm=5)
	# optimizer = kf.optimizers.RMSprop(lr=0.001 )

	weights = np.ones((MAX_VOCAB_SIZE+4))
	weights[0:2] = 0
	model.compile(optimizer=optimizer, loss=weighted_categorical_crossentropy(weights))

	encoder_model_inf = kf.models.Model(encoder_input, encoder_states)

	decoder_state_input_h = kf.layers.Input(shape=(lstm_hidden,))
	decoder_state_input_c = kf.layers.Input(shape=(lstm_hidden,))

	decoder_out, decoder_h, decoder_c = decoder_LSTM(decoder_embed_input,
										 initial_state=[decoder_state_input_h, decoder_state_input_c])

	decoder_model_inf = kf.models.Model(inputs=[decoder_input, decoder_state_input_h, decoder_state_input_c],
										outputs=[decoder_dense(decoder_out), decoder_h, decoder_c])

	try:
		model.load_weights(filename.format('train'))
		# print("Found weights, loading")
	except:
		# print("No weights found or smth els")
		pass
	return model, encoder_model_inf, decoder_model_inf