import tensorflow.keras as kf
from FILES.Config.config import *
import tensorflow.keras.backend as K

# def custom_loss(yTrue,yPred):
#
# 	def

def create_embedding_layer(tokenizer):
	import numpy as np
	embedding_index = tokenizer.encode_dict
	embedding_matrix = np.zeros((tokenizer.size + 1, tokenizer.size), dtype='float32')
	for w, i in embedding_index.items():
		embedding_matrix[i,embedding_index[w]]=1.0
	from tensorflow.keras.layers import Embedding
	return Embedding(tokenizer.size + 1,
								tokenizer.size,
								weights=[embedding_matrix],
								input_length=MAX_QUESTION_SIZE,
								trainable=False)


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


def new_candidate_model(tokenizer, filename, lstm_hidden=512):

	encoder_input = kf.layers.Input(shape=(None,))
	encoder_masking = kf.layers.Masking(mask_value=0.0)(encoder_input)
	encoder_embed_input = create_embedding_layer(tokenizer)(encoder_masking)
	encoder_LSTM = kf.layers.LSTM(lstm_hidden, return_state=True)
	encoder_outputs, encoder_h, encoder_c = encoder_LSTM(encoder_embed_input)
	encoder_states = [encoder_h, encoder_c]
	decoder_input = kf.layers.Input(shape=(None,))
	decoder_masking = kf.layers.Masking(mask_value=0.0)(decoder_input)
	decoder_embed_input = create_embedding_layer(tokenizer)(decoder_masking)
	decoder_LSTM = kf.layers.LSTM(lstm_hidden, return_sequences=True, return_state=True)
	# decoder model
	decoder_out, _, _ = decoder_LSTM(decoder_embed_input, initial_state=encoder_states)
	decoder_dense = kf.layers.Dense(tokenizer.size, activation='softmax', kernel_regularizer=kf.regularizers.l2(0.01),
									activity_regularizer=kf.regularizers.l1(0.01))
	decoder_out = decoder_dense(decoder_out)
	model = kf.models.Model(inputs=[encoder_input, decoder_input], outputs=[decoder_out])
	optimizer = kf.optimizers.RMSprop(lr=0.001, clipnorm=5)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy')

	encoder_model_inf = kf.models.Model(encoder_input, encoder_states)

	decoder_state_input_h = kf.layers.Input(shape=(lstm_hidden,))
	decoder_state_input_c = kf.layers.Input(shape=(lstm_hidden,))
	decoder_input_states = [decoder_state_input_h, decoder_state_input_c]

	decoder_out, decoder_h, decoder_c = decoder_LSTM(decoder_embed_input,
													 initial_state=decoder_input_states)

	decoder_states = [decoder_h, decoder_c]
	decoder_out = decoder_dense(decoder_out)
	decoder_model_inf = kf.models.Model(inputs=[decoder_input] + decoder_input_states,
										outputs=[decoder_out] + decoder_states)

	model.load_weights(filename.format('train'))
	return model, encoder_model_inf, decoder_model_inf