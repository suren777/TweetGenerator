import tensorflow.keras as kf
from FILES.Config.config import *

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

class DialogueModel():
	def __init__(self, vocab_size=None, location=None):
		self.encoder_model_inf = None
		self.decoder_model_inf = None
		if location is None:
			# encoder model
			if vocab_size is None:
				return None
			self.vocab_size = vocab_size
			encoder_input = kf.layers.Input(shape=(None, vocab_size))
			encoder_LSTM = kf.layers.LSTM(256, return_state=True)
			encoder_outputs, encoder_h, encoder_c = encoder_LSTM(encoder_input)
			encoder_states = [encoder_h, encoder_c]
			decoder_input = kf.layers.Input(shape=(None, vocab_size))
			decoder_LSTM = kf.layers.LSTM(256, return_sequences=True, return_state=True)
			# decoder model
			decoder_out, _, _ = decoder_LSTM(decoder_input, initial_state=encoder_states)
			decoder_dense = kf.layers.Dense(vocab_size, activation='softmax')
			decoder_out = decoder_dense(decoder_out)
			self.model = kf.models.Model(inputs=[encoder_input, decoder_input], outputs=[decoder_out])
			self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
			self.model.summary()
		else:
			self.model=kf.models.load_model(location)
			self.vocab_size = self.model.layers[0].input_shape[2]
			self.encoder_model_inf = self.getEncoderModel()
			self.decoder_model_inf = self.getDecoderModel()

	def getModel(self):
		return self.model

	def getEncoderModel(self):
		if self.encoder_model_inf is None:
			encoder_input = kf.layers.Input(shape=(None, self.vocab_size))
			encoder_LSTM = kf.layers.LSTM(256, return_state=True)
			encoder_outputs, encoder_h, encoder_c = encoder_LSTM(encoder_input)
			encoder_states = [encoder_h, encoder_c]
			self.encoder_model_inf = kf.models.Model(encoder_input, encoder_states)
			self.encoder_model_inf.layers[-1].set_weights(self.model.layers[2].get_weights())
		return self.encoder_model_inf

	def getDecoderModel(self):

		if self.decoder_model_inf is None:
			decoder_state_input_h = kf.layers.Input(shape=(256,))
			decoder_state_input_c = kf.layers.Input(shape=(256,))
			decoder_input_states = [decoder_state_input_h, decoder_state_input_c]

			decoder_input = kf.layers.Input(shape=(None, self.vocab_size))
			decoder_LSTM = kf.layers.LSTM(256, return_sequences=True, return_state=True)
			decoder_out, decoder_h, decoder_c = decoder_LSTM(decoder_input,
															 initial_state=decoder_input_states)

			decoder_states = [decoder_h, decoder_c]
			decoder_dense = kf.layers.Dense(self.vocab_size, activation='softmax')
			decoder_out = decoder_dense(decoder_out)

			self.decoder_model_inf = kf.models.Model(inputs=[decoder_input] + decoder_input_states,
													 outputs=[decoder_out] + decoder_states)
			self.decoder_model_inf.layers[-2].set_weights(self.model.layers[-2].get_weights())
			self.decoder_model_inf.layers[-1].set_weights(self.model.layers[-1].get_weights())

		return self.decoder_model_inf
