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
	def __init__(self, vocab_size, location=None):
		# encoder model
		if vocab_size is None:
			return None
		self.vocab_size = vocab_size
		self.encoder_input = kf.layers.Input(shape=(None, vocab_size))
		self.encoder_LSTM = kf.layers.LSTM(256, return_state=True)
		self.encoder_outputs, self.encoder_h, self.encoder_c = self.encoder_LSTM(self.encoder_input)
		self.encoder_states = [self.encoder_h, self.encoder_c]
		self.decoder_input = kf.layers.Input(shape=(None, vocab_size))
		self.decoder_LSTM = kf.layers.LSTM(256, return_sequences=True, return_state=True)
		# decoder model
		self.decoder_out, _, _ = self.decoder_LSTM(self.decoder_input, initial_state=self.encoder_states)
		self.decoder_dense = kf.layers.Dense(vocab_size, activation='softmax')
		self.decoder_out = self.decoder_dense(self.decoder_out)
		self.model = kf.models.Model(inputs=[self.encoder_input, self.decoder_input], outputs=[self.decoder_out])
		self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
		self.model.summary()
		if location is not None:
			kf.backend.clear_session()
			model = kf.models.load_model(location)
			for l, layer in enumerate(model.layers):
				weight = layer.get_weights()
				if weight is not None:
					self.model.layers[l].set_weights(weight)


	def getModel(self):
		return self.model

	def getEncoderModelPrediction(self,x):
		return kf.models.Model(self.encoder_input, self.encoder_states).predict(x)

	def getDecoderModelPrediction(self, x):


		self.decoder_state_input_h = kf.layers.Input(shape=(256,))
		self.decoder_state_input_c = kf.layers.Input(shape=(256,))
		self.decoder_input_states = [self.decoder_state_input_h, self.decoder_state_input_c]

		self.decoder_out, self.decoder_h, self.decoder_c = self.decoder_LSTM(self.decoder_input,
														 initial_state=self.decoder_input_states)

		self.decoder_states = [self.decoder_h, self.decoder_c]
		self.decoder_dense = kf.layers.Dense(self.vocab_size, activation='softmax')
		self.decoder_out = self.decoder_dense(self.decoder_out)

		self.decoder_model_inf = kf.models.Model(inputs=[self.decoder_input] + self.decoder_input_states,
												 outputs=[self.decoder_out] + self.decoder_states)


		return self.decoder_model_inf.predict(x)
