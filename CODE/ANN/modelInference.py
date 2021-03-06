from CODE.features.utils import *
from CODE.Config.config import *
import pandas as pd
from CODE.ANN.model import new_candidate_model
import numpy as np


def reply_model(source):
	return decode_seq(source)


def decode_seq(inp_seq):
	# Initial states value is coming from the encoder
	tokenizer = get_char_tokenizer()
	filename = 'FILES/SavedModels/model-{}.hdf5'
	_, encoder, decoder = new_candidate_model(tokenizer, filename, lstm_hidden=LSTM_HIDDEN_SIZE)

	enc_sentence = tokenizer.encode_input_sequences(inp_seq)
	states_val = encoder.predict(enc_sentence)
	target_seq = tokenizer.create_empty_input_ch()

	message = []

	for _ in range(MAX_ANSWER_SIZE):
		decoder_out, decoder_h, decoder_c = decoder.predict(x=[[target_seq]] + states_val)
		target_seq = np.argmax(decoder_out[0, -1, 2:])+2
		if target_seq == tokenizer.encode_dict[tokenizer.esc]:
			break
		message += [tokenizer.decode_dict[target_seq]]
		states_val = [decoder_h, decoder_c]

	return (' ').join(message)

if __name__ == '__main__':
	# load datasets
	dataFolder = r"FILES/Datasets/"
	raw_data_set = r"question-answer{0}.csv"
	train = pd.read_csv(dataFolder + raw_data_set.format('-both-cache'))[:100]

	# load model
	filename = 'FILES/SavedModels/model-{}.hdf5'


	for i in range(10):
		sentence = [train.values[i, 0]]
		result = decode_seq(sentence)
		print(" ".join([' - + ']*10))
		print("Input: {}".format(sentence))
		print("Output: {}". format(result))
		print("Actual: {}".format(train.values[i, 1]))

