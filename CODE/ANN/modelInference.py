from CODE.features.utils import *
import tensorflow.keras as kf
from nltk.translate.bleu_score import corpus_bleu
from FILES.Config.config import *
import pandas as pd
from CODE.ANN.model import DialogueModel
from tensorflow.keras import backend



# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate target given source sequence
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)

# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
	actual, predicted = list(), list()
	for i, source in enumerate(sources):
		# translate encoded source text
		source = source.reshape((1, source.shape[0]))
		translation = predict_sequence(model, tokenizer, source)
		raw_target, raw_src = raw_dataset[i]
		if i < 10:
			print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
		actual.append(raw_target.split())
		predicted.append(translation.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

def reply_model(source, model):

	tokenizer = get_char_tokenizer()
	modelDecode = model.getDecoderModel()
	modelEncode = model.getEncoderModel()

	return decode_seq(source, modelEncode, modelDecode, tokenizer)


def decode_seq(inp_seq, encoder_model_inf, decoder_model_inf, tokenizer):
	# Initial states value is coming from the encoder
	enc_sentence = tokenizer.sentence_to_categorical(inp_seq).reshape(1, MAX_QUESTION_SIZE, tokenizer.size)
	backend.clear_session()
	states_val = encoder_model_inf.predict(enc_sentence)
	target_seq = tokenizer.create_empty_input(tokenizer.size)

	translated_sent = ''
	stop_condition = False

	while not stop_condition:

		decoder_out, decoder_h, decoder_c = decoder_model_inf.predict(x=[target_seq] + states_val)

		max_val_index = np.argmax(decoder_out[0, -1, :])
		sampled_fra_char = tokenizer.decode_dict[max_val_index]
		translated_sent += sampled_fra_char

		if ((sampled_fra_char == tokenizer.pad) or (len(translated_sent) > MAX_ANSWER_SIZE)):
			stop_condition = True

		target_seq = np.zeros((1, 1, tokenizer.size))
		target_seq[0, 0, max_val_index] = 1

		states_val = [decoder_h, decoder_c]

	return translated_sent

if __name__ == '__main__':
	# load datasets
	dataFolder = r"FILES/Datasets/"
	raw_data_set = r"question-answer{0}.csv"
	train = pd.read_csv(dataFolder + raw_data_set.format('-train'))[1:1000]
	tokenizer = get_char_tokenizer()

	# load model
	filename = 'FILES/SavedModels/model-train.hdf5'
	model = DialogueModel(location=filename)

	modelDecode = model.getDecoderModel()
	modelEncode = model.getEncoderModel()

	for i in range(1):
		sentence = train.values[i,0]
		result = decode_seq(sentence, modelEncode, modelDecode, tokenizer)
		print("Input: {}".format(sentence))
		print("Output: {}". format(result))
		print("Actual: {}".format(train.values[i, 1]))

