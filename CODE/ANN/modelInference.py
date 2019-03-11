from CODE.features.utils import  *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu



# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

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

def reply_model(source):
	dataset = load_clean_sentences(dataFolder+'english-german-both.pkl')
	tokenizer = create_tokenizer(dataset[:, 0])
	encodedSequence = encode_sequences(tokenizer,  10, source)
	filename = 'FILES/SavedModels/model.h5'
	model = load_model(filename)
	encodedSequence = encodedSequence[0].reshape((1, encodedSequence[0].shape[0]))
	return predict_sequence(model, tokenizer, encodedSequence)


if __name__ == '__main__':
	# load datasets
	dataFolder = r"FILES/Datasets/"
	dataset = load_clean_sentences(dataFolder+'english-german-both.pkl')
	train = load_clean_sentences(dataFolder+'english-german-train.pkl')
	test = load_clean_sentences(dataFolder+'english-german-test.pkl')

	# prepare english tokenizer
	eng_tokenizer = create_tokenizer(dataset[:, 0])
	eng_vocab_size = len(eng_tokenizer.word_index) + 1
	eng_length = max_length(dataset[:, 0])
	# prepare german tokenizer
	ger_tokenizer = create_tokenizer(dataset[:, 1])
	ger_vocab_size = len(ger_tokenizer.word_index) + 1
	ger_length = max_length(dataset[:, 1])
	# prepare data
	trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
	testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])

	# load model
	filename = 'FILES/SavedModels/model.h5'
	model = load_model(filename)
	# test on some training sequences
	print('train')
	evaluate_model(model, eng_tokenizer, trainX, train)
	# test on some test sequences
	print('test')
	evaluate_model(model, eng_tokenizer, testX, test)