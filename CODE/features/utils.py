import pickle
from CODE.features.Tokenizer import Tokenizer

tokenFolder = r"FILES/Extra/"
# load a clean dataset
def load_clean_sentences(filename):
    return  pickle.load(open(filename, 'rb'))

def get_char_tokenizer(dataset=None, tokenizerFile='tokenizer_char.pkl', tokenType='char'):
    try:
        with open(tokenFolder+tokenizerFile, 'rb') as f:
            tokenizer = pickle.load(f)
    except:
        print("No tokenizer found, creating a new one")
        if dataset is not None:
            tokenizer = Tokenizer(dataset, tokenType)
            with open(tokenFolder + tokenizerFile, 'wb') as f:
                pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            tokenizer = None
            print("No dataset found.")
    return tokenizer

# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    pickle.dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)
