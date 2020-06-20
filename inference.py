import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


def read_helper_file():
    # Read Word Index
    infile = open("files/word_2_indices", 'rb')
    word_index = pickle.load(infile)
    infile.close()

    infile = open("files/indices_2_word", 'rb')
    index_word = pickle.load(infile)
    infile.close()
    return word_index,index_word

# Parameters


def predict_captions(image, model):
    word_index, index_word = read_helper_file()
    start_word = ["<start>"]
    max_len = 40
    while True:
        par_caps = [word_index[i] for i in start_word]
        par_caps = pad_sequences([par_caps], maxlen=max_len, padding='post')
        pred = model.predict([np.array([image]), np.array(par_caps)])
        word_pred = index_word[np.argmax(pred[0])]
        start_word.append(word_pred)

        if word_pred == "<end>" or len(start_word) > max_len:
            break

    return ' '.join(start_word[1:-1])

def createCaption(photo,model):
    model.reset_states()
    word_index, index_word = read_helper_file()
    in_text = 'startseq'
    max_length = 34
    for i in range(max_length):
        sequence = [word_index[w] for w in in_text.split() if w in word_index]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_word[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final