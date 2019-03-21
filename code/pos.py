from . import oov

def get_pos(sentence, heads, words, freqs_word, pos_vocab):
    sentence = sentence.split(' ')
    for i, word in enumerate(sentence):
        if word in pos_vocab:
            pos = pos_vocab[word]
        else:
            predecessor = '/////START/////' if i == 0 else sentence[i-1]
            pos = oov.oov(word, predecessor, oov_params)
