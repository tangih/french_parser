def get_vocab(words):
    vocab = []
    for word_list in words:
        for word in word_list:
            vocab.append(word)
    vocab.append('/////START/////')
    return vocab


def oov(sentence, index, unigram, bigram, n_unigram):
    sentence = ['/////START/////'] + sentence
    lam = 0.5

    """
    in order to compute the probability associated to a spelling mistake 
    """


if __name__ =='__main__':
    vocab = get_vocab(words)
    unigram, n_unigram = unigram(sentences)
    bigram, n_bigram = bigram(vocab, sentences)

