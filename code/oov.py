import random
import pickle
from operator import itemgetter
from itertools import islice
import re
import numpy as np

from . import spelling

def tmp():
    with open('../polyglot-fr.pkl', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        words, embeddings = u.load()
        print(words, embeddings)
    # words, embeddings = pickle.load(open('../polyglot-fr.pkl', 'rb'))
    # words, embeddings = pickle.load(open('/home/polyglot/en/words_embeddings_32.pkl', 'rb'))
    print("Emebddings shape is {}".format(embeddings.shape))


    # Special tokens
    Token_ID = {"<UNK>": 0, "<S>": 1, "</S>":2, "<PAD>": 3}
    # ID_Token = {v:k for k,v in Token_ID.iteritems()}

    # Map words to indices and vice versa
    word_id = {w:i for (i, w) in enumerate(words)}
    id_word = dict(enumerate(words))

    # Noramlize digits by replacing them with #
    DIGITS = re.compile("[0-9]", re.UNICODE)

    # Number of neighbors to return.
    k = 5

    knn("Jordan", embeddings, word_id, id_word)
    knn("1986", embeddings, word_id, id_word)
    knn("JAPAN", embeddings, word_id, id_word)


def case_normalizer(word, dictionary):
    """ In case the word is not available in the vocabulary,
     we can try multiple case normalizing procedure.
     We consider the best substitute to be the one with the lowest index,
     which is equivalent to the most frequent alternative."""
    w = word
    lower = (dictionary.get(w.lower(), 1e12), w.lower())
    upper = (dictionary.get(w.upper(), 1e12), w.upper())
    title = (dictionary.get(w.title(), 1e12), w.title())
    results = [lower, upper, title]
    results.sort()
    index, w = results[0]
    if index != 1e12:
        return w
    return word


def normalize(word, word_id):
    """ Find the closest alternative in case the word is OOV."""
    if not word in word_id:
        word = DIGITS.sub("#", word)
    if not word in word_id:
        word = case_normalizer(word, word_id)

    if not word in word_id:
        return None
    return word


def l2_nearest(embeddings, word_index, k):
    """Sorts words according to their Euclidean distance.
       To use cosine distance, embeddings has to be normalized so that their l2 norm is 1."""

    e = embeddings[word_index]
    distances = (((embeddings - e) ** 2).sum(axis=1) ** 0.5)
    sorted_distances = sorted(enumerate(distances), key=itemgetter(1))
    return zip(*sorted_distances[:k])


def knn(word, embeddings, word_id, id_word):
    word = normalize(word, word_id)
    if not word:
        print("OOV word")
        return
    word_index = word_id[word]
    indices, distances = l2_nearest(embeddings, word_index, k)
    neighbors = [id_word[idx] for idx in indices]
    for i, (word, distance) in enumerate(zip(neighbors, distances)):
        print(i, '\t', word, '\t\t', distance)

def get_oov_params():
    pass

def oov(original, predecessor, oov_params):
    """
    in order to compute the probability associated to a spelling mistake
    """
    unigram, bigram, n_unigram, vocab, dicts = oov_params
    max_dist_levenshtein = 2
    # cand, xl_probs = spelling.candidates(original, vocab, dicts, max_dist=max_dist_levenshtein)
    # if len(cand) == 0:
    #
    # scores = spelling.sorted_candidates(cand, predecessor, xl_probs,
    #                                                  unigram, bigram, n_unigram)
    # if


if __name__ == '__main__':
    # load sequoia dataset
    # dataset = []
    # with open('sequoia-corpus+fct.mrg_strict', 'r') as f:
    #     for line in f:
    #         dataset.append(line[:-1])
    # n_samples = len(dataset)
    # random.shuffle(dataset)
    # train_set = dataset[:int(0.8 * n_samples)]
    # valid_set = dataset[int(0.8 * n_samples): int(0.9 * n_samples)]
    # test_set = dataset[int(0.9 * n_samples):]

    # heads, rules, freqs_pos, words, freqs_word, sentences = pcfg.create_pcfg(train_set)
    print('--- OOV MODULE ---')
