import random
import pickle
from operator import itemgetter
from itertools import islice
import re
import numpy as np

import spelling


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
    DIGITS = re.compile("[0-9]", re.UNICODE)

    if not word in word_id:
        word = DIGITS.sub("#", word)
    if not word in word_id:
        word = case_normalizer(word, word_id)

    if not word in word_id:
        return None
    return word


def compute_embeddings(vocab):
    """
    compute embeddings for our SEQUOIA lexicon
    """
    with open('../polyglot-fr.pkl', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        words, embeddings = u.load()
        print(words, embeddings)
    # words, embeddings = pickle.load(open('../polyglot-fr.pkl', 'rb'))
    # words, embeddings = pickle.load(open('/home/polyglot/en/words_embeddings_32.pkl', 'rb'))
    print("Emebddings shape is {}".format(embeddings.shape))

    # Map words to indices and vice versa
    word_id = {w: i for (i, w) in enumerate(words)}
    id_word = dict(enumerate(words))

    voc_embed = []
    for word in vocab:
        norm_word = normalize(word, word_id)
        voc_embed.append(embeddings[word_id[norm_word]])
    return voc_embed, (words, embeddings)


def knn(word, embeddings, word_id, voc_embed, k=5):
    norm_word = normalize(word, word_id)
    target = embeddings[word_id[norm_word]]
    distances = (((voc_embed - target) ** 2).sum(axis=1) ** 0.5)
    sorted_distances = sorted(enumerate(distances), key=itemgetter(1))
    indices, distances = zip(*sorted_distances[:k])
    return indices, distances


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
