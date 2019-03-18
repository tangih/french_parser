import numpy as np
import os
import random

from code import pcfg
from code import oov
from code import spelling
from code import cyk


if __name__ == '__main__':
    # load sequoia dataset
    dataset = []
    with open('sequoia-corpus+fct.mrg_strict', 'r') as f:
        for line in f:
            dataset.append(line[:-1])
    n_samples = len(dataset)
    random.shuffle(dataset)
    train_set = dataset[:int(0.8*n_samples)]
    valid_set = dataset[int(0.8*n_samples): int(0.9*n_samples)]
    test_set = dataset[int(0.9*n_samples):]

    heads, rules, freqs_pos, words, freqs_word, sentences = pcfg.create_pcfg(train_set)
    count = 0

    heads_ch, rules_ch, probs_ch = pcfg.chomsky_normal_form(heads, rules, freqs_pos)
    print(train_set[0])
    