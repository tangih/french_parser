import numpy as np
import os
import random
import code.pcfg as pcfg


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
    n_h = 4
    # print(heads[n_h])
    # print(list(zip(freqs[n_h], rules[n_h])))
    # print(sum(freqs[n_h]))
    # print(list(zip(words[n_h], freqs_word[n_h])))
