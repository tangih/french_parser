import numpy as np
import os
import random

from code import pcfg
from code import oov
from code import spelling
from code import cyk
from code import pos


if __name__ == '__main__':

    # load sequoia dataset
    print('Loading SEQUOIA dataset')
    dataset = []
    with open('sequoia-corpus+fct.mrg_strict', 'r') as f:
        for line in f:
            dataset.append(line[:-1])
    n_samples = len(dataset)
    random.seed(0)
    random.shuffle(dataset)
    train_set = dataset[:int(0.8*n_samples)]
    valid_set = dataset[int(0.8*n_samples): int(0.9*n_samples)]
    test_set = dataset[int(0.9*n_samples):]

    print('Parsing PCFG from dataset')
    # heads = ['A', 'B', 'C']
    # rules = [[['B'], ['a']], [['b'], ['A', 'C'], ['C']], [['c'], ['f']]]
    # freqs_pos = [[.5, .5], [.33, .33, .33], [.5, .5]]
    heads, rules, freqs_pos, words, freqs_word, sentences = pcfg.create_pcfg(train_set)

    # print('Modifying PCFG into Chomsky normal form')
    # new_rule_list, new_heads = pcfg.chomsky_normal_form(heads, rules, freqs_pos)

    pos_vocab = {}
    for i in range(len(heads)):
        pos = heads[i].lower()
        for j in range(len(words)):
            word = words[i][j]
            pos_vocab[word] = pos
    sentence = 'le petit homme mange'
    print(pos.get_pos(sentence, heads, words, freqs_word, pos_vocab))



    # print('CYK algorithm')
    # cyk.cyk(['ponct', 'npp'], new_rule_list, new_heads)