import numpy as np
import random
from . import pcfg


alphabet = 'abcdefghijklmnopqrstuvwxyz@'


def compute_chars(sentences):
    """
    compute approximate of `chars` values used in the [Kernighan, Church, Gale, '90] paper
    """
    uni_char = np.zeros(len(alphabet), dtype=np.float)
    bi_char = np.zeros((len(alphabet), len(alphabet)), dtype=np.float)
    N = 44e6  # original dataset contains 44 million words
    n_words = 0
    for sentence in sentences:
        word_list = sentence.split(' ')
        for word in word_list:
            n_words += 1
            for i in range(-1, len(word)):
                x = word[i] if i > 0 else '@'
                try:
                    id_x = alphabet.index(x)
                    uni_char[id_x] += 1
                except ValueError:
                    continue
                if i < len(word)-1:
                    y = word[i+1]
                    try:
                        id_y = alphabet.index(y)
                        bi_char[id_x, id_y] += 1
                    except ValueError:
                        continue
    return uni_char * (N / n_words), bi_char * (N / n_words)


def create_dicts(sentences):
    # DEL[X, Y] = deletion of Y after X
    DEL = np.array([0, 7, 58, 21, 3, 5, 18, 8, 61, 0, 4, 43, 5, 53, 0, 9, 0, 98, 28, 53, 62, 1, 0, 0, 2, 0, 2, 2,
                    1, 0, 22, 0, 0, 0, 183, 0, 0, 26, 0, 0, 2, 0, 0, 6, 17, 0, 6, 1, 0, 0, 0, 0, 37, 0, 70, 0, 63,
                    0, 0, 24, 320, 0, 9, 17, 0, 0, 33, 0, 0, 46, 6, 54, 17, 0, 0, 0, 1, 0, 12, 0, 7, 25, 45, 0,
                    10, 0, 62, 1, 1, 8, 4, 3, 3, 0, 0, 11, 1, 0, 3, 2, 0, 0, 6, 0, 80, 1, 50, 74, 89, 3, 1, 1, 6,
                    0, 0, 32, 9, 76, 19, 9, 1, 237, 223, 34, 8, 2, 1, 7, 1, 0, 4, 0, 0, 0, 13, 46, 0, 0, 79, 0, 0,
                    12, 0, 0, 4, 0, 0, 11, 0, 8, 1, 0, 0, 0, 1, 0, 25, 0, 0, 2, 83, 1, 37, 25, 39, 0, 0, 3, 0, 29,
                    4, 0, 0, 52, 7, 1, 22, 0, 0, 0, 1, 0, 15, 12, 1, 3, 20, 0, 0, 25, 24, 0, 0, 7, 1, 9, 22, 0, 0,
                    15, 1, 26, 0, 0, 1, 0, 1, 0, 26, 1, 60, 26, 23, 1, 9, 0, 1, 0, 0, 38, 14, 82, 41, 7, 0, 16, 71,
                    64, 1, 1, 0, 0, 1, 7, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                    0, 0, 4, 0, 0, 1, 15, 1, 8, 1, 5, 0, 1, 3, 0, 17, 0, 0, 0, 1, 5, 0, 0, 0, 1, 0, 0, 0, 24, 0, 1,
                    6, 48, 0, 0, 0, 217, 0, 0, 211, 2, 0, 29, 0, 0, 2, 12, 7, 3, 2, 0, 0, 11, 0, 15, 10, 0, 0, 33,
                    0, 0, 1, 42, 0, 0, 0, 180, 7, 7, 31, 0, 0, 9, 0, 4, 0, 0, 0, 0, 0, 21, 0, 42, 71, 68, 1, 160,
                    0, 191, 0, 0, 0, 17, 144, 21, 0, 0, 0, 127, 87, 43, 1, 1, 0, 2, 0, 11, 4, 3, 6, 8, 0, 5, 0, 4,
                    1, 0, 13, 9, 70, 26, 20, 0, 98, 20, 13, 47, 2, 5, 0, 1, 0, 25, 0, 0, 0, 22, 0, 0, 12, 15, 0, 0,
                    28, 1, 0, 30, 93, 0, 58, 1, 18, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 63, 4, 12, 19, 188, 0, 11, 5, 132, 0, 3, 33, 7, 157, 21, 2,
                    0, 277, 103, 68, 0, 10, 1, 0, 27, 0, 16, 0, 27, 0, 74, 1, 0, 18, 231, 0, 0, 2, 1, 0, 30, 30,
                    0, 4, 265, 124, 21, 0, 0, 0, 1, 0, 24, 1, 2, 0, 76, 1, 7, 49, 427, 0, 0, 31, 3, 3, 11, 1, 0,
                    203, 5, 137, 14, 0, 4, 0, 2, 0, 26, 6, 9, 10, 15, 0, 1, 0, 28, 0, 0, 39, 2, 111, 1, 0, 0, 129,
                    31, 66, 0, 0, 0, 0, 1, 0, 9, 0, 0, 0, 58, 0, 0, 0, 31, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0,
                    0, 0, 1, 0, 40, 0, 0, 1, 11, 1, 0, 11, 15, 0, 0, 1, 0, 2, 2, 0, 0, 2, 24, 0, 0, 0, 0, 0, 0, 0,
                    1, 0, 17, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 5, 0, 0, 0, 0, 1, 0, 2, 1, 34, 0, 2,
                    0, 1, 0, 1, 0, 0, 1, 2, 1, 1, 1, 0, 0, 17, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 20, 14, 41, 31, 20, 20, 7, 6, 20, 3, 6, 22, 16,
                    5, 5, 17, 0, 28, 26, 6, 2, 1, 24, 0, 0, 2]).reshape((27, 26))
    # ADD[X, Y] = insertion of Y after X
    ADD = np.array([15, 1, 14, 7, 10, 0, 1, 1, 33, 1, 4, 31, 2, 39, 12, 4, 3, 28, 134, 7, 28, 0, 1, 1, 4, 1, 3, 11,
                    0, 0, 7, 0, 1, 0, 50, 0, 0, 15, 0, 1, 1, 0, 0, 5, 16, 0, 0, 3, 0, 0, 0, 0, 19, 0, 54, 1, 13, 0,
                    0, 18, 50, 0, 3, 1, 1, 1, 7, 1, 0, 7, 25, 7, 8, 4, 0, 1, 0, 0, 18, 0, 3, 17, 14, 2, 0, 0, 9, 0,
                    0, 6, 1, 9, 13, 0, 0, 6, 119, 0, 0, 0, 0, 0, 5, 0, 39, 2, 8, 76, 147, 2, 0, 1, 4, 0, 3, 4, 6,
                    27, 5, 1, 0, 83, 417, 6, 4, 1, 10, 2, 8, 0, 1, 0, 0, 0, 2, 27, 1, 0, 12, 0, 0, 10, 0, 0, 0, 0,
                    0, 5, 23, 0, 1, 0, 0, 0, 1, 0, 8, 0, 0, 0, 5, 1, 5, 12, 8, 0, 0, 2, 0, 1, 1, 0, 1, 5, 69, 2, 3,
                    0, 1, 0, 0, 0, 4, 1, 0, 1, 24, 0, 10, 18, 17, 2, 0, 1, 0, 1, 4, 0, 0, 16, 24, 22, 1, 0, 5, 0,
                    3, 0, 10, 3, 13, 13, 25, 0, 1, 1, 69, 2, 1, 17, 11, 33, 27, 1, 0, 9, 30, 29, 11, 0, 0, 1, 0, 1,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 4, 0, 1, 9, 0,
                    0, 1, 1, 0, 1, 1, 0, 0, 2, 1, 0, 0, 95, 0, 1, 0, 0, 0, 4, 0, 3, 1, 0, 1, 38, 0, 0, 0, 79, 0, 2,
                    128, 1, 0, 7, 0, 0, 0, 97, 7, 3, 1, 0, 0, 2, 0, 11, 1, 1, 0, 17, 0, 0, 1, 6, 0, 1, 0, 102, 44,
                    7, 2, 0, 0, 47, 1, 2, 0, 1, 0, 0, 0, 15, 5, 7, 13, 52, 4, 17, 0, 34, 0, 1, 1, 26, 99, 12, 0, 0,
                    2, 156, 53, 1, 1, 0, 0, 1, 0, 14, 1, 1, 3, 7, 2, 1, 0, 28, 1, 0, 6, 3, 13, 64, 30, 0, 16, 59,
                    4, 19, 1, 0, 0, 1, 1, 23, 0, 1, 1, 10, 0, 0, 20, 3, 0, 0, 2, 0, 0, 26, 70, 0, 29, 52, 9, 1, 1,
                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 15, 2,
                    1, 0, 89, 1, 1, 2, 64, 0, 0, 5, 9, 7, 10, 0, 0, 132, 273, 29, 7, 0, 1, 0, 10, 0, 13, 1, 7, 20,
                    41, 0, 1, 50, 101, 0, 2, 2, 10, 7, 3, 1, 0, 1, 205, 49, 7, 0, 1, 0, 7, 0, 39, 0, 0, 3, 65, 1,
                    10, 24, 59, 1, 0, 6, 3, 1, 23, 1, 0, 54, 264, 183, 11, 0, 5, 0, 6, 0, 15, 0, 3, 0, 9, 0, 0, 1,
                    24, 1, 1, 3, 3, 9, 1, 3, 0, 49, 19, 27, 26, 0, 0, 2, 3, 0, 0, 2, 0, 0, 36, 0, 0, 0, 10, 0, 0,
                    1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 5, 1, 0, 0, 0, 0, 0, 0, 1, 10, 0, 0, 1, 1, 0, 1, 1, 0, 2, 0, 0, 1,
                    1, 8, 0, 2, 0, 4, 0, 0, 0, 0, 0, 18, 0, 1, 0, 0, 6, 1, 0, 0, 0, 1, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0,
                    1, 0, 0, 5, 1, 2, 0, 3, 0, 0, 0, 2, 0, 0, 1, 1, 6, 0, 0, 0, 1, 33, 1, 13, 0, 1, 0, 2, 0, 2, 0,
                    0, 0, 5, 1, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 46, 8, 9, 8, 26, 11, 14,
                    3, 5, 1, 17, 5, 6, 2, 2, 10, 0, 6, 23, 2, 11, 1, 2, 1, 1, 2]).reshape((27, 26))

    # SUB[X, Y] = substitution of X (correct) for Y (incorrect)
    SUB = np.array([0, 0, 7, 1, 342, 0, 0, 2, 118, 0, 1, 0, 0, 3, 76, 0, 0, 1, 35, 9, 9, 0, 1, 0, 5, 0, 0, 0, 9, 9,
                    2, 2, 3, 1, 0, 0, 0, 5, 11, 5, 0, 10, 0, 0, 2, 1, 0, 0, 8, 0, 0, 0, 6, 5, 0, 16, 0, 9, 5, 0, 0,
                    0, 1, 0, 7, 9, 1, 10, 2, 5, 39, 40, 1, 3, 7, 1, 1, 0, 1, 10, 13, 0, 12, 0, 5, 5, 0, 0, 2, 3, 7,
                    3, 0, 1, 0, 43, 30, 22, 0, 0, 4, 0, 2, 0, 388, 0, 3, 11, 0, 2, 2, 0, 89, 0, 0, 3, 0, 5, 93, 0,
                    0, 14, 12, 6, 15, 0, 1, 0, 18, 0, 0, 15, 0, 3, 1, 0, 5, 2, 0, 0, 0, 3, 4, 1, 0, 0, 0, 6, 4, 12,
                    0, 0, 2, 0, 0, 0, 4, 1, 11, 11, 9, 2, 0, 0, 0, 1, 1, 3, 0, 0, 2, 1, 3, 5, 13, 21, 0, 0, 1, 0,
                    3, 0, 1, 8, 0, 3, 0, 0, 0, 0, 0, 0, 2, 0, 12, 14, 2, 3, 0, 3, 1, 11, 0, 0, 2, 0, 0, 0, 103, 0,
                    0, 0, 146, 0, 1, 0, 0, 0, 0, 6, 0, 0, 49, 0, 0, 0, 2, 1, 47, 0, 2, 1, 15, 0, 0, 1, 1, 9, 0, 0,
                    1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 1, 2, 8, 4, 1, 1, 2, 5, 0, 0, 0, 0,
                    5, 0, 2, 0, 0, 0, 6, 0, 0, 0, .4, 0, 0, 3, 2, 10, 1, 4, 0, 4, 5, 6, 13, 0, 1, 0, 0, 14, 2, 5,
                    0, 11, 10, 2, 0, 0, 0, 0, 0, 0, 1, 3, 7, 8, 0, 2, 0, 6, 0, 0, 4, 4, 0, 180, 0, 6, 0, 0, 9, 15,
                    13, 3, 2, 2, 3, 0, 2, 7, 6, 5, 3, 0, 1, 19, 1, 0, 4, 35, 78, 0, 0, 7, 0, 28, 5, 7, 0, 0, 1, 2,
                    0, 2, 91, 1, 1, 3, 116, 0, 0, 0, 25, 0, 2, 0, 0, 0, 0, 14, 0, 2, 4, 14, 39, 0, 0, 0, 18, 0, 0,
                    11, 1, 2, 0, 6, 5, 0, 2, 9, 0, 2, 7, 6, 15, 0, 0, 1, 3, 6, 0, 4, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                    27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 30, 12, 2, 2, 8, 2, 0,
                    5, 8, 4, 20, 1, 14, 0, 0, 12, 22, 4, 0, 0, 1, 0, 0, 11, 8, 27, 33, 35, 4, 0, 1, 0, 1, 0, 27, 0,
                    6, 1, 7, 0, 14, 0, 15, 0, 0, 5, 3, 20, 1, 3, 4, 9, 42, 7, 5, 19, 5, 0, 1, 0, 14, 9, 5, 5, 6, 0,
                    11, 37, 0, 0, 2, 19, 0, 7, 6, 20, 0, 0, 0, 44, 0, 0, 0, 64, 0, 0, 0, 0, 2, 43, 0, 0, 4, 0, 0,
                    0, 0, 2, 0, 8, 0, 0, 0, 7, 0, 0, 3, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 8, 3, 0, 0, 0, 0, 0, 0,
                    2, 2, 1, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 7, 0, 6, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 15, 0, 1, 7, 15, 0, 0,
                    0, 2, 0, 6, 1, 0, 7, 36, 8, 5, 0, 0, 1, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 5, 0, 0, 0, 0,
                    2, 21, 3, 0, 0, 0, 0, 3, 0]).reshape((26, 26))

    # REV[X, Y] = reversal of XY
    REV = np.array([0, 0, 2, 1, 1, 0, 0, 0, 19, 0, 1, 14, 4, 25, 10, 3, 0, 27, 3, 5, 31, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
                    85, 0, 0, 15, 0, 0, 13, 0, 0, 0, 3, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0,
                    0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 4, 5, 0, 0, 0, 0, 60, 0, 0, 21, 6, 16, 11, 2,
                    0, 29, 5, 0, 85, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 15, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0,
                    0, 12, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 15, 8, 31,
                    3, 66, 1, 3, 0, 0, 0, 0, 9, 0, 5, 11, 0, 1, 13, 42, 35, 0, 6, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 12, 20, 0, 1, 0, 4, 0, 0, 0, 0, 0, 1, 3, 0,
                    0, 1, 1, 3, 9, 0, 0, 7, 0, 9, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 4, 0,
                    0, 0, 0, 0, 15, 0, 6, 2, 12, 0, 8, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 6, 4, 0, 0, 0, 0, 0, 0, 5,
                    0, 2, 0, 4, 0, 0, 0, 5, 0, 0, 1, 0, 5, 0, 1, 0, 11, 1, 1, 0, 0, 7, 1, 0, 0, 17, 0, 0, 0, 4, 0,
                    0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 5, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 24, 0, 3, 0, 14, 0, 2, 2, 0, 7, 30,
                    1, 0, 0, 0, 2, 10, 0, 0, 0, 2, 0, 4, 0, 0, 0, 9, 0, 0, 5, 15, 0, 0, 5, 2, 0, 1, 22, 0, 0, 0, 1,
                    3, 0, 0, 0, 16, 0, 4, 0, 3, 0, 4, 0, 0, 21, 49, 0, 0, 4, 0, 0, 3, 0, 0, 5, 0, 0, 11, 0, 2, 0,
                    0, 0, 22, 0, 5, 1, 1, 0, 2, 0, 2, 0, 0, 2, 1, 0, 20, 2, 0, 11, 11, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
                    0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 1, 10,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0]).reshape((26, 26))
    uni_chars, bi_chars = compute_chars(sentences)
    # use add-1 smoothing
    DEL = DEL + 1
    ADD = DEL + 1
    SUB = DEL + 1
    uni_chars = uni_chars + 1
    bi_chars = bi_chars + 1

    dele = np.zeros_like(DEL, dtype=np.float)
    add = np.zeros_like(ADD, dtype=np.float)
    sub = np.zeros_like(SUB, dtype=np.float)
    for i in range(len(alphabet)):
        for j in range(len(alphabet) - 1):
            dele[i, j] = DEL[i, j] / bi_chars[i, j]
            add[i, j] = ADD[i, j] / uni_chars[i]
    for i in range(len(alphabet) - 1):
        for j in range(len(alphabet) - 1):
            sub[i, j] = SUB[i, j] / uni_chars[j]
    return sub, dele, add


def levenshtein_dist(s1, s2, dele, add, sub):
    """
    Computes Levenshtein distance, between candidate word s1 and original word s2,
    with associated probability of error
    Note: Levenshtein distance does not take into account reversal, maybe should add it
    """
    m = np.zeros((len(s1)+1, len(s2)+1), dtype=np.int)
    p = np.zeros((len(s1)+1, len(s2)+1), dtype=np.float)
    for i in range(len(s1)+1):
        m[i, 0] = i
        # compute probability for deletion
        if i == 0:
            p[i, 0] = 1
        else:
            ind = alphabet.index('@')
            p[i, 0] = p[i-1, 0] * dele[ind, alphabet.index(s1[i-1])]
    for j in range(len(s2)+1):
        # compute probability for insertion
        if j == 0:
            p[0, j] = 1
        else:
            prev_char = '@' if j == 1 else s2[j-2]
            p[0, j] = p[0, j-1] * add[alphabet.index(prev_char),
                                      alphabet.index(s2[j-1])]
        m[0, j] = j
    for i in range(1, 1+len(s1)):
        for j in range(1, len(s2)+1):
            if s1[i-1] == s2[j-1]:
                k = np.argmin([m[i-1, j] + 1, m[i, j-1] + 1, m[i-1, j-1]])
                if k == 0:
                    # deletion
                    m[i, j] = m[i-1, j] + 1
                    prev_char = '@' if j == 1 else s2[j-2]
                    p[i, j] = p[i-1, j] * dele[alphabet.index(prev_char), alphabet.index(s1[i-1])]
                elif k == 1:
                    # insertion
                    m[i, j] = m[i, j-1] + 1
                    prev_char = '@' if j == 1 else s2[j-2]
                    p[i, j] = p[i, j-1] * add[alphabet.index(prev_char), alphabet.index(s2[j-1])]
                else:
                    # no mistake
                    m[i, j] = m[i-1, j-1]
                    p[i, j] = p[i-1, j-1]
            else:
                k = np.argmin([m[i-1, j] + 1, m[i, j-1] + 1, m[i-1, j-1] + 1])
                if k == 0:
                    # deletion
                    m[i, j] = m[i-1, j] + 1
                    prev_char = '@' if j == 1 else s2[j-2]
                    p[i, j] = p[i-1, j] * dele[alphabet.index(prev_char), alphabet.index(s1[i-1])]
                elif k == 1:
                    # insertion
                    m[i, j] = m[i, j-1] + 1
                    prev_char = '@' if j == 1 else s2[j-2]
                    p[i, j] = p[i, j-1] * add[alphabet.index(prev_char), alphabet.index(s2[j-1])]
                else:
                    # substitution
                    m[i, j] = m[i-1, j-1] + 1
                    p[i, j] = p[i-1, j-1] * sub[alphabet.index(s1[i-1]), alphabet.index(s2[j-1])]
                    # recall that in sub[X, Y], Y is the correct word

    return m[len(s1), len(s2)], p[len(s1), len(s2)]


def candidates(word, vocab, dicts, max_dist=2):
    sub, dele, add = dicts
    candidates = []
    probs = []
    for candidate in vocab:
        dist, prob = levenshtein_dist(word, candidate, dele, add, sub)
        if dist <= max_dist:
            candidates.append(candidate)
            probs.append(prob)
    return candidates, probs


def get_vocab(words):
    vocab = []
    for word_list in words:
        for word in word_list:
            vocab.append(word)
    vocab.append('/////START/////')
    return vocab


def unigram(train_sentence_list):
    occur = {'/////START/////': 0}
    count = 0
    for sentence in train_sentence_list:
        sample = sentence.split(' ')
        occur['/////START/////'] += 1
        count += 1
        for word in sample:
            if word not in occur:
                occur[word] = 0
            occur[word] += 1
            count += 1
    return occur, count


def bigram(label_set, train_sentence_list):
    occur = [{} for _ in label_set] + [{}]
    count = 0
    for sentence in train_sentence_list:
        sample = sentence.split(' ')
        word = label_set.index(sample[0])
        if word not in occur[-1]:
            occur[-1][word] = 0
        occur[-1][word] += 1
        count += 1
        for i in range(0, len(sample)-1):
            id_word1 = label_set.index(sample[i])
            word2 = sample[i+1]
            if word2 not in occur[id_word1]:
                occur[id_word1][word2] = 0
            occur[id_word1][word2] += 1
            count += 1
    return occur, count


def p_li(w1, w2, unigram, bigram, n_unigram, lam):
    label_set = list(unigram.keys())
    id_w1 = label_set.index(w1)
    unigram_count = unigram[w1]
    assert unigram_count > 0, 'Error, no occurence of {} was found'.format(w1)
    bigram_count = 0 if w2 not in bigram[id_w1] else bigram[id_w1][w2]
    p_mle = bigram_count / unigram_count
    p_uni = unigram[w2] / n_unigram
    return lam * p_uni + (1 - lam) * p_mle


def sorted_candidates(candidates, pred, xl_probs, unigram, bigram, n_unigram):
    scores = []
    lam = .7
    for i in range(len(candidates)):
        candidate = candidates[i]
        # take the log-probabilities for numerical reasons
        p_xl_log = np.log(xl_probs[i])
        p_li_log = np.log(p_li(pred, candidate, unigram, bigram, n_unigram, lam))
        scores.append(p_xl_log + p_li_log)
    return scores


if __name__ == '__main__':
    # load sequoia dataset
    dataset = []
    with open('../sequoia-corpus+fct.mrg_strict', 'r') as f:
        for line in f:
            dataset.append(line[:-1])
    n_samples = len(dataset)
    random.shuffle(dataset)
    train_set = dataset[:int(0.8 * n_samples)]
    valid_set = dataset[int(0.8 * n_samples): int(0.9 * n_samples)]
    test_set = dataset[int(0.9 * n_samples):]

    heads, rules, freqs_pos, words, freqs_word, sentences = pcfg.create_pcfg(train_set)

    ######################################################################################

    dicts = create_dicts(sentences)
    vocab = get_vocab(words)
    unigram, n_unigram = unigram(sentences)
    bigram, n_bigram = bigram(vocab, sentences)
    print(levenshtein_dist('actress', 'across', dele, add, sub))
