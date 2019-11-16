# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
import random
from collections import Counter
import numpy as np


def read_data(fname):
    data = []
    with open(fname) as file:
        for line in file:
            label, text = line.strip().lower().split("\t", 1)
            data.append((label, text))
    return data


def text_to_bigrams(text):
    return ["%s%s" % (c1, c2) for c1, c2 in zip(text, text[1:])]


def text_to_unigrams(text):
    return ["%s" % char for char in text]


def get_hidden(x, W, b):
    h = np.dot(x, W) + b
    return np.tanh(h)


def relu(x):
    '''
    relu function get max between(0,x)
    :param x: sumple data.
    :return: max.
    '''
    return max(0, x)


def tanh_derivative(x):
    return 1 - (np.tanh(x) ** 2)


def relu_derivative(x):
    '''
    calculate relu derivative.
    :param x: paramter.
    :return: relu derivative 1 or 0.
    '''
    if x > 0:
        return 1
    return 0


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    ex = np.exp(x - np.max(x))
    x = ex / ex.sum(axis=0)
    return x


TRAIN = [(l, text_to_bigrams(t)) for l, t in read_data("./Dataset/train")]
DEV = [(l, text_to_bigrams(t)) for l, t in read_data("./Dataset/dev")]
TEST = [(l, text_to_bigrams(t)) for l, t in read_data("./Dataset/test")]

TRAIN_UNI = [(l, text_to_unigrams(t)) for l, t in read_data("./Dataset/train")]
DEV_UNI = [(l, text_to_unigrams(t)) for l, t in read_data("./Dataset/dev")]

fc = Counter()
for l, feats in TRAIN:
    fc.update(feats)

fc_uni = Counter()
for l, feats in TRAIN_UNI:
    fc_uni.update(feats)

# 600 most common bigrams in the training set.
vocab = set([x for x, c in fc.most_common(600)])

vocab_uni = set([x for x, c in fc_uni.most_common(600)])

# label strings to IDs
L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in TRAIN]))))}
# IDs to label strings
I2L = {i: l for i, l in enumerate(list(sorted(set([l for l, t in TRAIN]))))}
# feature strings (bigrams) to IDs
F2I = {f: i for i, f in enumerate(list(sorted(vocab)))}

# label strings to IDs
L2I_UNI = {l: i for i, l in enumerate(list(sorted(set([l for l, t in TRAIN_UNI]))))}
# IDs to label strings
I2L_UNI = {i: l for i, l in enumerate(list(sorted(set([l for l, t in TRAIN_UNI]))))}
# feature strings (bigrams) to IDs
F2I_UNI = {f: i for i, f in enumerate(list(sorted(vocab_uni)))}

def xavier_init(in_dim, out_dim = None):
    if out_dim is None:
        return np.random.randn(in_dim)/np.sqrt(in_dim/2)
    return np.random.randn(in_dim,out_dim)/np.sqrt(2/out_dim)