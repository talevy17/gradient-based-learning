import mlpn as model
import numpy as np
import random
import utils as ut
import math

STUDENT = {'name': 'Tal Levy',
           'ID': '---'}


def feats_to_vec(features, f2I):
    x_vec = np.zeros(len(f2I))
    for f in features:
        if f in f2I:
            x_vec[f2I[f]] += 1
    return x_vec


def test_predictions(dataset, params, f2I, i2L):
    with open('test.pred', 'w') as file:
        for index, (label, features) in enumerate(dataset):
            x = feats_to_vec(features, f2I)
            pred = model.predict(x, params)
            if not index == 0:
                file.write('\n')
            file.write(i2L[pred])


def accuracy_on_dataset(dataset, params, f2I, l2I, feat_parser=feats_to_vec):
    good = bad = 0.0
    for label, features in dataset:
        x = feat_parser(features, f2I)
        y = l2I[label]
        pred = model.predict(x, params)
        if pred == y:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def train_classifier(train_data, dev_data, params, f2I, l2I, feat_parser=feats_to_vec, learning_rate=1.0,
                     learning_decay=np.inf, num_iterations=30):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    lr = learning_rate
    for I in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        if I == learning_decay or I == learning_decay * 2:
            lr /= 10
        for label, features in train_data:
            x = feat_parser(features, f2I)  # convert features to a vector.
            y = l2I[label]  # convert the label to number if needed.
            loss, grads = model.loss_and_gradients(x, y, params)
            cum_loss += loss
            for i in range(0, len(params), 2):
                params[i] -= learning_rate * grads[i]
                b = params[i + 1]
                params[i + 1] = np.squeeze((b - learning_rate * grads[i + 1].T).T)
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params, f2I, l2I, feat_parser=feat_parser)
        dev_accuracy = accuracy_on_dataset(dev_data, params, f2I, l2I, feat_parser=feat_parser)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


def bigram_model():
    train_data = ut.TRAIN
    l2I = ut.L2I
    f2I = ut.F2I
    i2L = ut.I2L
    dev_data = ut.DEV
    test_data = ut.TEST
    in_dim = len(ut.vocab)
    out_dim = len(l2I)
    epochs = 15
    learning_rate = 0.01
    learning_decay = 5

    params = model.create_classifier([in_dim, 1000, out_dim])
    trained_params = train_classifier(train_data, dev_data, params, f2I, l2I, learning_rate=learning_rate,
                                      learning_decay=learning_decay, num_iterations=epochs)
    # test_predictions(test_data, trained_params, f2I, i2L)


if __name__ == '__main__':
    bigram_model()
