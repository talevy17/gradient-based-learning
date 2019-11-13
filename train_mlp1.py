import mlp1 as model
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


def accuracy_on_dataset(dataset, params, f2I, l2I):
    good = bad = 0.0
    for label, features in dataset:
        x = feats_to_vec(features, f2I)
        y = l2I[label]
        pred = model.predict(x, params)
        if pred == y:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def update(params, gradients, learning_rate):
    W, b, U, b_tag = params
    dW, db, dU, db_tag = [np.dot(learning_rate, g) for g in gradients]
    W, b, U, b_tag = W - dW, b - db, U - dU, b_tag - db_tag
    return [W, b, U, b_tag]


def train_classifier(train_data, dev_data, num_iterations, learning_rate, learning_decay, params, f2I, l2I):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    _params = params
    lr = learning_rate
    for I in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        if I == learning_decay:
            lr /= 10
        for label, features in train_data:
            x = feats_to_vec(features, f2I)  # convert features to a vector.
            y = l2I[label]  # convert the label to number if needed.
            loss, grads = model.loss_and_gradients(x, y, _params)
            cum_loss += loss
            _params = update(_params, grads, lr)

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params, f2I, l2I)
        dev_accuracy = accuracy_on_dataset(dev_data, params, f2I, l2I)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return _params


def main():
    train_data = ut.TRAIN
    l2I = ut.L2I
    f2I = ut.F2I
    i2L = ut.I2L
    dev_data = ut.DEV
    test_data = ut.TEST
    in_dim = len(ut.vocab)
    out_dim = len(l2I)
    hid_dim = 2 ** math.floor(math.log2(in_dim - out_dim))
    # hid_dim = 100
    num_iterations = 10
    learning_rate = 0.01
    learning_decay = 5

    params = model.create_classifier(in_dim, hid_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, learning_decay, params, f2I,
                                      l2I)
    # test_predictions(test_data, trained_params, f2I, i2L)


if __name__ == '__main__':
    main()
