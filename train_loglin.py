import loglinear as ll
import numpy as np
import random
import utils as ut

STUDENT = {'name': 'Tal Levy',
           'ID': '---'}


def feats_to_vec(features):
    f2I = ut.F2I
    x_vec = np.zeros(len(f2I))
    for f in features:
        if f in f2I:
            x_vec[f2I[f]] += 1
    return x_vec


def test_predictions(dataset, params):
    with open('test.pred', 'w') as file:
        for index, (label, features) in enumerate(dataset):
            x = feats_to_vec(features)
            pred = ll.predict(x, params)
            if not index == 0:
                file.write('\n')
            file.write(ut.I2L[pred])


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        x = feats_to_vec(features)
        y = ut.L2I[label]
        pred = ll.predict(x, params)
        if pred == y:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
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
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = ut.L2I[label]  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x, y, _params)
            cum_loss += loss
            W, b = _params
            Wg, bg = grads
            W -= np.dot(lr, Wg)
            b -= np.dot(lr, bg)
            _params = [W, b]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return _params


def bigram_model():
    print("The model will be trained on bigrams:")
    train_data = ut.TRAIN
    dev_data = ut.DEV
    test_data = ut.TEST
    in_dim = len(ut.vocab)
    out_dim = len(ut.L2I)
    num_iterations = 10
    learning_rate = 0.01

    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
    # test_predictions(test_data, trained_params)


def unigram_model():
    print("The model will be trained on unigrams:")
    train_data = ut.TRAIN_UNI
    dev_data = ut.DEV_UNI
    # test_data = ut.TEST
    in_dim = len(ut.vocab_uni)
    out_dim = len(ut.L2I_UNI)
    num_iterations = 10
    learning_rate = 0.01

    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
    # test_predictions(test_data, trained_params, f2I, i2L)


if __name__ == '__main__':
    bigram_model()
    # unigram_model()
