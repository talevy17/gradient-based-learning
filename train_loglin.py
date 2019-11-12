import loglinear as ll
import numpy as np
import random
import utils as ut
STUDENT = {'name': 'Tal Levy',
           'ID': '---'}
def construct_dict_encoders(unique_items):
    item2idx = dict((item, index) for index, item in enumerate(unique_items))
    idx2item = dict((index, item) for index, item in enumerate(unique_items))
    return item2idx, idx2item


def feats_to_vec(features, f2I):
    x_vec  = np.zeros(len(f2I))
    for f in features:
        if f in f2I:
            x_vec[f2I[f]] +=1
    return x_vec


def accuracy_on_dataset(dataset, params,f2I,l2I):
    good = bad = 0.0
    for label, features in dataset:
        x = feats_to_vec(features,f2I)
        y = l2I[label]
        pred = ll.predict(x, params)
        if pred == y:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params, f2I,l2I):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    _params = params
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features,f2I) # convert features to a vector.
            y = l2I[label]                  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x, y, _params)
            cum_loss += loss
            W, b = _params
            Wg, bg = grads
            W -= np.dot(learning_rate, Wg)
            b -= np.dot(learning_rate, bg)
            _params = [W, b]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params,f2I,l2I)
        dev_accuracy = accuracy_on_dataset(dev_data, params,f2I,l2I)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return _params


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    
    # ...

    train_data = ut.TRAIN
    l2I = ut.L2I
    f2I = ut.F2I
    dev_data = ut.DEV
    in_dim = len(ut.vocab)
    out_dim = len(l2I)
    num_iterations =1
    learning_rate = 0.001

    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params, f2I,l2I)

