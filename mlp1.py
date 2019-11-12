import numpy as np
import utils as ut

STUDENT = {'name': 'Tal Levy',
           'ID': '---'}


def classifier_output(x, params):
    W, b, U, b_tag = params
    h = np.dot(x, W) + b
    g = np.vectorize(ut.relu)
    h = g(h)
    probs = np.dot(h, U) + b_tag
    return ut.softmax(probs)


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    W, b, U, b_tag = params
    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    params = []
    return params

