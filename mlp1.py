import numpy as np
import utils as ut

STUDENT = {'name': 'Tal Levy',
           'ID': '---'}


def get_hidden(x, W, b):
    h = np.dot(x, W) + b
    return np.tanh(h)


def classifier_output(x, params):
    W, b, U, b_tag = params
    h = get_hidden(x, W, b)
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
    pred_vec = classifier_output(x, params)
    y_vec = np.zeros(len(pred_vec))
    y_vec[y] = 1
    gb_tag = pred_vec - y_vec
    sub = gb_tag.reshape(-1, 1)
    h = get_hidden(x, W, b)
    gU = (h * sub).T
    dh_dz1 = U.T * ut.tanh_derivative(np.dot(x, W) + b)
    gb = np.dot(sub.T, dh_dz1)[0]
    gW = np.dot(sub.T, dh_dz1).T.dot(x.reshape(-1, 1).T).T
    loss = 0
    if pred_vec[y] > 0:
        loss = -np.log(pred_vec[y])
    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W = np.random.randn(in_dim, hid_dim)
    b = np.random.randn(hid_dim)
    U = np.random.randn(hid_dim, out_dim)
    b_tag = np.random.randn(out_dim)
    return [W, b, U, b_tag]

