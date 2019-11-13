import numpy as np
import utils as ut

STUDENT = {'name': 'Tal Levy',
           'ID': '---'}


def get_hidden(x, W, b):
    h = np.dot(x, W) + b
    g = np.vectorize(ut.relu)
    return g(h)


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


def calc_grad(W, pred_vec, x, y):
    rows = W.shape[0]
    cols = W.shape[1]
    gW = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            gW[i, j] = -x[i] * ((y == j) - pred_vec[j])
    return gW


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
    h = get_hidden(x, W, b)
    gU = calc_grad(U, pred_vec, h, y)
    # gU = np.dot(get_hidden(x, W, b), gb_tag)
    dl_dh = np.dot(U, gb_tag)
    relu_der = np.vectorize(ut.relu_derivative)
    dh_dz1 = relu_der(np.dot(x, W) + b)
    gb = np.dot(dl_dh, dh_dz1)
    gW = np.dot(np.dot(dl_dh, x), dh_dz1)
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
    W = np.zeros((in_dim, hid_dim))
    b = np.zeros(hid_dim)
    U = np.zeros((hid_dim, out_dim))
    b_tag = np.zeros(out_dim)
    return [W, b, U, b_tag]

