import numpy as np
import utils as ut

STUDENT = {'name': 'Tal Levy',
           'ID': '---'}


def single_layer_forward(x, W, b):
    z = np.dot(x, W) + b
    return z, np.tanh(z)


def classifier_output(x, params):
    zi = 0
    h = []
    h_curr = x
    num_params = len(params)
    for i in range(0, num_params, 2):
        h.append(h_curr)
        zi, h_curr = single_layer_forward(h_curr, params[i], params[i + 1])
    return ut.softmax(zi), h


def predict(x, params):
    probs, _ = classifier_output(x, params)
    return np.argmax(probs)


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    pred_vec, mem = classifier_output(x, params)
    h = mem
    y_vec = np.zeros(len(pred_vec))
    y_vec[y] = 1
    weights = []
    bias = h[::-1]
    for i in range(len(params) - 2, -1, -2):
        weights.append(params[i])
    gradients = []
    num_weights = len(weights)
    index = 0
    sub = (pred_vec - y_vec).reshape(-1, 1)
    for i in range(num_weights):
        if i != index:
            sub = sub.T.dot((weights[index]).T * ut.tanh_derivative(bias[index])).T
            index += 1
        gb = sub
        gW = np.dot(sub, bias[index].reshape(-1, 1).T)
        gW = gW.T
        gradients.append(gb)
        gradients.append(gW)
    gradients = gradients[::-1]
    loss = 0
    if pred_vec[y] > 0:
        loss = -np.log(pred_vec[y])
    return loss, gradients


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        W = np.random.randn(in_dim, out_dim) * np.sqrt(2 / (out_dim + in_dim))
        b = np.random.randn(out_dim) * np.sqrt(1 / out_dim)
        b.reshape(b.shape[0], 1)
        params.append(W)
        params.append(b)
    return params

