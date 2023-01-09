import numpy as np

from nnconfig import learning_rate


def field_to_1d(field):
    field_1d = np.array(field).ravel()
    res = []
    for el in field_1d:
        if el == 1:
            res.append(1)
        else:
            res.append(0)
    for el in field_1d:
        if el == 2:
            res.append(1)
        else:
            res.append(0)
    return res


def forward_pass(inp, w_h, b_h, w_o, b_o):
    inp = np.array(inp)
    inp.shape += (1,)
    h = w_h @ inp + b_h
    h = 1 / (1 + np.exp(-h))
    o = w_o @ h + b_o
    o = 1 / (1 + np.exp(-o))
    return o


def error_backprop(inp, chosen, is_right, w_h, b_h, w_o, b_o):
    inp = np.array(inp)
    inp.shape += (1,)

    h = w_h @ inp + b_h
    h = 1 / (1 + np.exp(-h))
    o = w_o @ h + b_o
    o = 1 / (1 + np.exp(-o))

    if is_right:
        right = np.zeros((9, 1))
        right[chosen, 0] = 1
    else:
        right = np.ones((9, 1))
        right[chosen, 0] = 0

    delta_o = right - o
    w_o += learning_rate * delta_o @ np.transpose(h)
    b_o += learning_rate * delta_o

    delta_h = np.transpose(w_o) @ -delta_o * (h * (1 - h))
    w_h -= learning_rate * delta_h @ np.transpose(inp)
    b_h -= learning_rate * delta_h
