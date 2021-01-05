#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

from sklearn import svm
from elm import classic_ELM


def linear_data(shape=5, bsz=500):
    """Generate linearly separable data

    Args:
        shape (int, optional): dimensions of the data. Defaults to 5.
        bsz (int, optional): batch size. Defaults to 500.

    Returns:
        tuple: (x_train, y_train, x_eval, y_eval)
    """
    x_train = np.random.rand(bsz*10, shape)
    y_train = np.sum(x_train, axis=1) > shape/2
    y_train = y_train.astype(np.float)

    x_eval = np.random.rand(bsz, shape)
    y_eval = np.sum(x_eval, axis=1) > shape/2
    y_eval = y_eval.astype(np.float)
    return x_train, y_train, x_eval, y_eval


def xor_data(shape=2, bsz=500):
    """Generate linearly separable data

    Args:
        shape (int, optional): dimensions of the data. Defaults to and must be 2.
        bsz (int, optional): batch size. Defaults to 500.

    Returns:
        tuple: (x_train, y_train, x_eval, y_eval)
    """
    assert shape == 2
    x_train = np.random.rand(bsz*10, shape)
    y_train = (x_train[:, 0] > 0.5) * (x_train[:, 1] > 0.5)
    y_train = y_train + ((x_train[:, 0] < 0.5) * (x_train[:, 1] < 0.5))
    y_train = y_train.astype(np.float)

    x_eval = np.random.rand(bsz, shape)
    y_eval = (x_eval[:, 0] > 0.5) * (x_eval[:, 1] > 0.5)
    y_eval = y_eval + ((x_eval[:, 0] < 0.5) * (x_eval[:, 1] < 0.5))
    y_eval = y_eval.astype(np.float)
    return x_train, y_train, x_eval, y_eval


def show_plane(x, y, y_infer=None, pic_name='dummy.png'):
    """Draw points in 2-D plane

    Args:
        x (np.ndarray): shape=(bsz, 2), contain a batch of points
        y (np.ndarray): shape=(bsz,), contain a batch of labels
        y (np.ndarray): shape=(bsz,), contain a batch of labels infered by model
        pic_name (str, optional): name of saved figure. Defaults to 'dummy.png'.
    """
    assert x.shape[1] == 2
    assert x.shape[0] == y.shape[0]
    bsz = x.shape[0]
    # Show training set in figure
    if y_infer is None:
        for i in range(0, bsz):
            if y[i]:
                plt.plot(x[i, 0], x[i, 1], 'r.')
            else:
                plt.plot(x[i, 0], x[i, 1], 'g.')
    else:
        for i in range(0, bsz):
            if y[i] == y_infer[i]:
                plt.plot(x[i, 0], x[i, 1], 'g.')
            else:
                plt.plot(x[i, 0], x[i, 1], 'r.')
    plt.savefig(pic_name)


def test_elm(times=10, draw=False):
    """Test for n times to checkout average accuracy with elm

    Args:
        times (int, optional): [description]. Defaults to 10.
    """
    # parameters
    assert times > 0
    shape = 2
    bsz = 1000
    hidden_size = 100
    normal_c = 10

    # main part
    acc_list = []
    start_time = time.time()
    x_test = None
    y_test = None
    y_t = None
    for i in trange(0, times):
        x_train, y_train, x_test, y_test = xor_data(shape, bsz)
        elm = classic_ELM(shape, hidden_size, norm=normal_c)
        elm.train(x_train, y_train)
        y_t = elm.infer(x_test)
        y_t = y_t.astype(np.float)
        acc = np.sum(y_t == y_test) / len(y_t)
        acc_list.append(acc)
    end_time = time.time()
    acc = np.sum(acc_list) / len(acc_list)
    print('ACC:', acc)
    print('Time consuming: %f seconds for %d times' %
          ((end_time - start_time), times))

    # draw last epoch
    if draw:
        show_plane(x_test, y_test, y_t, pic_name='elm.png')
    return acc


def test_svm(times=10, draw=False):
    """Test for n times to checkout average accuracy with svm

    Args:
        times (int, optional): [description]. Defaults to 10.
    """
    # parameters
    assert times > 0
    shape = 2
    bsz = 1000

    # main part
    acc_list = []
    start_time = time.time()
    x_test = None
    y_test = None
    y_t = None
    for i in trange(0, times):
        svm_c = svm.SVC(gamma='scale', kernel='rbf')
        x_train, y_train, x_test, y_test = xor_data(shape, bsz)
        svm_c.fit(x_train, y_train)
        y_t = svm_c.predict(x_test) > 0.5
        y_t = y_t.astype(np.float)
        acc = np.sum(y_t == y_test) / len(y_t)
        acc_list.append(acc)
    end_time = time.time()
    acc = np.sum(acc_list) / len(acc_list)
    print('ACC:', acc)
    print('Time consuming: %f seconds for %d times' %
          ((end_time - start_time), times))

    # draw last epoch
    if draw:
        show_plane(x_test, y_test, y_t, pic_name='svm.png')
    pass


if __name__ == "__main__":
    test_elm(100, False)
