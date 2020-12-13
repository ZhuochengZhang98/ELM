#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.core.fromnumeric import shape

# TODO
# 1. 实现高斯核函数
# 2. 对比高斯核函数、随机矩阵下SVM和ELM的分类效果、速度
# 3. 实现多层ELM（待定）
# 4. 使用Bert作为核函数进行情感分类


class ELM:
    def __init__(self, args) -> None:
        pass


class basic_ELM:
    def __init__(self, input_shape, hidden_dim, act='sigmoid') -> None:
        """basic ELM for two class classify

        Args:
            input_shape (int): number of input features
            hidden_dim (int): number of hidden layer
            act (str, optional): activite function. Defaults to 'sigmoid'.
        """
        self.w = np.random.rand(input_shape, hidden_dim)
        self.b = np.random.rand(hidden_dim)
        self.beta = np.random.rand(hidden_dim, 1)
        self.act = self.act_func(act)

    def train(self, x, y):
        """train basic elm

        Args:
            x (numpy.ndarray): (bsz, hidden_dim) train examples
            y (numpy.ndarray): (bsz, ) labels
        """
        assert x.shape[1] == self.w.shape[0]
        x = np.matmul(x, self.w) + self.b
        x = self.act(x)
        self.beta = np.matmul(np.linalg.pinv(x), y)

    def test(self, x):
        """infer via elm

        Args:
            x (numpy.ndarray): (bsz, hidden_dim)

        Returns:
            y (numpy.ndarray): (bsz, ) predict results
        """
        assert x.shape[1] == self.w.shape[0]
        x = np.matmul(x, self.w) + self.b
        x = self.act(x)
        y = np.matmul(x, self.beta)
        return y

    def infer(self, x):
        """infer via elm

        Args:
            x (numpy.ndarray): (bsz, hidden_dim)

        Returns:
            y (numpy.ndarray): (bsz, ) predict labels
        """
        labels = self.test(x) > 0.5
        return labels.astype(np.int)

    def act_func(self, act):
        if act == 'tanh':
            return np.tanh
        elif act == 'sigmoid':
            return lambda x: 1/(1+np.exp(-x))
        else:
            return lambda x: x


class classic_ELM:
    def __init__(self, input_dim, hidden_dim, act='sigmoid', norm=None, classes=2):
        """ELM for classify problem

        Args:
            input_shape (int): number of input features
            hidden_dim (int): number of hidden layer
            act (str, optional): activite function. Defaults to 'sigmoid'.
            norm (float or None, optional): normalize coefficient. Defaults to None.
            classes (int, optional): classes number. Defaults to 2.
        """
        if (input_dim is not None) and (hidden_dim is not None):
            self.use_linear = True
            self.w = np.random.rand(1, classes, input_dim, hidden_dim)
            self.b = np.random.rand(1, classes, 1, hidden_dim)
            self.act = self.act_func(act)
            self.beta = np.random.rand(1, classes, hidden_dim, 1)
        else:
            self.use_linear = False
            self.w = self.b = self.act = None
            self.beta = np.random.rand(1)

        self.C = norm
        self.classes = classes
        self.input_shape = input_dim
        self.hidden_dim = hidden_dim
        assert (self.C is None) or (type(1.0/self.C) is float)

    def train(self, x, y):
        """train classic elm(specific elm for classify problem)

        Args:
            x (numpy.ndarray): (bsz, hidden_dim) train examples
            y (numpy.ndarray): (bsz, ) labels
        """
        bsz, h_0 = x.shape
        if self.use_linear:
            assert h_0 == self.input_shape
            x = np.reshape(x, (bsz, 1, 1, h_0))
            x = np.matmul(x, self.w) + self.b
            x = self.act(x)
        else:
            x = np.reshape(x, (bsz, self.classes, 1, -1))
        # change into one-hot label
        y = np.eye(self.classes)[y.astype(np.int)]
        y = y.T.reshape(-1, bsz, 1)
        if self.C:
            x = x.transpose([2, 1, 0, 3])
            x_t = x.transpose(0, 1, 3, 2)
            x_norm = np.matmul(x_t, x) + 1.0/self.C
            x_inv = np.matmul(np.linalg.pinv(x_norm), x_t)
            self.beta = np.matmul(x_inv, y)
        else:
            x = x.transpose([2, 1, 0, 3])
            x_inv = np.linalg.pinv(x)
            self.beta = np.matmul(x_inv, y)

    def test(self, x):
        """infer via elm

        Args:
            x (numpy.ndarray): (bsz, hidden_dim)

        Returns:
            y (numpy.ndarray): (bsz, classes) predict probability
        """
        bsz, h_0 = x.shape
        if self.use_linear:
            assert h_0 == self.input_shape
            x = np.reshape(x, (bsz, 1, 1, h_0))
            x = np.matmul(x, self.w) + self.b
            x = self.act(x)
        else:
            x = np.reshape(x, (bsz, self.classes, 1, -1))
        y = np.matmul(x, self.beta)
        return y

    def infer(self, x):
        """infer via elm

        Args:
            x (numpy.ndarray): (bsz, hidden_dim)

        Returns:
            y (numpy.ndarray): (bsz, ) predict labels
        """
        y = self.test(x)
        y = np.argmax(y, axis=1).squeeze()
        return y

    def act_func(self, act):
        if act == 'tanh':
            return np.tanh
        elif act == 'sigmoid':
            return lambda x: 1/(1+np.exp(-x))
        else:
            return lambda x: x


class normal_ELM:
    def __init__(self, input_shape, output_shape, act='sigmoid', norm=1) -> None:
        self.w = np.random.rand(input_shape, output_shape)
        self.b = np.random.rand(output_shape)
        self.beta = np.random.rand(output_shape, 1)
        self.act = self.act_func(act)
        self.C = norm

    def train(self, x, y):
        """train normalized elm

        Args:
            x (numpy.ndarray): (bsz, hidden_dim) train examples
            y (numpy.ndarray): (bsz, ) labels
        """
        assert x.shape[1] == self.w.shape[0]
        x = np.matmul(x, self.w) + self.b
        shape = x.shape
        x = self.act(x.flatten())
        x = np.reshape(x, shape)
        self.beta = np.matmul(np.matmul(
            np.linalg.pinv(np.matmul(x.T, x) + 1/self.C), x.T), y)
        # self.beta = np.matmul(np.linalg.pinv(x), y)

    def test(self, x):
        """infer via elm

        Args:
            x (numpy.ndarray): (bsz, hidden_dim)

        Returns:
            y (numpy.ndarray): (bsz, ) predict results
        """
        assert x.shape[1] == self.w.shape[0]
        x = np.matmul(x, self.w) + self.b
        x = self.act(x)
        y = np.matmul(x, self.beta)
        return y

    def infer(self, x):
        labels = self.test(x) > 0.5
        labels = labels.astype(np.int)
        return labels

    def act_func(self, act):
        if act == 'tanh':
            return np.tanh
        elif act == 'sigmoid':
            return lambda x: 1/(1+np.exp(-x))
        else:
            return lambda x: x


if __name__ == "__main__":
    a = np.random.rand(30, 3)
    a[:, 0] = a[:, 0] * 20 + 30
    a[:, 2] = a[:, 2] + 5
    b = rbf_kernel(a, 10)
    print(b.shape)
