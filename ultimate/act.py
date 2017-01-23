# coding:utf-8
import numpy as np

__all__ = ['activate']

splits = 2


def activate(X, mode):
    act_fn = {'relu': relu, 'lrelu': leaky_relu, 'elu': elu, 'tanh': np.tanh}
    if mode not in act_fn.keys(): raise NotImplementedError
    X = act_fn[mode](X)
    return X


def relu(X):
    size = X.shape[0]
    batchSize = size / splits
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    for start, end in zip(startRange, endRange):
        xtmp = X[start:end]
        X[start:end] = 0.5 * (xtmp + abs(xtmp))
    return X


def leaky_relu(X, alpha=0.2):
    f1 = 0.5 * (1 + alpha)
    f2 = 0.5 * (1 - alpha)
    size = X.shape[0]
    batchSize = size / splits
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    for start, end in zip(startRange, endRange):
        xtmp = X[start:end]
        X[start:end] = f1 * xtmp + f2 * abs(xtmp)
    return X


def elu(X, alpha=1):
    size = X.shape[0]
    batchSize = size / splits
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    for start, end in zip(startRange, endRange):
        xtmp = X[start:end]
        X[start:end][xtmp < 0.] = alpha * (np.exp(xtmp[xtmp < 0.]) - 1)
    return X
