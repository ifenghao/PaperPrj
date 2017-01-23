# coding:utf-8
import numpy as np
from numpy.linalg import solve
from scipy.linalg import orth
import theano
import theano.tensor as T
from theano.sandbox.neighbours import images2neibs
from lasagne.theano_extensions.padding import pad as lasagnepad

__all__ = ['compute_beta_direct', 'compute_beta_rand',
           'orthonormalize',
           'normal_random', 'uniform_random',
           'im2col', 'im2col_compfn',
           'norm2d', 'norm4d', 'norm2dglobal', 'norm4dglobal',
           'whiten2d', 'whiten2d']

splits = 2


########################################################################################################################


def compute_beta_reg(Hmat, Tmat, C):
    rows, cols = Hmat.shape
    if rows <= cols:
        beta = np.dot(Hmat.T, solve(np.eye(rows) / C + np.dot(Hmat, Hmat.T), Tmat))
    else:
        beta = solve(np.eye(cols) / C + np.dot(Hmat.T, Hmat), np.dot(Hmat.T, Tmat))
    return beta


def compute_beta_rand(Hmat, Tmat, C):
    Crand = abs(np.random.uniform(0.1, 1.1)) * C
    return compute_beta_reg(Hmat, Tmat, Crand)


def compute_beta_direct(Hmat, Tmat):
    rows, cols = Hmat.shape
    if rows <= cols:
        beta = np.dot(Hmat.T, solve(np.dot(Hmat, Hmat.T), Tmat))
    else:
        beta = solve(np.dot(Hmat.T, Hmat), np.dot(Hmat.T, Tmat))
    return beta


########################################################################################################################


def orthonormalize(filters):
    ndim = filters.ndim
    if ndim != 2:
        filters = np.expand_dims(filters, axis=0)
    rows, cols = filters.shape
    if rows >= cols:
        orthonormal = orth(filters)
    else:
        orthonormal = orth(filters.T).T
    if ndim != 2:
        orthonormal = np.squeeze(orthonormal, axis=0)
    return orthonormal


########################################################################################################################


# 随机投影矩阵不同于一般的BP网络的初始化,要保持和输入一样的单位方差
def normal_random(input_unit, hidden_unit):
    std = 1.
    return np.random.normal(loc=0, scale=std, size=(input_unit, hidden_unit)), \
           np.random.normal(loc=0, scale=std, size=hidden_unit)


def uniform_random(input_unit, hidden_unit):
    ranges = 1.
    return np.random.uniform(low=-ranges, high=ranges, size=(input_unit, hidden_unit)), \
           np.random.uniform(low=-ranges, high=ranges, size=hidden_unit)


########################################################################################################################


# 只使用一次计算出结果的时候可以使用,但是需要反复计算时速度很慢
def im2col(inputX, fsize, stride, pad, ignore_border=False):
    assert inputX.ndim == 4
    if isinstance(fsize, (int, float)): fsize = (int(fsize), int(fsize))
    if isinstance(stride, (int, float)): stride = (int(stride), int(stride))
    Xrows, Xcols = inputX.shape[-2:]
    X = T.tensor4()
    if not ignore_border:  # 保持下和右的边界
        rowpad = colpad = 0
        rowrem = (Xrows - fsize[0]) % stride[0]
        if rowrem: rowpad = stride[0] - rowrem
        colrem = (Xcols - fsize[1]) % stride[1]
        if colrem: colpad = stride[1] - colrem
        pad = ((0, rowpad), (0, colpad))
    Xpad = lasagnepad(X, pad, batch_ndim=2)
    neibs = images2neibs(Xpad, fsize, stride, 'ignore_borders')
    im2colfn = theano.function([X], neibs, allow_input_downcast=True)
    return im2colfn(inputX)


# 根据图像的行列尺寸编译im2col函数,之后直接使用函数即可,比每次都编译速度快很多
def im2col_compfn(shape, fsize, stride, pad, ignore_border=False):
    assert len(shape) == 2
    if isinstance(fsize, (int, float)): fsize = (int(fsize), int(fsize))
    if isinstance(stride, (int, float)): stride = (int(stride), int(stride))
    rows, cols = shape
    X = T.tensor4()
    if not ignore_border:  # 保持下和右的边界
        rowpad = colpad = 0
        rowrem = (rows - fsize[0]) % stride[0]
        if rowrem: rowpad = stride[0] - rowrem
        colrem = (cols - fsize[1]) % stride[1]
        if colrem: colpad = stride[1] - colrem
        pad = ((0, rowpad), (0, colpad))
    Xpad = lasagnepad(X, pad, batch_ndim=2)
    neibs = images2neibs(Xpad, fsize, stride, 'ignore_borders')
    im2colfn = theano.function([X], neibs, allow_input_downcast=True)
    return im2colfn


########################################################################################################################


# 对每一个patch里的元素去均值归一化
def norm2d(X, reg=0.1):
    size = X.shape[0]
    batchSize = size // splits
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    for start, end in zip(startRange, endRange):
        Xtmp = X[start:end]
        mean = Xtmp.mean(axis=1)
        Xtmp -= mean[:, None]
        normalizer = np.sqrt((Xtmp ** 2).mean(axis=1) + reg)
        Xtmp /= normalizer[:, None]
        X[start:end] = Xtmp
    return X


# 对每一个样本的每个通道元素去均值归一化
def norm4d(X, reg=0.1):
    raw_shape = X.shape
    X = X.reshape((X.shape[0], -1))
    X = norm2d(X, reg)
    X = X.reshape(raw_shape)
    return X


def norm2dglobal(X, mean=None, normalizer=None, reg=0.1):
    if mean is None or normalizer is None:
        mean = X.mean(axis=0)
        normalizer = 0.  # 分解求方差
        size = X.shape[0]
        batchSize = size // splits
        startRange = range(0, size - batchSize + 1, batchSize)
        endRange = range(batchSize, size + 1, batchSize)
        if size % batchSize != 0:
            startRange.append(size - size % batchSize)
            endRange.append(size)
        for start, end in zip(startRange, endRange):
            Xtmp = X[start:end]
            Xtmp -= mean[None, :]  # no copy,原始X的相应元素也被改变
            normalizer += (Xtmp ** 2).sum(axis=0) / size
        normalizer = np.sqrt(normalizer + reg)
        X /= normalizer[None, :]
        return X, mean, normalizer
    else:
        X = (X - mean[None, :]) / normalizer[None, :]
        return X


def norm4dglobal(X, mean=None, normalizer=None, reg=0.1):
    if mean is None or normalizer is None:
        mean = X.mean(axis=(0, 2, 3))
        normalizer = 0.  # 分解求方差
        size = X.shape[0]
        batchSize = size // splits
        startRange = range(0, size - batchSize + 1, batchSize)
        endRange = range(batchSize, size + 1, batchSize)
        if size % batchSize != 0:
            startRange.append(size - size % batchSize)
            endRange.append(size)
        for start, end in zip(startRange, endRange):
            Xtmp = X[start:end]
            Xtmp -= mean[None, :, None, None]  # no copy,原始X的相应元素也被改变
            normalizer += (Xtmp ** 2).sum(axis=(0, 2, 3)) / size
        normalizer = np.sqrt(normalizer + reg)
        X /= normalizer[None, :, None, None]
        return X, mean, normalizer
    else:
        X = (X - mean[None, :, None, None]) / normalizer[None, :, None, None]
        return X


def whiten2d(X, mean=None, P=None):
    if mean is None or P is None:
        mean = X.mean(axis=0)
        X -= mean
        cov = np.dot(X.T, X) / X.shape[0]
        D, V = np.linalg.eig(cov)
        reg = np.mean(D)
        P = V.dot(np.diag(np.sqrt(1 / (D + reg)))).dot(V.T)
        X = X.dot(P)
        return X, mean, P
    else:
        X -= mean
        X = X.dot(P)
        return X


def whiten4d(X, mean=None, P=None):
    raw_shape = X.shape
    X = X.reshape((X.shape[0], -1))
    if mean is None and P is None:
        mean = X.mean(axis=0)
        X -= mean
        cov = np.dot(X.T, X) / X.shape[0]
        D, V = np.linalg.eig(cov)
        reg = np.mean(D)
        P = V.dot(np.diag(np.sqrt(1 / (D + reg)))).dot(V.T)
        X = X.dot(P)
        X = X.reshape(raw_shape)
        return X, mean, P
    else:
        X -= mean
        X = X.dot(P)
        X = X.reshape(raw_shape)
        return X
