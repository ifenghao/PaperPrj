# coding:utf-8
import numpy as np
from scipy.linalg import orth

splits = 5

__all__ = ['normal_random_bscale', 'uniform_random_bscale', 'sparse_random_bscale',
           'orthonormalize',
           'relu',
           'norm', 'whiten',
           'add_noise_decomp']


def deploy(batch, n_jobs):
    dis_batch = (batch // n_jobs) * np.ones(n_jobs, dtype=np.int)
    dis_batch[:batch % n_jobs] += 1
    starts = np.cumsum(dis_batch)
    starts = [0] + starts.tolist()
    return starts


def normal_random(input_unit, hidden_unit):
    std = 1.
    return np.random.normal(loc=0, scale=std, size=(input_unit, hidden_unit))


def uniform_random(input_unit, hidden_unit):
    ranges = 1.
    return np.random.uniform(low=-ranges, high=ranges, size=(input_unit, hidden_unit))


def sparse_random(input_unit, hidden_unit):
    uniform = np.random.uniform(low=0., high=1., size=(input_unit, hidden_unit))
    s = 3.
    neg = np.where(uniform < 1. / (2. * s))
    pos = np.where(uniform > (1. - 1. / (2. * s)))
    W = np.zeros_like(uniform, dtype=float)
    W[neg] = -np.sqrt(s / input_unit)
    W[pos] = np.sqrt(s / input_unit)
    return W


def mean(X, W):
    n_sample = X.shape[0]
    rand_idx = np.random.randint(0, n_sample, 10000)
    X_hidden = np.dot(X[rand_idx, :-1], W)
    hidden_mean = np.mean(abs(X_hidden), axis=0)
    return hidden_mean


def normal_random_bscale(X, input_unit, hidden_unit, bscale):
    std = 1.
    W = np.random.normal(loc=0, scale=std, size=(input_unit - 1, hidden_unit))
    b = np.random.normal(loc=0, scale=std, size=(1, hidden_unit))
    bscale = mean(X, W) / bscale
    b *= bscale
    return np.vstack((W, b))


def uniform_random_bscale(X, input_unit, hidden_unit, bscale):
    ranges = 1.
    W = np.random.uniform(low=-ranges, high=ranges, size=(input_unit - 1, hidden_unit))
    b = np.random.uniform(low=-ranges, high=ranges, size=(1, hidden_unit))
    bscale = mean(X, W) / bscale
    b *= bscale
    return np.vstack((W, b))


def sparse_random_bscale(X, input_unit, hidden_unit, bscale):
    uniform = np.random.uniform(low=0., high=1., size=(input_unit - 1, hidden_unit))
    s = 3.
    neg = np.where(uniform < 1. / (2. * s))
    pos = np.where(uniform > (1. - 1. / (2. * s)))
    W = np.zeros_like(uniform, dtype=float)
    W[neg] = -np.sqrt(s / input_unit)
    W[pos] = np.sqrt(s / input_unit)
    b = np.random.uniform(low=-1., high=1., size=(1, hidden_unit))
    bscale = mean(X, W) / bscale
    b *= bscale
    return np.vstack((W, b))


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


def relu(X):
    size = X.shape[0]
    starts = deploy(size, splits)
    for i in xrange(splits):
        xtmp = X[starts[i]:starts[i + 1]]
        X[starts[i]:starts[i + 1]] = 0.5 * (xtmp + abs(xtmp))
    return X


########################################################################################################################


def norm(X, reg=0.1):
    raw_shape = X.shape
    if len(raw_shape) > 2:
        X = X.reshape((raw_shape[0], -1))
    X = norm2d(X, reg)
    if len(raw_shape) > 2:
        X = X.reshape(raw_shape)
    return X


# 对每一个patch里的元素去均值归一化
def norm2d(X, reg=0.1):
    size = X.shape[0]
    starts = deploy(size, splits)
    for i in xrange(splits):
        Xtmp = X[starts[i]:starts[i + 1]]
        mean = Xtmp.mean(axis=1)
        Xtmp -= mean[:, None]
        normalizer = np.sqrt((Xtmp ** 2).mean(axis=1) + reg)
        Xtmp /= normalizer[:, None]
        X[starts[i]:starts[i + 1]] = Xtmp
    return X


def whiten(X, mean=None, P=None):
    raw_shape = X.shape
    if len(raw_shape) > 2:
        X = X.reshape((raw_shape[0], -1))
    tup = whiten2d(X, mean, P)
    lst = list(tup)
    if len(raw_shape) > 2:
        lst[0] = lst[0].reshape(raw_shape)
    return lst


def whiten2d(X, mean=None, P=None):
    if mean is None or P is None:
        mean = X.mean(axis=0)
        X -= mean
        cov = np.dot(X.T, X) / X.shape[0]
        D, V = np.linalg.eig(cov)
        reg = np.mean(D)
        # reg = 0.1
        P = V.dot(np.diag(np.sqrt(1 / (D + reg)))).dot(V.T)
        P = abs(P)
        X = X.dot(P)
    else:
        X -= mean
        X = X.dot(P)
    return X, mean, P


########################################################################################################################


def add_mn(X, percent=0.5):
    retain_prob = 1. - percent
    binomial = np.random.uniform(low=0., high=1., size=X.shape)
    binomial = np.asarray(binomial < retain_prob, dtype=float)
    return X * binomial


def add_gs(X, std=None, scale=10.):
    if std is None:
        Xmean = np.mean(abs(X))
        std = Xmean / scale
    normal = np.random.normal(loc=0, scale=std, size=X.shape)
    X += normal
    return X


def add_noise_decomp(X, noise_type, args):
    noise_dict = {'mn': add_mn, 'gs': add_gs}
    if noise_type not in noise_dict.keys():
        raise NotImplementedError
    noise_fn = noise_dict[noise_type]
    size = X.shape[0]
    starts = deploy(size, splits)
    for i in xrange(splits):
        X[starts[i]:starts[i + 1]] = noise_fn(X[starts[i]:starts[i + 1]], *args)
    return X
