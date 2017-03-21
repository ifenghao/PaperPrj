# coding:utf-8
import numpy as np
import theano
import theano.tensor as T
import time

splits = 2

__all__ = ['design_P_mn_gpu', 'design_Q_mn_gpu', 'design_P_mn_split_gpu', 'design_Q_mn_split_gpu']


def floatX(x):
    return x.astype(theano.config.floatX)


def deploy(batch, n_jobs):
    dis_batch = (batch // n_jobs) * np.ones(n_jobs, dtype=np.int)
    dis_batch[:batch % n_jobs] += 1
    starts = np.cumsum(dis_batch)
    starts = [0] + starts.tolist()
    return starts


def add_Q_noise(S_X, p_noise):
    n_feature = S_X.shape[0]
    S_X *= (1. - p_noise) ** 2
    diag_idx = T.arange(n_feature - 1)
    S_X_diag = S_X[diag_idx, diag_idx]
    S_X = T.set_subtensor(S_X_diag, S_X_diag / (1. - p_noise))
    S_X_row = S_X[-1, :]
    S_X = T.set_subtensor(S_X_row, S_X_row / (1. - p_noise))
    S_X_col = S_X[:, -1]
    S_X = T.set_subtensor(S_X_col, S_X_col / (1. - p_noise))
    return S_X


def compile_Q_mn(W, p_noise):
    X = T.matrix('X')
    P_positive = T.vector('P_positive')
    W = theano.shared(floatX(W), 'W', borrow=True)
    S_X = T.dot(X.T, X)
    S_X = add_Q_noise(S_X, p_noise)
    W_p = W * P_positive
    half_p = T.dot(W_p.T, S_X)
    Q_i = T.dot(half_p, W_p)
    Q_i_diag = T.sum(half_p * W.T, axis=1)
    diag_idx = T.arange(W.shape[1])
    Q_i = T.set_subtensor(Q_i[diag_idx, diag_idx], Q_i_diag)
    func = theano.function([X, P_positive], Q_i, allow_input_downcast=True)
    return func


def design_Q_mn_gpu(X, W, P_positive, p_noise):
    n_batch, n_feature = X.shape
    func = compile_Q_mn(W, p_noise)
    Q = 0.  # np.zeros((n_hidden, n_hidden), dtype=float)
    for i in xrange(n_batch):
        start = time.time()
        X_row = X[[i], :]
        P_row = P_positive[i, :]
        Q_i = func(X_row, P_row)
        Q += Q_i
        print i, time.time() - start
    return Q


def add_P_noise(S_X, p_noise):
    S_X *= 1. - p_noise
    S_X_row = S_X[-1, :]
    S_X = T.set_subtensor(S_X_row, S_X_row / (1. - p_noise))
    return S_X


def compile_P_mn(W, p_noise):
    X = T.matrix('X')
    P_positive = T.vector('P_positive')
    W = theano.shared(floatX(W), 'W', borrow=True)
    S_X = T.dot(X.T, X)[:, :-1]  # 最后一列为偏置
    S_X = add_P_noise(S_X, p_noise)
    W_p = W * P_positive
    P_i = T.dot(W_p.T, S_X)
    func = theano.function([X, P_positive], P_i, allow_input_downcast=True)
    return func


def design_P_mn_gpu(X, W, P_positive, p_noise):
    n_batch, n_feature = X.shape
    func = compile_P_mn(W, p_noise)
    P = 0.  # np.zeros((n_hidden, n_feature), dtype=float)
    for i in xrange(n_batch):
        start = time.time()
        X_row = X[[i], :]
        P_row = P_positive[i, :]
        P_i = func(X_row, P_row)
        P += P_i
        print i, time.time() - start
    return P


########################################################################################################################


def get_Q_i(i, X, W, P_positive, p_noise):
    S_X = T.dot(X[[i], :].T, X[[i], :])
    S_X = add_Q_noise(S_X, p_noise)
    W_p = W * P_positive[i, :]
    half_p = T.dot(W_p.T, S_X)
    Q_i = T.dot(half_p, W_p)
    Q_i_diag = T.sum(half_p * W.T, axis=1)
    diag_idx = T.arange(W.shape[1])
    Q_i = T.set_subtensor(Q_i[diag_idx, diag_idx], Q_i_diag)
    return Q_i


def compile_Q_mn_split(W, p_noise):
    X = T.matrix('X')
    P_positive = T.matrix('P_positive')
    W = theano.shared(floatX(W), 'W', borrow=True)
    idx = T.arange(X.shape[0], dtype='int32')
    results, updates = theano.scan(get_Q_i, sequences=[idx], outputs_info=None,
                                   non_sequences=[X, W, P_positive, p_noise])
    Q = T.sum(results, axis=0)
    func = theano.function([X, P_positive], Q, allow_input_downcast=True)
    return func


def design_Q_mn_split_gpu(X, W, P_positive, p_noise, n_split):
    func = compile_Q_mn_split(W, p_noise)
    starts = deploy(X.shape[0], n_split)
    Q = 0.
    for i in xrange(n_split):
        start = time.time()
        X_split = X[starts[i]:starts[i + 1], :]
        P_split = P_positive[starts[i]:starts[i + 1], :]
        Q_split = func(X_split, P_split)
        Q += Q_split
        print i, time.time() - start
    return Q


def get_P_i(i, X, W, P_positive, p_noise):
    S_X = T.dot(X[[i], :].T, X[[i], :])[:, :-1]  # 最后一列为偏置
    S_X = add_P_noise(S_X, p_noise)
    W_p = W * P_positive[i, :]
    P_i = T.dot(W_p.T, S_X)
    return P_i


def compile_P_mn_split(W, p_noise):
    X = T.matrix('X')
    P_positive = T.matrix('P_positive')
    W = theano.shared(floatX(W), 'W', borrow=True)
    idx = T.arange(X.shape[0], dtype='int32')
    results, updates = theano.scan(get_P_i, sequences=[idx], outputs_info=None,
                                   non_sequences=[X, W, P_positive, p_noise])
    P = T.sum(results, axis=0)
    func = theano.function([X, P_positive], P, allow_input_downcast=True)
    return func


def design_P_mn_split_gpu(X, W, P_positive, p_noise, n_split):
    func = compile_P_mn_split(W, p_noise)
    starts = deploy(X.shape[0], n_split)
    P = 0.  # np.zeros((n_hidden, n_feature), dtype=float)
    for i in xrange(n_split):
        start = time.time()
        X_split = X[starts[i]:starts[i + 1], :]
        P_split = P_positive[starts[i]:starts[i + 1], :]
        P_split = func(X_split, P_split)
        P += P_split
        print i, time.time() - start
    return P
