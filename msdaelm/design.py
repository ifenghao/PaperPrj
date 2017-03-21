# coding:utf-8
import numpy as np
from scipy import stats
from sklearn.externals.joblib import Parallel, delayed, cpu_count
from util import deploy
import time

__all__ = ['relu_probs_mn', 'relu_probs_gs',
           'relu_probs_mn_jobs', 'relu_probs_gs_jobs',
           'add_P_noise', 'add_Q_noise',
           'design_P_mn', 'design_Q_mn', 'design_P_gs', 'design_Q_gs',
           'design_P_mn_jobs', 'design_Q_mn_jobs', 'design_P_gs_jobs', 'design_Q_gs_jobs']


# 慢速实现
# def design_Q_noise(n_feature, p_noise):
#     q = np.ones((n_feature, 1)) * (1 - p_noise)
#     q[-1] = 1.
#     q_noise = np.dot(q, q.T)
#     diag_idx = np.diag_indices(n_feature - 1)
#     q_noise[diag_idx] = 1 - p_noise
#     return q_noise
#
#
# def design_Q_mn_gpu(X, W, P_positive, p_noise):
#     n_feature, n_hidden = W.shape
#     Q_noise = design_Q_noise(n_feature, p_noise)
#     Q = np.zeros((n_hidden, n_hidden), dtype=float)
#     for i in xrange(n_hidden):
#         for j in xrange(i, n_hidden):
#             if i == j:
#                 P_col = P_positive[:, j]
#                 S_X = np.dot(X.T, (X.T * P_col).T)
#                 S_X *= Q_noise
#                 Q[i, j] = W[:, [i]].T.dot(S_X).dot(W[:, j])
#             else:
#                 P_row = P_positive[:, i]
#                 P_col = P_positive[:, j]
#                 S_X = np.dot(X.T * P_row, (X.T * P_col).T)
#                 S_X *= Q_noise
#                 Q[i, j] = W[:, [i]].T.dot(S_X).dot(W[:, j])
#                 Q[j, i] = Q[i, j]
#     return Q
#
#
# def design_P_noise(n_feature, p_noise):
#     q = np.ones((n_feature, 1)) * (1 - p_noise)
#     q[-1] = 1.
#     q_noise = np.tile(q, (1, n_feature))
#     return q_noise
#
#
# def design_P_mn_gpu(X, W, P_positive, p_noise):
#     n_sample, n_feature = X.shape
#     n_feature, n_hidden = W.shape
#     q_noise = design_P_noise(n_feature, p_noise)
#     P = np.zeros((n_feature, n_hidden), dtype=float)
#     for i in xrange(n_sample):
#         X_row = X[[i], :]
#         P_row = P_positive[i, :]
#         P += np.dot((W * P_row).T, X_row.T.dot(X_row) * q_noise)
#     return P


def relu_probs_mn(X, W, p_noise):  # noise表示置零的概率
    n_feature, n_hidden = W.shape
    hidden_positive_prob = None
    for i in xrange(n_hidden):
        X_hidden = X * W[:, i]
        mu = np.sum(X_hidden[:, :-1], axis=1) * (1. - p_noise)
        mu += X_hidden[:, -1]  # 偏置期望为本身
        sigma = np.sqrt(np.sum(X_hidden[:, :-1] ** 2, axis=1) * (1. - p_noise) * p_noise)  # 偏置方差为0
        col_positive_prob = 1. - stats.norm.cdf(-mu / sigma)
        hidden_positive_prob = np.concatenate([hidden_positive_prob, col_positive_prob[:, None]], axis=1) \
            if hidden_positive_prob is not None else col_positive_prob[:, None]
    return hidden_positive_prob


def relu_probs_mn_jobs(X, W, p_noise, n_jobs):
    if n_jobs < 0:
        n_jobs = max(cpu_count() + 1 + n_jobs, 1)
    starts = deploy(X.shape[0], n_jobs)
    hidden_positive_probs = Parallel(n_jobs=n_jobs)(
        delayed(relu_probs_mn)(X[starts[i]:starts[i + 1]], W, p_noise) for i in range(n_jobs))
    return np.concatenate(hidden_positive_probs, axis=0)


# 快速实现
def add_Q_noise(S_X, p_noise):
    n_feature = S_X.shape[0]
    S_X *= (1. - p_noise) ** 2
    diag_idx = np.diag_indices(n_feature - 1)
    S_X[diag_idx] /= 1. - p_noise
    S_X[-1, :] /= 1. - p_noise
    S_X[:, -1] /= 1. - p_noise
    return S_X


def design_Q_mn(X, W, P_positive, p_noise):
    n_sample, n_feature = X.shape
    n_feature, n_hidden = W.shape
    Q = 0.  # np.zeros((n_hidden, n_hidden), dtype=float)
    for i in xrange(n_sample):
        start = time.time()
        X_row = X[[i], :]
        S_X = np.dot(X_row.T, X_row)
        S_X = add_Q_noise(S_X, p_noise)
        P_row = P_positive[i, :]
        W_p = W * P_row
        half_p = np.dot(W_p.T, S_X)
        Q_i = np.dot(half_p, W_p)
        Q_i_diag = np.sum(half_p * W.T, axis=1)
        diag_idx = np.diag_indices(n_hidden)
        Q_i[diag_idx] = Q_i_diag
        Q += Q_i
        print i, time.time() - start
    return Q


def design_Q_mn_jobs(X, W, P_positive, p_noise, n_jobs):
    if n_jobs < 0:
        n_jobs = max(cpu_count() + 1 + n_jobs, 1)
    starts = deploy(X.shape[0], n_jobs)
    Q = Parallel(n_jobs=n_jobs)(
        delayed(design_Q_mn)(
            X[starts[i]:starts[i + 1]], W, P_positive[starts[i]:starts[i + 1]], p_noise)
        for i in range(n_jobs))
    return sum(Q)


def add_P_noise(S_X, p_noise):
    S_X *= 1. - p_noise
    S_X[-1, :] /= 1. - p_noise
    return S_X


def design_P_mn(X, W, P_positive, p_noise):
    n_sample, n_feature = X.shape
    P = 0.  # np.zeros((n_hidden, n_feature), dtype=float)
    for i in xrange(n_sample):
        start = time.time()
        X_row = X[[i], :]
        S_X = X_row.T.dot(X_row)[:, :-1]  # 最后一列为偏置
        S_X = add_P_noise(S_X, p_noise)
        P_row = P_positive[i, :]
        W_p = W * P_row
        P += np.dot(W_p.T, S_X)
        print i, time.time() - start
    return P


def design_P_mn_jobs(X, W, P_positive, p_noise, n_jobs):
    if n_jobs < 0:
        n_jobs = max(cpu_count() + 1 + n_jobs, 1)
    starts = deploy(X.shape[0], n_jobs)
    P = Parallel(n_jobs=n_jobs)(
        delayed(design_P_mn)(
            X[starts[i]:starts[i + 1]], W, P_positive[starts[i]:starts[i + 1]], p_noise)
        for i in range(n_jobs))
    return sum(P)


########################################################################################################################


def relu_probs_gs(X, W, s_noise):  # noise表示高斯标准差
    mu = np.dot(X, W)  # 期望
    sigma = np.sqrt(np.sum(W[:-1, :] ** 2, axis=0))  # 方差,偏置方差为0
    sigma *= s_noise
    hidden_positive_prob = 1. - stats.norm.cdf(-mu / sigma)
    return hidden_positive_prob


def relu_probs_gs_jobs(X, W, s_noise, n_jobs):
    if n_jobs < 0:
        n_jobs = max(cpu_count() + 1 + n_jobs, 1)
    starts = deploy(X.shape[0], n_jobs)
    hidden_positive_probs = Parallel(n_jobs=n_jobs)(
        delayed(relu_probs_gs)(X[starts[i]:starts[i + 1]], W, s_noise) for i in range(n_jobs))
    return np.concatenate(hidden_positive_probs, axis=0)


def design_Q_gs(X, W, P_positive):  # gs和mn相比仅仅不需要add_Q_noise
    n_sample, n_feature = X.shape
    n_feature, n_hidden = W.shape
    Q = 0.  # np.zeros((n_hidden, n_hidden), dtype=float)
    for i in xrange(n_sample):
        start = time.time()
        X_row = X[[i], :]
        S_X = np.dot(X_row.T, X_row)
        P_row = P_positive[i, :]
        W_p = W * P_row
        half_p = np.dot(W_p.T, S_X)
        Q_i = np.dot(half_p, W_p)
        Q_i_diag = np.sum(half_p * W.T, axis=1)
        diag_idx = np.diag_indices(n_hidden)
        Q_i[diag_idx] = Q_i_diag
        Q += Q_i
        print i, time.time() - start
    return Q


def design_Q_gs_jobs(X, W, P_positive, n_jobs):
    if n_jobs < 0:
        n_jobs = max(cpu_count() + 1 + n_jobs, 1)
    starts = deploy(X.shape[0], n_jobs)
    Q = Parallel(n_jobs=n_jobs)(
        delayed(design_Q_gs)(
            X[starts[i]:starts[i + 1]], W, P_positive[starts[i]:starts[i + 1]])
        for i in range(n_jobs))
    return sum(Q)


def design_P_gs(X, W, P_positive):  # gs和mn相比仅仅不需要add_P_noise
    n_sample, n_feature = X.shape
    P = 0.  # np.zeros((n_hidden, n_feature), dtype=float)
    for i in xrange(n_sample):
        start = time.time()
        X_row = X[[i], :]
        S_X = X_row.T.dot(X_row)[:, :-1]  # 最后一列为偏置
        P_row = P_positive[i, :]
        P += np.dot((W * P_row).T, S_X)
        print i, time.time() - start
    return P


def design_P_gs_jobs(X, W, P_positive, n_jobs):
    if n_jobs < 0:
        n_jobs = max(cpu_count() + 1 + n_jobs, 1)
    starts = deploy(X.shape[0], n_jobs)
    P = Parallel(n_jobs=n_jobs)(
        delayed(design_P_gs)(
            X[starts[i]:starts[i + 1]], W, P_positive[starts[i]:starts[i + 1]])
        for i in range(n_jobs))
    return sum(P)


########################################################################################################################


def Q_dropout(Q, p_dropout):
    Q *= (1. - p_dropout) ** 2
    diag_idx = np.diag_indices_from(Q)
    Q[diag_idx] /= 1. - p_dropout
    return Q


def P_dropout(P, p_dropout):
    P *= 1. - p_dropout
    return P
