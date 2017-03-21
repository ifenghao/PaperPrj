# coding:utf-8
'''
msda使用elm的随机投影和relu激活
'''
import numpy as np
from numpy.linalg import solve
from design import *
from util import *
from util_gpu import *
from copy import copy

__all__ = ['mDALayer', 'ELMAELayer', 'mDEAELayer', 'mLDEAELayer',
           'GCNLayer', 'ZCAWhitenLayer']


class Layer(object):
    def get_train_output_for(self, inputX):
        raise NotImplementedError

    def get_test_output_for(self, inputX):
        raise NotImplementedError


class mDALayer(Layer):
    def __init__(self, C, noise):
        self.C = C
        self.noise = noise

    def _get_beta(self, X):  # 原始输入加入偏置
        n_features = X.shape[1]
        S_X = np.dot(X.T, X)
        Q = add_Q_noise(copy(S_X), self.noise)
        P = add_P_noise(copy(S_X), self.noise)
        P = P[:, :-1]  # 最后一列为偏置
        reg = np.eye(n_features) / self.C
        reg[-1, -1] = 0.
        beta = solve(reg + Q, P)
        return beta

    def get_train_output_for(self, inputX):
        n_samples, n_features = inputX.shape
        bias = np.ones((n_samples, 1), dtype=float)
        inputX = np.hstack((inputX, bias))
        self.beta = self._get_beta(inputX)
        output = np.dot(inputX, self.beta)
        output = relu(output)
        return output

    def get_test_output_for(self, inputX):
        n_samples, n_features = inputX.shape
        bias = np.ones((n_samples, 1), dtype=float)
        inputX = np.hstack((inputX, bias))
        output = np.dot(inputX, self.beta)
        output = relu(output)
        return output


class ELMAELayer(Layer):
    def __init__(self, C, n_hidden, noise_type, noise):
        self.C = C
        self.n_hidden = n_hidden
        self.noise_type = noise_type
        self.noise = noise

    def _get_beta(self, X):
        n_samples = X.shape[0]
        bias = np.ones((n_samples, 1), dtype=float)
        X = np.hstack((X, bias))  # 最后一列偏置
        n_features = X.shape[1]
        W = uniform_random_bscale(X, n_features, self.n_hidden, 10.)
        # W = normal_random_bscale(X, n_features, self.n_hidden, 10.)
        # W = sparse_random_bscale(X, n_features, self.n_hidden, 10.)
        # W = orthonormalize(W)
        H = np.dot(X, W)
        H = relu(H)
        Q = np.dot(H.T, H)
        P = np.dot(H.T, X[:, :-1])
        reg = np.eye(self.n_hidden) / self.C
        beta = solve(reg + Q, P)
        return beta.T

    def get_train_output_for(self, inputX):
        self.beta = self._get_beta(inputX)
        output = np.dot(inputX, self.beta)
        output = relu(output)
        return output

    def get_test_output_for(self, inputX):
        output = np.dot(inputX, self.beta)
        output = relu(output)
        return output


# class ELMAELayer_hidden(Layer):
#     def __init__(self, C, n_hidden, noise_type, noise):
#         self.C = C
#         self.n_hidden = n_hidden
#         self.noise_type = noise_type
#         self.noise = noise
#
#     def _get_beta(self, X):
#         # X = add_noise_decomp(X, self.noise_type, (self.noise,))
#         n_features = X.shape[1]
#         # W = uniform_random(n_features, self.n_hidden)
#         # b = uniform_random(1, self.n_hidden)
#         # W = normal_random(n_features, self.n_hidden)
#         # b = normal_random(1, self.n_hidden)
#         W = sparse_random(n_features, self.n_hidden)
#         b = uniform_random(1, self.n_hidden)
#         # W = orthonormalize(W)
#         H = np.dot(X, W)
#         hmean = np.mean(abs(H), axis=0)
#         bscale = hmean / 10.
#         H += b * bscale
#         H = relu(H)
#         H = add_noise_decomp(H, self.noise_type, (self.noise,))
#         Q = np.dot(H.T, H)
#         P = np.dot(H.T, X)
#         reg = np.eye(self.n_hidden) / self.C
#         reg[-1, -1] = 0.
#         beta = solve(reg + Q, P)
#         return beta.T
#
#     def get_train_output_for(self, inputX):
#         self.beta = self._get_beta(inputX)
#         output = np.dot(inputX, self.beta)
#         output = relu(output)
#         return output
#
#     def get_test_output_for(self, inputX):
#         output = np.dot(inputX, self.beta)
#         output = relu(output)
#         return output
#
#
# class mDALayer_hidden(Layer):
#     def __init__(self, C, n_hidden, noise_type, noise):
#         self.C = C
#         self.n_hidden = n_hidden
#         self.noise_type = noise_type
#         self.noise = noise
#
#     def _get_beta(self, X):
#         n_features = X.shape[1]
#         # W = uniform_random(n_features, self.n_hidden)
#         # b = uniform_random(1, self.n_hidden)
#         # W = normal_random(n_features, self.n_hidden)
#         # b = normal_random(1, self.n_hidden)
#         W = sparse_random(n_features, self.n_hidden)
#         b = uniform_random(1, self.n_hidden)
#         # W = orthonormalize(W)
#         H = np.dot(X, W)
#         hmean = np.mean(abs(H), axis=0)
#         bscale = hmean / 10.
#         H += b * bscale
#         H = relu(H)
#         Q = np.dot(H.T, H)
#         P = np.dot(H.T, X)
#         Q = Q_dropout(Q, self.noise)
#         P = P_dropout(P, self.noise)
#         reg = np.eye(self.n_hidden) / self.C
#         beta = solve(reg + Q, P)
#         return beta.T
#
#     def get_train_output_for(self, inputX):
#         self.beta = self._get_beta(inputX)
#         output = np.dot(inputX, self.beta)
#         output = relu(output)
#         return output
#
#     def get_test_output_for(self, inputX):
#         output = np.dot(inputX, self.beta)
#         output = relu(output)
#         return output


class mLDEAELayer(Layer):
    def __init__(self, C, n_hidden, noise):
        self.C = C
        self.n_hidden = n_hidden
        self.noise = noise

    def _get_beta(self, X):
        n_samples = X.shape[0]
        bias = np.ones((n_samples, 1), dtype=float)
        X = np.hstack((X, bias))  # 最后一列偏置
        n_features = X.shape[1]
        W = uniform_random_bscale(X, n_features, self.n_hidden, 10.)
        # W = normal_random_bscale(X, n_features, self.n_hidden, 10.)
        # W = sparse_random_bscale(X, n_features, self.n_hidden, 10.)
        S_X = np.dot(X.T, X)
        S_X_noise1 = add_Q_noise(copy(S_X), self.noise)
        Q = None
        left = np.dot(W.T, S_X_noise1)
        for i in xrange(self.n_hidden):
            right = np.dot(left, W[:, [i]])
            Q = np.concatenate((Q, right), axis=1) if Q is not None else right
        S_X_noise2 = add_P_noise(copy(S_X[:, :-1]), self.noise)  # 最后一列为偏置
        P = np.dot(W.T, S_X_noise2)
        reg = np.eye(self.n_hidden) / self.C
        reg[-1, -1] = 0.
        beta = solve(reg + Q, P)
        return beta.T

    def get_train_output_for(self, inputX):
        self.beta = self._get_beta(inputX)
        output = np.dot(inputX, self.beta)
        output = relu(output)
        return output

    def get_test_output_for(self, inputX):
        output = np.dot(inputX, self.beta)
        output = relu(output)
        return output


# class mDAELMLayer(Layer):
#     def __init__(self, C, n_hidden, p_noise):
#         self.C = C
#         self.n_hidden = n_hidden
#         self.p_noise = p_noise
#
#     def _get_beta(self, inputX):
#         n_features = inputX.shape[1]
#         S_X = np.dot(inputX.T, inputX)
#         q_noise = np.ones((n_features, 1)) * (1 - self.p_noise)
#         q_noise[-1] = 1.
#         q1 = np.dot(q_noise, q_noise.T)
#         diag_idx = np.diag_indices(n_features - 1)
#         q1[diag_idx] = 1 - self.p_noise
#         S_X_noise1 = S_X * q1
#         W = normal_random(n_features, self.n_hidden)
#         Q = None
#         left = np.dot(W.T, S_X_noise1)
#         for i in xrange(self.n_hidden):
#             right = np.dot(left, W[:, [i]])
#             Q = np.concatenate((Q, right), axis=1) if Q is not None else right
#         Q *= 0.25
#         diag_idx = np.diag_indices_from(Q)
#         Q[diag_idx] *= 2
#         q2 = np.tile(q_noise.T, (n_features, 1))
#         S_X_noise2 = S_X * q2
#         P = np.dot(W.T, S_X_noise2) * 0.5
#         beta = solve(np.eye(self.n_hidden) / self.C + Q, P)
#         return beta.T
#
#     def get_train_output_for(self, inputX):
#         n_samples, n_features = inputX.shape
#         bias = np.ones((n_samples, 1), dtype=float)
#         inputX = np.hstack((inputX, bias))
#         self.beta = self._get_beta(inputX)
#         output = np.dot(inputX, self.beta)
#         output = relu(output)
#         return output
#
#     def get_test_output_for(self, inputX):
#         n_samples, n_features = inputX.shape
#         bias = np.ones((n_samples, 1), dtype=float)
#         inputX = np.hstack((inputX, bias))
#         output = np.dot(inputX, self.beta)
#         output = relu(output)
#         return output


class mDEAELayer(Layer):
    def __init__(self, C, n_hidden, noise_type, noise):
        self.C = C
        self.n_hidden = n_hidden
        self.noise_type = noise_type
        self.noise = noise

    def _get_beta(self, X):  # 只在计算beta时增加偏置列,原始输入不使用偏置
        n_samples = X.shape[0]
        bias = np.ones((n_samples, 1), dtype=float)
        X = np.hstack((X, bias))  # 最后一列偏置
        n_features = X.shape[1]
        W = uniform_random_bscale(X, n_features, self.n_hidden, 10.)
        # W = normal_random_bscale(X, n_features, self.n_hidden, 10.)
        # W = sparse_random_bscale(X, n_features, self.n_hidden, 10.)
        if self.noise_type == 'mn':
            P_positive = relu_probs_mn_jobs(X, W, self.noise, -1)
            Q = design_Q_mn_jobs(X, W, P_positive, self.noise, 2)
            P = design_P_mn_jobs(X, W, P_positive, self.noise, 2)
            # Q = design_Q_mn_split_gpu(X, W, P_positive, self.noise, 1000)
            # P = design_P_mn_split_gpu(X, W, P_positive, self.noise, 1000)
        elif self.noise_type == 'gs':
            P_positive = relu_probs_gs_jobs(X, W, self.noise, -1)
            Q = design_Q_gs_jobs(X, W, P_positive, 2)
            P = design_P_gs_jobs(X, W, P_positive, 2)
        else:
            raise NotImplementedError
        reg = np.eye(self.n_hidden) / self.C
        reg[-1, -1] = 0.
        beta = solve(reg + Q, P)
        return beta.T

    def get_train_output_for(self, inputX):
        self.beta = self._get_beta(inputX)
        output = np.dot(inputX, self.beta)
        output = relu(output)
        return output

    def get_test_output_for(self, inputX):
        output = np.dot(inputX, self.beta)
        output = relu(output)
        return output


########################################################################################################################


class GCNLayer(Layer):
    def get_train_output_for(self, inputX, reg=0.1):
        return norm(inputX, reg)

    def get_test_output_for(self, inputX, reg=0.1):
        return norm(inputX, reg)


class ZCAWhitenLayer(Layer):
    def get_train_output_for(self, inputX):
        inputX, self.mean, self.P = whiten(inputX)
        return inputX

    def get_test_output_for(self, inputX):
        inputX, self.mean, self.P = whiten(inputX, self.mean, self.P)
        return inputX
