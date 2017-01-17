# coding:utf-8
import gc
import numpy as np
from numpy.linalg import solve
from copy import copy, deepcopy
from util import *
from act import *

__all__ = ['Layer', 'Classifier_ELMtimescv', 'Classifier_KELMcv']


########################################################################################################################


class Layer(object):
    def get_train_output_for(self, inputX):
        raise NotImplementedError

    def get_test_output_for(self, inputX):
        raise NotImplementedError


class CVInner(object):
    def get_train_acc(self, inputX, inputy):
        raise NotImplementedError

    def get_test_acc(self, inputX, inputy):
        raise NotImplementedError


class CVOuter(object):
    def train_cv(self, inputX, inputy):
        raise NotImplementedError

    def test_cv(self, inputX, inputy):
        raise NotImplementedError


def accuracy(ypred, ytrue):
    if ypred.ndim == 2:
        ypred = np.argmax(ypred, axis=1)
    if ytrue.ndim == 2:
        ytrue = np.argmax(ytrue, axis=1)
    return np.mean(ypred == ytrue)


########################################################################################################################


class Classifier_ELM(Layer):
    def __init__(self, act_mode, C, n_times):
        self.act_mode = act_mode
        self.C = C
        self.n_times = n_times

    def get_train_output_for(self, inputX, inputy=None):
        n_hidden = int(self.n_times * inputX.shape[1])
        self.W, self.b = normal_random(input_unit=inputX.shape[1], hidden_unit=n_hidden)
        self.W = orthonormalize(self.W)
        self.b = orthonormalize(self.b)
        H = np.dot(inputX, self.W) + self.b
        del inputX
        H = activate(H, self.act_mode)
        self.beta = compute_beta_rand(H, inputy, self.C)
        out = np.dot(H, self.beta)
        return out

    def get_test_output_for(self, inputX):
        H = np.dot(inputX, self.W) + self.b
        del inputX
        H = activate(H, self.act_mode)
        out = np.dot(H, self.beta)
        return out


class Classifier_ELMcv(CVInner):
    def __init__(self, act_mode, C_range, n_times):
        self.act_mode = act_mode
        self.C_range = C_range
        self.n_times = n_times

    def get_train_acc(self, inputX, inputy):
        n_hidden = int(self.n_times * inputX.shape[1])
        print 'hiddens =', n_hidden
        self.W, self.b = normal_random(input_unit=inputX.shape[1], hidden_unit=n_hidden)
        self.W = orthonormalize(self.W)
        self.b = orthonormalize(self.b)
        H = np.dot(inputX, self.W) + self.b
        del inputX
        H = activate(H, self.act_mode)
        rows, cols = H.shape
        K = np.dot(H, H.T) if rows <= cols else np.dot(H.T, H)
        self.beta_list = []
        optacc = 0.
        optC = None
        for C in self.C_range:
            Crand = abs(np.random.uniform(0.1, 1.1)) * C
            beta = np.dot(H.T, solve(np.eye(rows) / Crand + K, inputy)) if rows <= cols \
                else solve(np.eye(cols) / Crand + K, np.dot(H.T, inputy))
            out = np.dot(H, beta)
            acc = accuracy(out, inputy)
            self.beta_list.append(copy(beta))
            print '\t', C, acc
            if acc > optacc:
                optacc = acc
                optC = C
        return optC, optacc

    def get_test_acc(self, inputX, inputy):
        H = np.dot(inputX, self.W) + self.b
        del inputX
        H = activate(H, self.act_mode)
        optacc = 0.
        optC = None
        for beta, C in zip(self.beta_list, self.C_range):
            out = np.dot(H, beta)
            acc = accuracy(out, inputy)
            print '\t', C, acc
            if acc > optacc:
                optacc = acc
                optC = C
        return optC, optacc


class Classifier_ELMtimescv(CVOuter):
    def __init__(self, act_mode, n_rep, C_range, times_range):
        self.act_mode = act_mode
        self.C_range = C_range
        self.n_rep = n_rep
        self.times_range = times_range
        self.clf_list = []

    def train_cv(self, inputX, inputy):
        optacc = 0.
        optC = None
        for n_times in self.times_range:
            print 'times', n_times, ':'
            for j in xrange(self.n_rep):
                print 'repeat', j
                clf = Classifier_ELMcv(self.act_mode, self.C_range, n_times)
                C, acc = clf.get_train_acc(inputX, inputy)
                self.clf_list.append(deepcopy(clf))
                if acc > optacc:
                    optacc = acc
                    optC = C
            print 'train opt', optC, optacc

    def test_cv(self, inputX, inputy):
        optacc = 0.
        optC = None
        for clf in self.clf_list:
            print 'times', clf.n_times, ':'
            C, acc = clf.get_test_acc(inputX, inputy)
            if acc > optacc:
                optacc = acc
                optC = C
            print 'test opt', optC, optacc


########################################################################################################################


def addtrans_decomp(X, Y=None):
    if Y is None: Y = X
    size = X.shape[0]
    batchSize = size / 10
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        Xtmp = X[start:end, :] + Y[:, start:end].T
        result = np.concatenate([result, Xtmp], axis=0) if len(result) != 0 else Xtmp
    return result


def kernel(Xtr, Xte=None, kernel_type='rbf', kernel_args=(1.,)):
    rows_tr = Xtr.shape[0]
    if not isinstance(kernel_args, (tuple, list)): kernel_args = (kernel_args,)
    if kernel_type == 'rbf':
        if Xte is None:
            H = np.repeat(np.sum(Xtr ** 2, axis=1, keepdims=True), rows_tr, axis=1)
            # omega = H + H.T - 2 * np.dot(Xtr, Xtr.T)
            omega = addtrans_decomp(H) - 2 * np.dot(Xtr, Xtr.T)
            del H, Xtr
            omega = np.exp(-omega / kernel_args[0])
        else:
            rows_te = Xte.shape[0]
            Htr = np.repeat(np.sum(Xtr ** 2, axis=1, keepdims=True), rows_te, axis=1)
            Hte = np.repeat(np.sum(Xte ** 2, axis=1, keepdims=True), rows_tr, axis=1)
            # omega = Htr + Hte.T - 2 * np.dot(Xtr, Xte.T)
            omega = addtrans_decomp(Htr, Hte) - 2 * np.dot(Xtr, Xte.T)
            del Htr, Hte, Xtr, Xte
            omega = np.exp(-omega / kernel_args[0])
    elif kernel_type == 'lin':
        if Xte is None:
            omega = np.dot(Xtr, Xtr.T)
        else:
            omega = np.dot(Xtr, Xte.T)
    elif kernel_type == 'poly':
        if Xte is None:
            omega = (np.dot(Xtr, Xtr.T) + kernel_args[0]) ** kernel_args[1]
        else:
            omega = (np.dot(Xtr, Xte.T) + kernel_args[0]) ** kernel_args[1]
    elif kernel_type == 'wav':
        if Xte is None:
            H = np.repeat(np.sum(Xtr ** 2, axis=1, keepdims=True), rows_tr, axis=1)
            omega = H + H.T - 2 * np.dot(Xtr, Xtr.T)
            H1 = np.repeat(np.sum(Xtr, axis=1, keepdims=True), rows_tr, axis=1)
            omega1 = H1 - H1.T
            omega = np.cos(omega1 / kernel_args[0]) * np.exp(-omega / kernel_args[1])
        else:
            rows_te = Xte.shape[0]
            Htr = np.repeat(np.sum(Xtr ** 2, axis=1, keepdims=True), rows_te, axis=1)
            Hte = np.repeat(np.sum(Xte ** 2, axis=1, keepdims=True), rows_tr, axis=1)
            omega = Htr + Hte.T - 2 * np.dot(Xtr, Xte.T)
            Htr1 = np.repeat(np.sum(Xtr, axis=1, keepdims=True), rows_te, axis=1)
            Hte1 = np.repeat(np.sum(Xte, axis=1, keepdims=True), rows_tr, axis=1)
            omega1 = Htr1 - Hte1.T
            omega = np.cos(omega1 / kernel_args[0]) * np.exp(-omega / kernel_args[1])
    else:
        raise NotImplementedError
    return omega


class Classifier_KELM(Layer):
    def __init__(self, C, kernel_type, kernel_args):
        self.C = C
        self.kernel_type = kernel_type
        self.kernel_args = kernel_args

    def get_train_output_for(self, inputX, inputy=None):
        self.trainX = inputX
        omega = kernel(inputX, self.kernel_type, self.kernel_args)
        rows = omega.shape[0]
        Crand = abs(np.random.uniform(0.1, 1.1)) * self.C
        self.beta = solve(np.eye(rows) / Crand + omega, inputy)
        out = np.dot(omega, self.beta)
        return out

    def get_test_output_for(self, inputX):
        omega = kernel(self.trainX, inputX, self.kernel_type, self.kernel_args)
        del inputX
        out = np.dot(omega.T, self.beta)
        return out


class Classifier_KELMcv(CVOuter):
    def __init__(self, C_range, kernel_type, kernel_args_list):
        self.C_range = C_range
        self.kernel_type = kernel_type
        self.kernel_args_list = kernel_args_list

    def train_cv(self, inputX, inputy):
        self.trainX = inputX
        self.beta_list = []
        optacc = 0.
        optC = None
        optarg = None
        for kernel_args in self.kernel_args_list:
            omega = kernel(inputX, None, self.kernel_type, kernel_args)
            rows = omega.shape[0]
            for C in self.C_range:
                Crand = abs(np.random.uniform(0.1, 1.1)) * C
                beta = solve(np.eye(rows) / Crand + omega, inputy)
                out = np.dot(omega, beta)
                acc = accuracy(out, inputy)
                self.beta_list.append(copy(beta))
                print '\t', kernel_args, C, acc
                if acc > optacc:
                    optacc = acc
                    optC = C
                    optarg = kernel_args
            del omega
            gc.collect()
        print 'train opt', optarg, optC, optacc

    def test_cv(self, inputX, inputy):
        optacc = 0.
        optC = None
        optarg = None
        num = 0
        for kernel_args in self.kernel_args_list:
            omega = kernel(self.trainX, inputX, self.kernel_type, kernel_args)
            for C in self.C_range:
                out = np.dot(omega.T, self.beta_list[num])
                acc = accuracy(out, inputy)
                print '\t', kernel_args, C, acc
                num += 1
                if acc > optacc:
                    optacc = acc
                    optC = C
                    optarg = kernel_args
            del omega
            gc.collect()
        print 'test opt', optarg, optC, optacc
