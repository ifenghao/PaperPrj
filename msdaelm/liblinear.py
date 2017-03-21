# coding:utf-8
import os
import numpy as np
from copy import deepcopy
from multiprocessing import Pool, cpu_count
import sys

sys.path.append('/home/zfh/downloadcode/liblinear-2.1/python')
import liblinearutil

save_root = os.path.join('/home', 'zhufenghao', 'liblinear_models')
if not os.path.exists(save_root):
    os.makedirs(save_root)

save_model_count = 0
load_model_count = 0

__all__ = ['Classifier_liblinearcv', 'Classifier_liblinearcv_jobs']


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


class Classifier_liblinear(CVInner):
    def __init__(self, id, C):
        self.id = id
        self.C = C

    def get_train_acc(self, inputX, inputy):
        inputX, inputy = inputX.tolist(), inputy.tolist()
        prob = liblinearutil.problem(inputy, inputX)
        param = liblinearutil.parameter('-q -s 2 -B 1.0 -c ' + str(self.C))
        self.model = liblinearutil.train(prob, param)
        _, p_acc, _ = liblinearutil.predict(inputy, inputX, self.model)
        return p_acc[0]

    def save(self):
        liblinearutil.save_model(os.path.join(save_root, str(self.id) + '.model'), self.model)
        del self.model  # ctypes objects containing pointers cannot be pickled

    def get_test_acc(self, inputX, inputy):
        inputX, inputy = inputX.tolist(), inputy.tolist()
        _, p_acc, _ = liblinearutil.predict(inputy, inputX, self.model)
        return p_acc[0]

    def load(self):
        self.model = liblinearutil.load_model(os.path.join(save_root, str(self.id) + '.model'))
        return self


class Classifier_liblinearcv(CVOuter):
    def __init__(self, C_range):
        self.C_range = C_range

    def train_cv(self, inputX, inputy):
        if inputy.ndim == 2: inputy = np.argmax(inputy, axis=1)
        optacc = 0.
        optC = None
        self.clf_list = []
        for id, C in enumerate(self.C_range):
            clf = Classifier_liblinear(id, C)
            acc = clf.get_train_acc(inputX, inputy)
            clf.save()
            self.clf_list.append(deepcopy(clf))
            print '\t', C, acc
            if acc > optacc:
                optacc = acc
                optC = C
        print 'train opt', optC, optacc

    def test_cv(self, inputX, inputy):
        if inputy.ndim == 2: inputy = np.argmax(inputy, axis=1)
        optacc = 0.
        optC = None
        for clf in self.clf_list:
            clf = clf.load()
            acc = clf.get_test_acc(inputX, inputy)
            print '\t', clf.C, acc
            if acc > optacc:
                optacc = acc
                optC = clf.C
        print 'test opt', optC, optacc


def train_lin_job(id, C, inputX, inputy):
    clf = Classifier_liblinear(id, C)
    acc = clf.get_train_acc(inputX, inputy)
    clf.save()
    return clf, acc


def test_lin_job(clf, inputX, inputy):
    clf = clf.load()
    acc = clf.get_test_acc(inputX, inputy)
    return clf, acc


class Classifier_liblinearcv_jobs(CVOuter):
    def __init__(self, C_range):
        self.C_range = C_range

    def train_cv(self, inputX, inputy):
        if inputy.ndim == 2: inputy = np.argmax(inputy, axis=1)
        pool = Pool(processes=cpu_count())
        jobs = []
        for id, C in enumerate(self.C_range):
            jobs.append(pool.apply_async(train_lin_job, (id, C, inputX, inputy)))
        pool.close()
        pool.join()
        optacc = 0.
        optC = None
        self.clf_list = []
        for one_job in jobs:
            clf, acc = one_job.get()
            self.clf_list.append(deepcopy(clf))
            print '\t', clf.C, acc
            if acc > optacc:
                optacc = acc
                optC = clf.C
        print 'train opt', optC, optacc

    def test_cv(self, inputX, inputy):
        if inputy.ndim == 2: inputy = np.argmax(inputy, axis=1)
        pool = Pool(processes=cpu_count())
        jobs = []
        for clf in self.clf_list:
            jobs.append(pool.apply_async(test_lin_job, (clf, inputX, inputy)))
        pool.close()
        pool.join()
        optacc = 0.
        optC = None
        for one_job in jobs:
            clf, acc = one_job.get()
            print '\t', clf.C, acc
            if acc > optacc:
                optacc = acc
                optC = clf.C
        print 'test opt', optC, optacc
