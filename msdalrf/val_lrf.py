# coding:utf-8

import time
from collections import OrderedDict

import numpy as np

import utils
from msdalrf.clf import *
from msdalrf.layer_lrf_decomp import *


class mSDAELM(object):
    def __init__(self, C):
        self.C = C

    def _build(self):
        net = OrderedDict()
        # layer1
        # net['layer1'] = LRFLayer(dir_name='lrf_layer1', C=self.C, n_hidden=32, fsize=5,
        #                          pad=2, stride=1, pad_=0, stride_=1, noise=0.5,
        #                          pool_size=2, mode='max', add_pool=False, visual=True)
        # net['layer1'] = LRFLayer_chs(dir_name='lrf_layer1', C=self.C, n_hidden=32, fsize=5,
        #                              pad=2, stride=1, pad_=0, stride_=1, noise=0.25,
        #                              pool_size=None, mode=None, add_pool=False, visual=True)
        # layer2
        # net['layer2'] = LRFLayer(dir_name='lrf_layer2', C=self.C, n_hidden=32, fsize=5,
        #                          pad=2, stride=1, pad_=0, stride_=1, noise=0.5,
        #                          pool_size=7, mode='max', add_pool=True, visual=True)
        net['decomp'] = DecompLayer(dir_name='lrf_layer1', C=self.C,
                                    n_hidden1=32, fsize1=5, pad1=2, stride1=1, pad1_=0, stride1_=1, noise1=0.5,
                                    n_hidden2=32, fsize2=5, pad2=2, stride2=1, pad2_=0, stride2_=1, noise2=0.5,
                                    pool_size=7, mode='max', visual=False)
        net['decomp'] = DecompLayer_chs(dir_name='lrf_layer1', C=self.C,
                                        n_hidden1=32, fsize1=5, pad1=2, stride1=1, pad1_=0, stride1_=1, noise1=0.5,
                                        n_hidden2=32, fsize2=5, pad2=2, stride2=1, pad2_=0, stride2_=1, noise2=0.5,
                                        pool_size=8, mode='max', visual=False)
        return net

    def _get_train_output(self, net, inputX):
        out = inputX
        for name, layer in net.iteritems():
            out = layer.get_train_output_for(out)
            print 'add ' + name,
        print
        return out

    def _get_test_output(self, net, inputX):
        out = inputX
        for name, layer in net.iteritems():
            out = layer.get_test_output_for(out)
            print 'add ' + name,
        print
        return out

    def train(self, inputX, inputy):
        self.net = self._build()
        netout = self._get_train_output(self.net, inputX)
        netout = netout.reshape((netout.shape[0], -1))
        # # 训练分类器时每类只选取400个样本
        # idx = select(inputy, 400)
        # netout, inputy = netout[idx], inputy[idx]
        # self.classifier = Classifier_SVMlincv(C_range=10 ** np.arange(-3., 2., 0.5), tol=10 ** -2.5)
        # self.classifier2 = Classifier_SVMlincv(C_range=10 ** np.arange(-3., 2., 0.5), tol=10 ** -3.)
        # self.classifier3 = Classifier_SVMlincv(C_range=10 ** np.arange(-3., 2., 0.5), tol=10 ** -3.5)
        self.classifier4 = Classifier_SVMlincv(C_range=10 ** np.arange(-3., 2., 0.5), tol=10 ** -4.)
        # self.classifier.train_cv(netout, inputy)
        # self.classifier2.train_cv(netout, inputy)
        # self.classifier3.train_cv(netout, inputy)
        self.classifier4.train_cv(netout, inputy)

    def test(self, inputX, inputy):
        netout = self._get_test_output(self.net, inputX)
        netout = netout.reshape((netout.shape[0], -1))
        # self.classifier.test_cv(netout, inputy)
        # self.classifier2.test_cv(netout, inputy)
        # self.classifier3.test_cv(netout, inputy)
        self.classifier4.test_cv(netout, inputy)


def select(labels, num):
    if labels.ndim == 2:
        labels = np.argmax(labels, axis=1)
    n_class = len(np.unique(labels))
    counter = np.zeros(n_class)
    idx = []
    for i, label in enumerate(labels):
        if counter[label] < num:
            counter[label] += 1
            idx.append(i)
        elif np.all(counter >= num):
            break
    return idx


def select_rand(labels, num):
    if labels.ndim == 2:
        labels = np.argmax(labels, axis=1)
    sort_idx = np.argsort(labels)
    uni_labels, label_counts = np.unique(labels, return_counts=True)
    n_class = len(uni_labels)
    sum_counts = np.cumsum(label_counts)
    sum_counts = np.insert(sum_counts, 0, 0)
    idx = None
    for i in xrange(n_class):
        same_label = sort_idx[sum_counts[i]:sum_counts[i + 1]]
        same_label = np.random.permutation(same_label)[:num]
        idx = np.concatenate([idx, same_label]) if idx is not None else same_label
    return idx


def main():
    tr_X, te_X, tr_y, te_y = utils.load.mnist(onehot=False)
    tr_X = utils.pre.norm4d_per_sample(tr_X)
    te_X = utils.pre.norm4d_per_sample(te_X)
    # tr_X, te_X, tr_y, te_y = utils.pre.cifarWhiten('cifar10')
    # tr_y = utils.load.one_hot(tr_y, 10)
    # te_y = utils.load.one_hot(te_y, 10)
    # tr_X, te_X, tr_y, te_y = utils.load.cifar(onehot=True)
    # tr_X = utils.pre.norm4d_per_sample(tr_X, scale=55.)
    # te_X = utils.pre.norm4d_per_sample(te_X, scale=55.)
    # tr_X, te_X, tr_y, te_y, un_X = utils.load.stl10(onehot=False)
    # un_X = utils.pre.norm4d_per_sample(un_X)
    # tr_X = utils.pre.norm4d_per_sample(tr_X)
    # te_X = utils.pre.norm4d_per_sample(te_X)
    model = mSDAELM(C=1e5)
    model.train(tr_X, tr_y)
    model.test(te_X, te_y)
    print time.asctime()


if __name__ == '__main__':
    main()
