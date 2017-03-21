# coding:utf-8

import numpy as np
from collections import OrderedDict
import utils
from msdaelm.layer import *
from msdaelm.clf import *
import time


class mSDAELM(object):
    def __init__(self, C):
        self.C = C

    def _build(self):
        net = OrderedDict()
        # layer1
        net['gcn1'] = GCNLayer()
        net['whiten1'] = ZCAWhitenLayer()
        net['layer1'] = mDEAELayer(C=self.C, n_hidden=1000, noise_type='mn', noise=0.25)
        # layer2
        net['gcn2'] = GCNLayer()
        net['whiten1'] = ZCAWhitenLayer()
        net['layer2'] = mDEAELayer(C=self.C, n_hidden=1000, noise_type='mn', noise=0.25)
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
        self.classifier = Classifier_SVMlincv_jobs(C_range=10 ** np.arange(-3., 2., 0.5), tol=10 ** -2.5)
        self.classifier2 = Classifier_SVMlincv_jobs(C_range=10 ** np.arange(-3., 2., 0.5), tol=10 ** -3.)
        self.classifier3 = Classifier_SVMlincv_jobs(C_range=10 ** np.arange(-3., 2., 0.5), tol=10 ** -3.5)
        self.classifier4 = Classifier_SVMlincv_jobs(C_range=10 ** np.arange(-3., 2., 0.5), tol=10 ** -4.)
        self.classifier.train_cv(netout, inputy)
        self.classifier2.train_cv(netout, inputy)
        self.classifier3.train_cv(netout, inputy)
        self.classifier4.train_cv(netout, inputy)

    def test(self, inputX, inputy):
        netout = self._get_test_output(self.net, inputX)
        netout = netout.reshape((netout.shape[0], -1))
        self.classifier.test_cv(netout, inputy)
        self.classifier2.test_cv(netout, inputy)
        self.classifier3.test_cv(netout, inputy)
        self.classifier4.test_cv(netout, inputy)


def main():
    tr_X, te_X, tr_y, te_y = utils.load.mnist(onehot=False)
    tr_X = utils.pre.norm4d_per_sample(tr_X)
    te_X = utils.pre.norm4d_per_sample(te_X)
    tr_X = tr_X.reshape((tr_X.shape[0], -1))
    te_X = te_X.reshape((te_X.shape[0], -1))
    # tr_X, te_X, tr_y, te_y = utils.pre.cifarWhiten('cifar10')
    # tr_y = utils.load.one_hot(tr_y, 10)
    # te_y = utils.load.one_hot(te_y, 10)
    # tr_X, te_X, tr_y, te_y = utils.load.cifar(onehot=True)
    # tr_X = utils.pre.norm4d_per_sample(tr_X, scale=55.)
    # te_X = utils.pre.norm4d_per_sample(te_X, scale=55.)
    model = mSDAELM(C=1e5)
    model.train(tr_X, tr_y)
    model.test(te_X, te_y)
    print time.asctime()


if __name__ == '__main__':
    main()
