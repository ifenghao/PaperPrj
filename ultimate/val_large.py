# coding:utf-8
'''
基于ELM的局部感知自编码器
仿照多通道的卷积层,将上一层所有通道卷积后的特征图求和,得到下一层的一张特征图
cifar数据集
第一层使用crossall层
------
crossall采用分解计算防止内存溢出
在分解取inputX时,将已经取出的部分删除减少内存使用
增加block mn方法加噪
'''

import numpy as np
from collections import OrderedDict
from copy import copy
import utils
from ultimate.clf import *
from ultimate.layer import *

dir_name = 'val'


class LRFELMAE(object):
    def __init__(self, C):
        self.C = C

    def _build(self):
        net = OrderedDict()
        # net['gcn0'] = GCNLayer()
        ###################################### layer1 ######################################
        seqnet1 = OrderedDict()
        ############################################################
        array_args1 = {'percent': 0.25, 'block_list': None, 'mode': 'channel', 'p_add': 1.,
                       'map_mode': 'uniform', 'p_center_of_image': None, 'p_center_of_block': None,
                       'apply_mode': None}
        ae_args1 = copy(array_args1)
        ae_args1['apply_mode'] = 'omap'
        ae_args1['block_list'] = ((14, 14),)
        cccp_args1 = copy(array_args1)
        cccp_args1['apply_mode'] = 'cccp'
        cccp_args1['block_list'] = ((16, 16),)
        ######################################
        orient_args1 = {'pad': 0, 'stride': 1, 'percent': 0.25, 'block_list': None,
                        'p_add': 1., 'p_center_of_image': (0.6, 0.6),
                        'apply_mode': None}
        ae_args1 = copy(orient_args1)
        ae_args1['apply_mode'] = 'omap'
        ae_args1['block_list'] = ((14, 14),)
        cccp_args1 = copy(orient_args1)
        cccp_args1['apply_mode'] = 'cccp'
        cccp_args1['block_list'] = ((16, 16),)
        ######################################
        edge_args1 = {'pad': 0, 'stride': 1, 'percent': 0.25, 'block_list': None, 'edge_or_binary': None,
                      'p_add': 1., 'edge_args': {'border': 0.1, 'sigma': 0.9, 'lth': 0.5, 'hth': 0.8},
                      'apply_mode': None}
        ae_args1 = copy(edge_args1)
        ae_args1['apply_mode'] = 'omap'
        ae_args1['block_list'] = ((14, 14),)
        ae_args1['edge_or_binary'] = True
        cccp_args1 = copy(edge_args1)
        cccp_args1['apply_mode'] = 'cccp'
        cccp_args1['block_list'] = ((16, 16),)
        cccp_args1['edge_or_binary'] = True
        ############################################################
        layer1args = {'C': self.C, 'name': dir_name, 'n_hidden': 125, 'fsize': 5, 'act_mode': 'relu',
                      'pad': 2, 'stride': 1, 'pad_': 0, 'stride_': 1, 'beta_os': True, 'splits': 20,
                      'noise_type': 'mn', 'noise_args': {'percent': 0.25},
                      'add_pool': False, 'pool_type': 'pool', 'pool_size': 8, 'mode': 'max', 'pool_args': None,
                      'add_cccp': False, 'cccp_out': 32, 'cccp_noise_type': 'mn',
                      'cccp_noise_args': {'percent': 0.25}}
        seqnet1['layer1'] = [ELMAECrossAllLayer, layer1args]
        # seqnet1['gcn11'] = [GCNLayer, {}]
        ############################################################
        cccp_array_args1 = {'percent': 0.25, 'block_list': ((16, 16),), 'mode': 'channel', 'p_add': 1.,
                            'map_mode': 'uniform', 'p_center_of_image': None, 'p_center_of_block': None,
                            'apply_mode': 'cccp'}
        cccp_orient_args1 = {'pad': 0, 'stride': 1, 'percent': 0.25, 'block_list': ((16, 16),),
                             'p_add': 1., 'p_center_of_image': (0.8, 0.6),
                             'apply_mode': 'cccp'}
        cccp_edge_args1 = {'pad': 0, 'stride': 1, 'percent': 0.25, 'block_list': ((16, 16),),
                           'p_add': 1., 'edge_args': {'border': 0.1, 'sigma': 0.9, 'lth': 0.5, 'hth': 0.8},
                           'edge_or_binary': True, 'apply_mode': 'cccp'}
        ############################################################
        cccp1args = {'C': self.C, 'name': dir_name, 'n_out': 32, 'beta_os': True, 'splits': 20,
                      'act_mode': 'relu', 'noise_type': 'mn', 'noise_args': {'percent': 0.25}}
        # seqnet1['cccp1'] = [CCCPLayer, cccp1args]
        # seqnet1['gcn12'] = [GCNLayer, {}]
        ###################################### layer2 ######################################
        ############################################################
        array_args2 = {'percent': 0.25, 'block_list': None, 'mode': 'channel', 'p_add': 1.,
                       'map_mode': 'uniform', 'p_center_of_image': None, 'p_center_of_block': None,
                       'apply_mode': None}
        ae_args2 = copy(array_args2)
        ae_args2['apply_mode'] = 'omap'
        ae_args2['block_list'] = ((14, 14),)
        cccp_args2 = copy(array_args2)
        cccp_args2['apply_mode'] = 'cccp'
        cccp_args2['block_list'] = ((16, 16),)
        #####################################
        orient_args2 = {'pad': 0, 'stride': 1, 'percent': 0.25, 'block_list': None,
                        'p_add': 1., 'p_center_of_image': (0.8, 0.6),
                        'apply_mode': None}
        ae_args2 = copy(orient_args2)
        ae_args2['apply_mode'] = 'omap'
        ae_args2['block_list'] = ((14, 14),)
        cccp_args2 = copy(orient_args2)
        cccp_args2['apply_mode'] = 'cccp'
        cccp_args2['block_list'] = ((16, 16),)
        #####################################
        edge_args2 = {'pad': 0, 'stride': 1, 'percent': 0.25, 'block_list': None, 'edge_or_binary': None,
                      'p_add': 1., 'edge_args': {'border': 0.1, 'sigma': 0.9, 'lth': 0.5, 'hth': 0.8},
                      'apply_mode': None}
        ae_args2 = copy(edge_args2)
        ae_args2['apply_mode'] = 'omap'
        ae_args2['block_list'] = ((14, 14),)
        ae_args2['edge_or_binary'] = True
        cccp_args2 = copy(edge_args2)
        cccp_args2['apply_mode'] = 'cccp'
        cccp_args2['block_list'] = ((16, 16),)
        cccp_args2['edge_or_binary'] = True
        ############################################################
        layerargs2 = {'C': self.C, 'name': dir_name, 'n_hidden': 125, 'fsize': 5, 'act_mode': 'relu',
                      'pad': 2, 'stride': 1, 'pad_': 0, 'stride_': 1, 'beta_os': True, 'splits': 20,
                      'noise_type': 'mn', 'noise_args': {'percent': 0.25},
                      'add_pool': True, 'pool_type': 'fp', 'pool_size': 8.9, 'mode': 'max', 'pool_args': None,
                      'add_cccp': False, 'ccsize': 10, 'cccp_out': 32, 'cccp_noise_type': 'mn',
                      'cccp_noise_args': {'percent': 0.25}}
        seqnet1['layer2'] = [ELMAELayer, layerargs2]
        net['vertical'] = VerticalLayer(seqnet1, hsize=3)
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
        # 训练分类器时每类只选取400个样本
        idx = select(inputy, 400)
        netout, inputy = netout[idx], inputy[idx]
        # self.classifier = Classifier_ELMtimescv(n_rep=3, C_range=10 ** np.arange(-1., 3., 1.), times_range=[12, ])
        self.classifier = Classifier_KELMcv(C_range=10 ** np.arange(2., 4.5, 0.5), kernel_type='rbf',
                                            kernel_args_list=10 ** np.arange(4., 6.5, 0.5))
        return self.classifier.train_cv(netout, inputy)

    def test(self, inputX, inputy):
        netout = self._get_test_output(self.net, inputX)
        netout = netout.reshape((netout.shape[0], -1))
        return self.classifier.test_cv(netout, inputy)


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


def main():
    # tr_X, te_X, tr_y, te_y = utils.load.mnist(onehot=True)
    # tr_X = utils.pre.norm4d_per_sample(tr_X)
    # te_X = utils.pre.norm4d_per_sample(te_X)
    # tr_X, te_X, tr_y, te_y = utils.pre.cifarWhiten('cifar10')
    # tr_y = utils.load.one_hot(tr_y, 10)
    # te_y = utils.load.one_hot(te_y, 10)
    tr_X, te_X, tr_y, te_y = utils.load.cifar(onehot=True)
    tr_X = utils.pre.norm4d_per_sample(tr_X, scale=55., cross_ch=False)
    te_X = utils.pre.norm4d_per_sample(te_X, scale=55., cross_ch=False)
    # tr_X, tr_y = tr_X[:30000], tr_y[:30000]
    model = LRFELMAE(C=None)
    model.train(tr_X, tr_y)
    model.test(te_X, te_y)


if __name__ == '__main__':
    main()
