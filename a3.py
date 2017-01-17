# coding:utf-8
'''
基于ELM的局部感知自编码器
仿照多通道的卷积层,将上一层所有通道卷积后的特征图求和,得到下一层的一张特征图
大通道
block mask
先激活再池化
计算beta的两个方法都可以使用,只是计算复杂度不同
在多个通道的同一位置统一加噪
add_mn_block和add_mn_block_mch的percent一般要设置的较高,因为有边界舍弃和重叠
add_mn_array可以设置较低,因为都是有效遮挡只有重叠,没有边界舍弃
'''

import numpy as np
from collections import OrderedDict
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
        # layer1
        # net['layer1'] = ELMAELayer(C=self.C, n_hidden=32, filter_size=6, act_mode='relu',
        #                            pad=0, stride=1, pad_=0, stride_=1, noise_type='mn_array',
        #                            noise_args={'percent': 0.3, 'block_list': ((15, 15),),
        #                                        'mode': 'channel', 'p_add': 0.8,
        #                                        'map_mode': 'alter', 'p_center_of_image': 0.5,
        #                                        'p_center_of_block': 0.75},
        #                            pool_type='fp', pool_size=2., mode='max', pool_args={'overlap': True,},
        #                            add_cccp=False, cccp_out=32, cccp_noise_type='mn_array',
        #                            cccp_noise_args={'percent': 0.3, 'block_list': ((15, 15),),
        #                                             'mode': 'channel', 'p_add': 0.8,
        #                                             'map_mode': 'alter', 'p_center_of_image': 0.5,
        #                                             'p_center_of_block': 0.75})
        # net['layer1'] = ELMAELayer(C=self.C, name=dir_name, n_hidden=32, filter_size=6, act_mode='relu',
        #                            pad=0, stride=1, pad_=0, stride_=1, noise_type='mn_array_orient',
        #                            noise_args={'pad': 0, 'stride': 1, 'percent': 0.3,
        #                                        'block_list': ((15, 15),), 'p_add': 0.8,
        #                                        'p_center_of_image': (0.8, 0.5), 'p_center_of_block': 0.5},
        #                            pool_type='fp', pool_size=2., mode='max', pool_args={'overlap': True,},
        #                            add_cccp=False, cccp_out=32, cccp_noise_type='mn_array',
        #                            cccp_noise_args={'pad': 0, 'stride': 1, 'percent': 0.3,
        #                                             'block_list': ((15, 15),), 'p_add': 0.8,
        #                                             'p_center_of_image': (0.8, 0.5), 'p_center_of_block': 0.5})
        net['layer1'] = ELMAELayer(C=self.C, name=dir_name, n_hidden=32, filter_size=6, act_mode='relu',
                                   pad=0, stride=1, pad_=0, stride_=1, noise_type='mn_array_edge',
                                   noise_args={'pad': 0, 'stride': 1, 'percent': 0.3,
                                               'block_list': ((10, 10),), 'p_add': 0.8},
                                   pool_type='fp', pool_size=1.5, mode='max', pool_args={'overlap': True,},
                                   add_cccp=False, cccp_out=32, cccp_noise_type='mn_array_edge',
                                   cccp_noise_args={'pad': 0, 'stride': 1, 'percent': 0.3,
                                                    'block_list': ((10, 10),), 'p_add': 0.8})
        # net['gcn11'] = GCNLayer()
        # net['cccp1'] = CCCPLayer(C=self.C, n_out=32, noise_type='mn_array',
        #                          noise_args={'percent': 0.5, 'block_list': ((6, 6),),
        #                                      'mode': 'channel', 'p_add': 0.8})
        # net['gcn12'] = GCNLayer()
        # layer2
        # net['layer2'] = ELMAELayer(C=self.C, name=dir_name, n_hidden=32, filter_size=6, act_mode='relu',
        #                            pad=0, stride=1, pad_=0, stride_=1, noise_type='mn_array',
        #                            noise_args={'percent': 0.3, 'block_list': ((4, 4),),
        #                                        'mode': 'channel', 'p_add': 0.8,
        #                                        'map_mode': 'uniform', 'p_center_of_image': None,
        #                                        'p_center_of_block': None},
        #                            pool_type='fp', pool_size=2.4, mode='max', pool_args={'overlap': True,},
        #                            add_cccp=False, cccp_out=32, cccp_noise_type='mn_array',
        #                            cccp_noise_args={'percent': 0.3, 'block_list': ((4, 4),),
        #                                             'mode': 'channel', 'p_add': 0.8,
        #                                             'map_mode': 'uniform', 'p_center_of_image': None,
        #                                             'p_center_of_block': None})
        # net['layer2'] = ELMAELayer(C=self.C, name=dir_name, n_hidden=32, filter_size=6, act_mode='relu',
        #                            pad=0, stride=1, pad_=0, stride_=1, noise_type='mn_array_orient',
        #                            noise_args={'pad': 0, 'stride': 1, 'percent': 0.3,
        #                                        'block_list': ((4, 4),), 'p_add': 0.8,
        #                                        'p_center_of_image': (0.8, 0.5), 'p_center_of_block': 0.5},
        #                            pool_type='fp', pool_size=2.4, mode='max', pool_args={'overlap': True,},
        #                            add_cccp=False, cccp_out=32, cccp_noise_type='mn_array_orient',
        #                            cccp_noise_args={'pad': 0, 'stride': 1, 'percent': 0.3,
        #                                             'block_list': ((4, 4),), 'p_add': 0.8,
        #                                             'p_center_of_image': (0.8, 0.5), 'p_center_of_block': 0.5})
        net['layer2'] = ELMAELayer(C=self.C, name=dir_name, n_hidden=32, filter_size=6, act_mode='relu',
                                   pad=0, stride=1, pad_=0, stride_=1, noise_type='mn_array_edge',
                                   noise_args={'pad': 0, 'stride': 1, 'percent': 0.3,
                                               'block_list': ((7, 7),), 'p_add': 0.8},
                                   pool_type='fp', pool_size=3.7, mode='max', pool_args={'overlap': True,},
                                   add_cccp=False, cccp_out=32, cccp_noise_type='mn_array_edge',
                                   cccp_noise_args={'pad': 0, 'stride': 1, 'percent': 0.3,
                                                    'block_list': ((7, 7),), 'p_add': 0.8})
        # net['layer2'] = ELMAECrossPartLayer(C=self.C, n_hidden=32, filter_size=6, pad=0, stride=1, act_mode='relu',
        #                                     pad_=0, stride_=1, cross_size=4, noise_type='mn_array',
        #                                     noise_args={'percent': 0.3, 'block_list': ((4, 4),),
        #                                                 'mode': 'channel', 'p_add': 0.8,
        #                                                 'map_mode': 'uniform', 'p_center_of_image': None,
        #                                                 'p_center_of_block': None},
        #                                     pool_type='fp', pool_size=2.4, mode='max', pool_args={'overlap': True,},
        #                                     add_cccp=False, cccp_out=32, cccp_noise_type='mn_array',
        #                                     cccp_noise_args={'percent': 0.3, 'block_list': ((4, 4),),
        #                                                      'mode': 'channel', 'p_add': 0.8,
        #                                                      'map_mode': 'uniform', 'p_center_of_image': None,
        #                                                      'p_center_of_block': None})
        # net['gcn21'] = GCNLayer()
        # net['cccp2'] = CCCPLayer(C=self.C, n_out=1024, noise_type='mn_array', act_mode='relu',
        #                          noise_args={'percent': 0.3, 'block_list': ((2, 2),),
        #                                      'mode': 'channel', 'p_add': 0.8,
        #                                      'map_mode': 'uniform', 'p_center_of_image': None,
        #                                      'p_center_of_block': None})
        # net['gcn22'] = GCNLayer()
        # net['pool2']=PoolLayer(pool_type='fp', pool_size=1.5, mode='avg', pool_args={'overlap': True,})
        # layer3
        # net['layer3'] = ELMAELayer(C=self.C, n_hidden=32, filter_size=4, pad=0, stride=1,
        #                            pad_=0, stride_=1, noise_type='mn_array',
        #                            noise_args={'percent': 0.3, 'block_list': ((5, 5),),
        #                                        'mode': 'channel', 'p_add': 0.8},
        #                            pool_type='fp', pool_size=2.4, mode='max', pool_args={'overlap': True,},
        #                            add_cccp=False, cccp_out=32, cccp_noise_type='mn_array',
        #                            cccp_noise_args={'percent': 0.3, 'block_list': ((5, 5),),
        #                                             'mode': 'channel', 'p_add': 0.8})
        # net['layer3'] = ELMAECrossPartLayer(C=self.C, n_hidden=32, filter_size=6, pad=0, stride=1,
        #                                     pad_=0, stride_=1, cross_size=4, noise_type='mn_array',
        #                                     noise_args={'percent': 0.3, 'block_list': ((5, 5),),
        #                                                 'mode': 'channel', 'p_add': 0.8},
        #                                     pool_type='fp', pool_size=2.4, mode='max', pool_args={'overlap': True,},
        #                                     add_cccp=False, cccp_out=32, cccp_noise_type='mn_array',
        #                                     cccp_noise_args={'percent': 0.3, 'block_list': ((5, 5),),
        #                                                      'mode': 'channel', 'p_add': 0.8})
        # net['gcn21'] = GCNLayer()
        # net['cccp2'] = CCCPLayer(C=self.C, n_out=289, noise_level=0.2)
        # net['gcn22'] = GCNLayer()
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
        print
        return out

    def train(self, inputX, inputy):
        self.net = self._build()
        netout = self._get_train_output(self.net, inputX)
        netout = netout.reshape((netout.shape[0], -1))
        # self.classifier = Classifier_ELMtimescv(n_rep=3, C_range=10 ** np.arange(-1., 3., 1.), times_range=[12, ])
        self.classifier = Classifier_KELMcv(C_range=10 ** np.arange(3.5, 7., 1.5), kernel_type='rbf',
                                            kernel_args_list=10 ** np.arange(3., 7., 1.25))
        return self.classifier.train_cv(netout, inputy)

    def test(self, inputX, inputy):
        netout = self._get_test_output(self.net, inputX)
        netout = netout.reshape((netout.shape[0], -1))
        return self.classifier.test_cv(netout, inputy)


def main():
    tr_X, te_X, tr_y, te_y = utils.load.mnist(onehot=True)
    tr_X = utils.pre.norm4d_per_sample(tr_X)
    te_X = utils.pre.norm4d_per_sample(te_X)
    # tr_X, te_X, tr_y, te_y = utils.pre.cifarWhiten('cifar10')
    # tr_y = utils.load.one_hot(tr_y, 10)
    # te_y = utils.load.one_hot(te_y, 10)
    # tr_X, te_X, tr_y, te_y = utils.load.cifar(onehot=True)
    # tr_X = utils.pre.norm4d_per_sample(tr_X, scale=55.)
    # te_X = utils.pre.norm4d_per_sample(te_X, scale=55.)
    model = LRFELMAE(C=None)
    model.train(tr_X, tr_y)
    model.test(te_X, te_y)


if __name__ == '__main__':
    main()