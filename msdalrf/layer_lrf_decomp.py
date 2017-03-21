# coding:utf-8
'''
msda使用elm的随机投影和relu激活
'''
import numpy as np
from numpy.linalg import solve
from msdalrf.design_lrf import *
from msdalrf.util_lrf import *
from msdalrf.visual import *
from copy import copy, deepcopy

__all__ = ['DecompLayer', 'DecompLayer_chs']


class Layer(object):
    def get_train_output_for(self, inputX):
        raise NotImplementedError

    def get_test_output_for(self, inputX):
        raise NotImplementedError


def choose_method(patches, n_hidden, noise, C, method=None):
    n_samples = patches.shape[0]
    bias = np.ones((n_samples, 1), dtype=float)
    patches = np.hstack((patches, bias))  # 最后一列偏置
    n_features = patches.shape[1]
    W = uniform_random_bscale(patches, n_features, n_hidden, 10.)
    # W = normal_random_bscale(X, n_features, self.n_hidden, 10.)
    # W = sparse_random_bscale(X, n_features, self.n_hidden, 10.)
    if method is None:
        return W[:-1, :]
    elif method == 'ELMAE':
        H = dot_decomp_dim1(patches, W, splits=10)
        H = relu(H)
        Q = dottrans_decomp(H.T, splits=(1, 10))
        P = dot_decomp_dim2(H.T, patches[:, :-1], splits=10)
    elif method == 'mLDEAE':
        S_X = dottrans_decomp(patches.T, splits=(1, 10))
        S_X_noise1 = add_Q_noise(copy(S_X), noise)
        Q = None
        left = np.dot(W.T, S_X_noise1)
        for i in xrange(n_hidden):
            right = np.dot(left, W[:, [i]])
            Q = np.concatenate((Q, right), axis=1) if Q is not None else right
        S_X_noise2 = add_P_noise(copy(S_X[:, :-1]), noise)  # 最后一列为偏置
        P = np.dot(W.T, S_X_noise2)
    elif method == 'mDEAE':
        Q, P = collaborate_mn(patches, W, noise, splits=100)
    else:
        raise NotImplementedError
    reg = np.eye(n_hidden) / C
    reg[-1, -1] = 0.
    beta = solve(reg + Q, P)
    return beta.T


class BetaLayer(object):
    def __init__(self, dir_name, C, n_hidden, fsize, pad_, stride_, noise, visual):
        self.dir_name = dir_name
        self.C = C
        self.n_hidden = n_hidden
        self.fsize = fsize
        self.stride_ = stride_
        self.pad_ = pad_
        self.noise = noise
        self.visual = visual

    def getbeta(self, inputX):
        batches, channels, rows, cols = inputX.shape
        assert channels == 1
        im2colfn_getbeta = im2col_compfn((rows, cols), self.fsize, stride=self.stride_,
                                         pad=self.pad_, ignore_border=True)
        patches = im2colfn_getbeta(inputX)
        patches = norm(patches)
        # patches, mean, P = whiten(patches)
        beta = choose_method(patches, self.n_hidden, self.noise, self.C, method=None)
        if self.visual: save_beta_lrf(beta, self.dir_name, 'betatr')
        return beta


class ForwardLayer(Layer):
    def __init__(self, dir_name, n_hidden, fsize, pad, stride,
                 beta, pool_size, mode, add_pool, visual):
        self.dir_name = dir_name
        self.n_hidden = n_hidden
        self.fsize = fsize
        self.stride = stride
        self.pad = pad
        self.beta = beta
        self.pool_size = pool_size
        self.mode = mode
        self.add_pool = add_pool
        self.visual = visual

    def get_train_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        assert channels == 1
        oshape = conv_out_shape((batches, channels, rows, cols),
                                (self.n_hidden, channels, self.fsize, self.fsize),
                                pad=self.pad, stride=self.stride, ignore_border=False)
        self.orows, self.ocols = oshape[-2:]
        self.im2colfn_forward = im2col_compfn((rows, cols), self.fsize, stride=self.stride,
                                              pad=self.pad, ignore_border=False)
        patches = self.im2colfn_forward(inputX)
        if self.visual: save_map_lrf(inputX[[10, 100, 1000]], self.dir_name, 'intr')
        del inputX
        patches = norm(patches)
        # patches, _, _ = whiten(patches, self.mean, self.P)
        patches = dot_decomp_dim1(patches, self.beta, splits=10)
        patches = patches.reshape((batches, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
        if self.visual: save_map_lrf(patches[[10, 100, 1000]], self.dir_name, 'rawtr')
        # 激活
        patches = relu(patches)
        if self.visual: save_map_lrf(patches[[10, 100, 1000]], self.dir_name, 'relutr')
        # 池化
        if self.add_pool:
            patches = pool_fn(patches, self.pool_size, mode=self.mode)
            if self.visual: save_map_lrf(patches[[10, 100, 1000]], self.dir_name, 'pooltr')
        return patches

    def get_test_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        assert channels == 1
        patches = self.im2colfn_forward(inputX)
        if self.visual: save_map_lrf(inputX[[10, 100, 1000]], self.dir_name, 'inte')
        del inputX
        patches = norm(patches)
        # patches, _, _ = whiten(patches, self.mean, self.P)
        patches = dot_decomp_dim1(patches, self.beta, splits=10)
        patches = patches.reshape((batches, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
        if self.visual: save_map_lrf(patches[[10, 100, 1000]], self.dir_name, 'rawte')
        # 激活
        patches = relu(patches)
        if self.visual: save_map_lrf(patches[[10, 100, 1000]], self.dir_name, 'relute')
        # 池化
        if self.add_pool:
            patches = pool_fn(patches, self.pool_size, mode=self.mode)
            if self.visual: save_map_lrf(patches[[10, 100, 1000]], self.dir_name, 'poolte')
        return patches


class DecompLayer(Layer):
    def __init__(self, dir_name, C,
                 n_hidden1, fsize1, pad1, stride1, pad1_, stride1_, noise1,
                 n_hidden2, fsize2, pad2, stride2, pad2_, stride2_, noise2,
                 pool_size, mode, visual):
        self.dir_name = dir_name
        self.C = C
        # first layer
        self.n_hidden1 = n_hidden1
        self.fsize1 = fsize1
        self.stride1 = stride1
        self.pad1 = pad1
        self.stride1_ = stride1_
        self.pad1_ = pad1_
        self.noise1 = noise1
        # second layer
        self.n_hidden2 = n_hidden2
        self.fsize2 = fsize2
        self.stride2 = stride2
        self.pad2 = pad2
        self.stride2_ = stride2_
        self.pad2_ = pad2_
        self.noise2 = noise2
        self.pool_size = pool_size
        self.mode = mode
        self.visual = visual

    def get_train_output_for(self, inputX):
        assert inputX.shape[1] == 1
        betalayer1 = BetaLayer(self.dir_name, self.C, self.n_hidden1, self.fsize1,
                               self.pad1_, self.stride1_, self.noise1, self.visual)
        beta1 = betalayer1.getbeta(inputX)
        self.forwardlayer1_list = []
        self.forwardlayer2_list = []
        output = None
        for i in xrange(self.n_hidden1):
            forwardlayer1 = ForwardLayer(self.dir_name, self.n_hidden1, self.fsize1, self.pad1, self.stride1,
                                         beta1[:, [i]], None, None, False, self.visual)
            self.forwardlayer1_list.append(deepcopy(forwardlayer1))
            onech_out1 = forwardlayer1.get_train_output_for(inputX)
            betalayer2 = BetaLayer(self.dir_name, self.C, self.n_hidden2, self.fsize2,
                                   self.pad2_, self.stride2_, self.noise2, self.visual)
            beta2 = betalayer2.getbeta(onech_out1)
            for j in xrange(self.n_hidden2):
                forwardlayer2 = ForwardLayer(self.dir_name, self.n_hidden2, self.fsize2, self.pad2, self.stride2,
                                             beta2[:, [j]], self.pool_size, self.mode, True, self.visual)
                self.forwardlayer2_list.append(deepcopy(forwardlayer2))
                onech_out2 = forwardlayer2.get_train_output_for(onech_out1)
                output = np.concatenate([output, onech_out2], axis=1) if output is not None else onech_out2
                print i, j,
        return output

    def get_test_output_for(self, inputX):
        assert inputX.shape[1] == 1
        forwardlayer1_iter = iter(self.forwardlayer1_list)
        forwardlayer2_iter = iter(self.forwardlayer2_list)
        output = None
        for i in xrange(self.n_hidden1):
            forwardlayer1 = forwardlayer1_iter.next()
            onech_out1 = forwardlayer1.get_test_output_for(inputX)
            for j in xrange(self.n_hidden2):
                forwardlayer2 = forwardlayer2_iter.next()
                onech_out2 = forwardlayer2.get_test_output_for(onech_out1)
                output = np.concatenate([output, onech_out2], axis=1) if output is not None else onech_out2
                print i, j,
        return output


########################################################################################################################


def im2col_catch_compiled(inputX, im2colfn):
    assert inputX.ndim == 4
    patches = []
    for ch in xrange(inputX.shape[1]):
        patches1ch = im2colfn(inputX[:, [0], :, :])
        inputX = inputX[:, 1:, :, :]
        patches = np.concatenate([patches, patches1ch], axis=1) if len(patches) != 0 else patches1ch
    return patches


class BetaLayer_chs(object):
    def __init__(self, dir_name, C, n_hidden, fsize, pad_, stride_, noise, visual):
        self.dir_name = dir_name
        self.C = C
        self.n_hidden = n_hidden
        self.fsize = fsize
        self.stride_ = stride_
        self.pad_ = pad_
        self.noise = noise
        self.visual = visual

    def getbeta(self, inputX):
        batches, channels, rows, cols = inputX.shape
        assert channels > 1
        im2colfn_getbeta = im2col_compfn((rows, cols), self.fsize, stride=self.stride_,
                                         pad=self.pad_, ignore_border=True)
        patches = im2col_catch_compiled(inputX, im2colfn_getbeta)
        patches = norm(patches)
        # patches, mean, P = whiten(patches)
        beta = choose_method(patches, self.n_hidden, self.noise, self.C, method=None)
        if self.visual: save_beta_lrfchs(beta, channels, self.dir_name, 'betatr')
        return beta


class ForwardLayer_chs(Layer):
    def __init__(self, dir_name, n_hidden, fsize, pad, stride,
                 beta, pool_size, mode, add_pool, visual):
        self.dir_name = dir_name
        self.n_hidden = n_hidden
        self.fsize = fsize
        self.stride = stride
        self.pad = pad
        self.beta = beta
        self.pool_size = pool_size
        self.mode = mode
        self.add_pool = add_pool
        self.visual = visual

    def forward_decomp(self, inputX, beta, splits=1):
        assert inputX.ndim == 4
        batchSize = int(round(float(inputX.shape[0]) / splits))
        splits = int(np.ceil(float(inputX.shape[0]) / batchSize))
        patches = None
        for _ in xrange(splits):
            patchestmp = im2col_catch_compiled(inputX[:batchSize], self.im2colfn_forward)
            inputX = inputX[batchSize:]
            # 归一化
            patchestmp = norm(patchestmp)
            # patchestmp, _, _ = whiten(patchestmp, self.mean1, self.P1)
            patchestmp = np.dot(patchestmp, beta)
            patches = np.concatenate([patches, patchestmp], axis=0) if patches is not None else patchestmp
        return patches

    def get_train_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        assert channels > 1
        oshape = conv_out_shape((batches, channels, rows, cols),
                                (self.n_hidden, channels, self.fsize, self.fsize),
                                pad=self.pad, stride=self.stride, ignore_border=False)
        self.orows, self.ocols = oshape[-2:]
        self.im2colfn_forward = im2col_compfn((rows, cols), self.fsize, stride=self.stride,
                                              pad=self.pad, ignore_border=False)
        patches = self.forward_decomp(inputX, self.beta, splits=10)
        if self.visual: save_map_lrf(inputX[[10, 100, 1000]], self.dir_name, 'intr')
        del inputX
        patches = patches.reshape((batches, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
        if self.visual: save_map_lrf(patches[[10, 100, 1000]], self.dir_name, 'rawtr')
        # 激活
        patches = relu(patches)
        if self.visual: save_map_lrf(patches[[10, 100, 1000]], self.dir_name, 'relutr')
        # 池化
        if self.add_pool:
            patches = pool_fn(patches, self.pool_size, mode=self.mode)
            if self.visual: save_map_lrf(patches[[10, 100, 1000]], self.dir_name, 'pooltr')
        return patches

    def get_test_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        assert channels > 1
        patches = self.forward_decomp(inputX, self.beta, splits=10)
        if self.visual: save_map_lrf(inputX[[10, 100, 1000]], self.dir_name, 'inte')
        del inputX
        patches = patches.reshape((batches, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
        if self.visual: save_map_lrf(patches[[10, 100, 1000]], self.dir_name, 'rawte')
        # 激活
        patches = relu(patches)
        if self.visual: save_map_lrf(patches[[10, 100, 1000]], self.dir_name, 'relute')
        # 池化
        if self.add_pool:
            patches = pool_fn(patches, self.pool_size, mode=self.mode)
            if self.visual: save_map_lrf(patches[[10, 100, 1000]], self.dir_name, 'poolte')
        return patches


class DecompLayer_chs(Layer):
    def __init__(self, dir_name, C,
                 n_hidden1, fsize1, pad1, stride1, pad1_, stride1_, noise1,
                 n_hidden2, fsize2, pad2, stride2, pad2_, stride2_, noise2,
                 pool_size, mode, visual):
        self.dir_name = dir_name
        self.C = C
        # first layer
        self.n_hidden1 = n_hidden1
        self.fsize1 = fsize1
        self.stride1 = stride1
        self.pad1 = pad1
        self.stride1_ = stride1_
        self.pad1_ = pad1_
        self.noise1 = noise1
        # second layer
        self.n_hidden2 = n_hidden2
        self.fsize2 = fsize2
        self.stride2 = stride2
        self.pad2 = pad2
        self.stride2_ = stride2_
        self.pad2_ = pad2_
        self.noise2 = noise2
        self.pool_size = pool_size
        self.mode = mode
        self.visual = visual

    def get_train_output_for(self, inputX):
        assert inputX.shape[1] > 1
        betalayer1 = BetaLayer_chs(self.dir_name, self.C, self.n_hidden1, self.fsize1,
                                   self.pad1_, self.stride1_, self.noise1, self.visual)
        beta1 = betalayer1.getbeta(inputX)
        self.forwardlayer1_list = []
        self.forwardlayer2_list = []
        output = None
        for i in xrange(self.n_hidden1):
            forwardlayer1 = ForwardLayer_chs(self.dir_name, self.n_hidden1, self.fsize1, self.pad1, self.stride1,
                                             beta1[:, [i]], None, None, False, self.visual)
            self.forwardlayer1_list.append(deepcopy(forwardlayer1))
            onech_out1 = forwardlayer1.get_train_output_for(inputX)
            betalayer2 = BetaLayer(self.dir_name, self.C, self.n_hidden2, self.fsize2,
                                   self.pad2_, self.stride2_, self.noise2, self.visual)
            beta2 = betalayer2.getbeta(onech_out1)
            for j in xrange(self.n_hidden2):
                forwardlayer2 = ForwardLayer(self.dir_name, self.n_hidden2, self.fsize2, self.pad2, self.stride2,
                                             beta2[:, [j]], self.pool_size, self.mode, True, self.visual)
                self.forwardlayer2_list.append(deepcopy(forwardlayer2))
                onech_out2 = forwardlayer2.get_train_output_for(onech_out1)
                output = np.concatenate([output, onech_out2], axis=1) if output is not None else onech_out2
                print i, j,
        return output

    def get_test_output_for(self, inputX):
        assert inputX.shape[1] > 1
        forwardlayer1_iter = iter(self.forwardlayer1_list)
        forwardlayer2_iter = iter(self.forwardlayer2_list)
        output = None
        for i in xrange(self.n_hidden1):
            forwardlayer1 = forwardlayer1_iter.next()
            onech_out1 = forwardlayer1.get_test_output_for(inputX)
            for j in xrange(self.n_hidden2):
                forwardlayer2 = forwardlayer2_iter.next()
                onech_out2 = forwardlayer2.get_test_output_for(onech_out1)
                output = np.concatenate([output, onech_out2], axis=1) if output is not None else onech_out2
                print i, j,
        return output
