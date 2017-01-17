# coding:utf-8
import numpy as np
from collections import OrderedDict
from copy import copy, deepcopy
import gc
import utils
from util import *
from noise import *
from pool import *
from act import *

__all__ = ['ELMAELayer', 'ELMAECrossAllLayer', 'ELMAECrossPartLayer',
           'ELMAEMBetaLayer',
           'PoolLayer', 'SPPLayer', 'BNLayer', 'GCNLayer', 'MergeLayer',
           'CCCPLayer', 'CCCPMBetaLayer']

dir_name = 'val'


class Layer(object):
    def get_train_output_for(self, inputX):
        raise NotImplementedError

    def get_test_output_for(self, inputX):
        raise NotImplementedError


class ELMAELayer(Layer):
    def __init__(self, C, name, n_hidden, act_mode, filter_size, pad, stride, pad_, stride_,
                 noise_type, noise_args, pool_type, pool_size, mode, pool_args,
                 cccp_out, cccp_noise_type, cccp_noise_args, add_cccp):
        global dir_name
        dir_name = name
        self.C = C
        self.n_hidden = n_hidden
        self.act_mode = act_mode
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.stride_ = stride_
        self.pad_ = pad_
        self.noise_type = noise_type  # 不允许为mch
        self.noise_args = noise_args
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.mode = mode
        self.pool_args = pool_args
        self.cccp_out = cccp_out
        self.cccp_noise_type = cccp_noise_type
        self.cccp_noise_args = cccp_noise_args
        self.add_cccp = add_cccp

    def _get_beta(self, oneChannel, bias_scale=10):
        assert oneChannel.ndim == 4 and oneChannel.shape[1] == 1  # ELMAE的输入通道数必须为1,即只有一张特征图
        batches = oneChannel.shape[0]
        patch_size = self.filter_size ** 2
        # 生成随机正交滤波器
        W, b = normal_random(input_unit=self.filter_size ** 2, hidden_unit=self.n_hidden)
        W = orthonormalize(W)  # 正交后的幅度在-1~+1之间
        # 卷积前向输出,和取patch时一致
        patches = self.im2colfn_getbeta(oneChannel)
        ##########################
        patches = norm2d(patches)
        # patches, self.mean1, self.P1 = whiten2d(patches)
        ##########################
        # 在patches上加噪
        if self.noise_type == 'mn_array_edge':  # 需要原始图像
            self.noise_args['originX'] = oneChannel
        noise_patches = np.copy(patches)
        if self.noise_type in ('mn_block', 'mn_array', 'mn_array_orient', 'mn_array_edge'):  # 4d
            noise_patches = noise_patches.reshape((batches, self.orows_, self.ocols_, patch_size))
            noise_patches = noise_patches.transpose((0, 3, 1, 2))
        noise_patches = add_noise_decomp(noise_patches, self.noise_type, self.noise_args)
        if self.noise_type in ('mn_block', 'mn_array', 'mn_array_orient', 'mn_array_edge'):
            noise_patches = noise_patches.transpose((0, 2, 3, 1))
            noise_patches = noise_patches.reshape((-1, patch_size))
        hiddens = np.dot(noise_patches, W)
        del noise_patches
        hmax, hmin = np.max(hiddens, axis=0), np.min(hiddens, axis=0)
        scale = (hmax - hmin) / (2 * bias_scale)
        hiddens += b * scale
        hiddens = activate(hiddens, self.act_mode)
        # 计算beta
        beta = compute_beta_direct(hiddens, patches)
        beta = beta.T
        return beta

    def get_train_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        oshape_ = utils.basic.conv_out_shape((batches, channels, rows, cols),
                                             (self.n_hidden, channels, self.filter_size, self.filter_size),
                                             pad=self.pad_, stride=self.stride_, ignore_border=False)
        self.orows_, self.ocols_ = oshape_[-2:]
        oshape = utils.basic.conv_out_shape((batches, channels, rows, cols),
                                            (self.n_hidden, channels, self.filter_size, self.filter_size),
                                            pad=self.pad, stride=self.stride, ignore_border=False)
        self.orows, self.ocols = oshape[-2:]
        self.im2colfn_getbeta = im2col_compfn((rows, cols), self.filter_size, stride=self.stride_,
                                              pad=self.pad_, ignore_border=False)
        self.im2colfn_forward = im2col_compfn((rows, cols), self.filter_size, stride=self.stride,
                                              pad=self.pad, ignore_border=False)
        self.filters = []
        self.cccps = []
        output = []
        for ch in xrange(channels):
            oneChannel = inputX[:, [0], :, :]
            inputX = inputX[:, 1:, :, :]
            utils.visual.save_map(oneChannel[[10, 100, 1000]], dir_name, 'elmin')
            beta = self._get_beta(oneChannel)
            utils.visual.save_beta(beta, dir_name, 'beta')
            patches = self.im2colfn_forward(oneChannel)
            del oneChannel
            ##########################
            patches = norm2d(patches)
            # patches = whiten2d(patches, self.mean1, self.P1)
            ##########################
            patches = np.dot(patches, beta)
            patches = patches.reshape((batches, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
            utils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmraw')
            # 归一化
            # patches = norm4d(patches)
            # utils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmnorm')
            # 激活
            patches = activate(patches, self.act_mode)
            utils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmrelu')
            # 添加cccp层组合
            cccp = CCCPLayer(C=self.C, name=dir_name, n_out=self.cccp_out, act_mode=self.act_mode,
                             noise_type=self.cccp_noise_type, noise_args=self.cccp_noise_args)
            if self.add_cccp: patches = cccp.get_train_output_for(patches)
            # 池化
            patches = pool_op(patches, self.pool_size, self.pool_type, self.mode, self.pool_args)
            utils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmpool')
            # 组合最终结果
            output = np.concatenate([output, patches], axis=1) if len(output) != 0 else patches
            self.filters.append(copy(beta))
            self.cccps.append(deepcopy(cccp))
            print ch,
            gc.collect()
        return output

    def get_test_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        output = []
        for ch in xrange(channels):
            oneChannel = inputX[:, [0], :, :]
            inputX = inputX[:, 1:, :, :]
            utils.visual.save_map(oneChannel[[10, 100, 1000]], dir_name, 'elminte')
            patches = self.im2colfn_forward(oneChannel)
            del oneChannel
            ##########################
            patches = norm2d(patches)
            # patches = whiten2d(patches, self.mean1, self.P1)
            ##########################
            patches = np.dot(patches, self.filters[ch])
            patches = patches.reshape((batches, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
            utils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmrawte')
            # 归一化
            # patches = norm4d(patches)
            # utils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmnormte')
            # 激活
            patches = activate(patches, self.act_mode)
            utils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmrelute')
            # 添加cccp层组合
            if self.add_cccp: patches = self.cccps[ch].get_test_output_for(patches)
            # 池化
            patches = pool_op(patches, self.pool_size, self.pool_type, self.mode, self.pool_args)
            utils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmpoolte')
            # 组合最终结果
            output = np.concatenate([output, patches], axis=1) if len(output) != 0 else patches
            print ch,
            gc.collect()
        return output


def im2col_catch(inputX, fsize, stride, pad, ignore_border=False):
    assert inputX.ndim == 4
    patches = []
    for ch in xrange(inputX.shape[1]):
        patches1ch = im2col(inputX[:, [0], :, :], fsize, stride, pad, ignore_border)
        inputX = inputX[:, 1:, :, :]
        patches = np.concatenate([patches, patches1ch], axis=1) if len(patches) != 0 else patches1ch
    return patches


def im2col_catch_compiled(inputX, im2colfn):
    assert inputX.ndim == 4
    patches = []
    for ch in xrange(inputX.shape[1]):
        patches1ch = im2colfn(inputX[:, [0], :, :])
        inputX = inputX[:, 1:, :, :]
        patches = np.concatenate([patches, patches1ch], axis=1) if len(patches) != 0 else patches1ch
    return patches


class ELMAECrossAllLayer(Layer):
    def __init__(self, C, name, n_hidden, act_mode, filter_size, pad, stride, pad_, stride_,
                 noise_type, noise_args, pool_type, pool_size, mode, pool_args,
                 cccp_out, cccp_noise_type, cccp_noise_args, add_cccp):
        global dir_name
        dir_name = name
        self.C = C
        self.n_hidden = n_hidden
        self.act_mode = act_mode
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.stride_ = stride_
        self.pad_ = pad_
        self.noise_type = noise_type
        self.noise_args = noise_args
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.mode = mode
        self.pool_args = pool_args
        self.cccp_out = cccp_out
        self.cccp_noise_type = cccp_noise_type
        self.cccp_noise_args = cccp_noise_args
        self.add_cccp = add_cccp

    def _get_beta(self, inputX, bias_scale=10):
        assert inputX.ndim == 4
        batches, channels = inputX.shape[:2]
        patch_size = self.filter_size ** 2
        # 生成随机正交滤波器
        W, b = normal_random(input_unit=channels * self.filter_size ** 2, hidden_unit=self.n_hidden)
        W = orthonormalize(W)  # 正交后的幅度在-1~+1之间
        # 将所有输入的通道的patch串联
        patches = im2col_catch_compiled(inputX, self.im2colfn_getbeta)
        ##########################
        patches = norm2d(patches)
        # patches, self.mean1, self.P1 = whiten2d(patches)
        ##########################
        # 在patches上加噪
        if self.noise_type in ('mn_array_edge', 'mn_array_edge_mch'):  # 需要原始图像
            self.noise_args['originX'] = inputX
        noise_patches = np.copy(patches)
        if self.noise_type in ('mn_block_mch', 'mn_array_mch', 'mn_array_orient_mch', 'mn_array_edge_mch'):  # 5d
            noise_patches = noise_patches.reshape((batches, self.orows_, self.ocols_, channels, patch_size))
            noise_patches = noise_patches.transpose((0, 4, 3, 1, 2))
        elif self.noise_type in ('mn_block', 'mn_array', 'mn_array_orient', 'mn_array_edge'):  # 4d
            noise_patches = noise_patches.reshape((batches, self.orows_, self.ocols_, channels * patch_size))
            noise_patches = noise_patches.transpose((0, 3, 1, 2))
            # 直接对patch
            # noise_patches = noise_patches.reshape((patches.shape[0] * channels, -1))
        noise_patches = add_noise_decomp(noise_patches, self.noise_type, self.noise_args)
        if self.noise_type in ('mn_block_mch', 'mn_array_mch', 'mn_array_orient_mch', 'mn_array_edge_mch'):
            noise_patches = noise_patches.transpose((0, 3, 4, 2, 1))
            noise_patches = noise_patches.reshape((-1, channels * patch_size))
        elif self.noise_type in ('mn_block', 'mn_array', 'mn_array_orient', 'mn_array_edge'):
            noise_patches = noise_patches.transpose((0, 2, 3, 1))
            noise_patches = noise_patches.reshape((-1, channels * patch_size))
            # 直接对patch
            # noise_patches = noise_patches.reshape((patches.shape[0], -1))
        hiddens = np.dot(noise_patches, W)
        del noise_patches
        hmax, hmin = np.max(hiddens, axis=0), np.min(hiddens, axis=0)
        scale = (hmax - hmin) / (2 * bias_scale)
        hiddens += b * scale
        hiddens = activate(hiddens, self.act_mode)
        # 计算beta
        beta = compute_beta_direct(hiddens, patches)
        beta = beta.T
        return beta

    def forward_decomp(self, inputX, beta):
        assert inputX.ndim == 4
        batchSize = int(round(float(inputX.shape[0]) / 10))
        splits = int(np.ceil(float(inputX.shape[0]) / batchSize))
        patches = []
        for _ in xrange(splits):
            patchestmp = im2col_catch_compiled(inputX[:batchSize], self.im2colfn_forward)
            inputX = inputX[batchSize:]
            # 归一化
            patchestmp = norm2d(patchestmp)
            # patchestmp = whiten2d(patchestmp, self.mean1, self.P1)
            patchestmp = np.dot(patchestmp, beta)
            patches = np.concatenate([patches, patchestmp], axis=0) if len(patches) != 0 else patchestmp
        return patches

    def get_train_output_for(self, inputX):
        utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmin')
        batches, channels, rows, cols = inputX.shape
        oshape_ = utils.basic.conv_out_shape((batches, channels, rows, cols),
                                             (self.n_hidden, channels, self.filter_size, self.filter_size),
                                             pad=self.pad_, stride=self.stride_)
        self.orows_, self.ocols_ = oshape_[-2:]
        oshape = utils.basic.conv_out_shape((batches, channels, rows, cols),
                                            (self.n_hidden, channels, self.filter_size, self.filter_size),
                                            pad=self.pad, stride=self.stride)
        self.orows, self.ocols = oshape[-2:]
        self.im2colfn_getbeta = im2col_compfn((rows, cols), self.filter_size, stride=self.stride_,
                                              pad=self.pad_, ignore_border=False)
        self.im2colfn_forward = im2col_compfn((rows, cols), self.filter_size, stride=self.stride,
                                              pad=self.pad, ignore_border=False)
        # 学习beta
        self.beta = self._get_beta(inputX)
        utils.visual.save_beta_mch(self.beta, channels, dir_name, 'beta')
        # 前向计算
        inputX = self.forward_decomp(inputX, self.beta)
        inputX = inputX.reshape((batches, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
        utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmraw')
        # 归一化
        # inputX = norm4d(inputX)
        # utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmnorm')
        # 激活
        inputX = activate(inputX, self.act_mode)
        utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmrelu')
        # 添加cccp层组合
        self.cccp = CCCPLayer(C=self.C, name=dir_name, n_out=self.cccp_out, act_mode=self.act_mode,
                              noise_type=self.cccp_noise_type, noise_args=self.cccp_noise_args)
        if self.add_cccp: inputX = self.cccp.get_train_output_for(inputX)
        # 池化
        inputX = pool_op(inputX, self.pool_size, self.pool_type, self.mode, self.pool_args)
        utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmpool')
        return inputX

    def get_test_output_for(self, inputX):
        utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elminte')
        batches, channels, rows, cols = inputX.shape
        # 前向计算
        inputX = self.forward_decomp(inputX, self.beta)
        inputX = inputX.reshape((batches, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
        utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmrawte')
        # 归一化
        # inputX = norm4d(inputX)
        # utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmnormte')
        # 激活
        inputX = activate(inputX, self.act_mode)
        utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmrelute')
        # 添加cccp层组合
        if self.add_cccp: inputX = self.cccp.get_test_output_for(inputX)
        # 池化
        inputX = pool_op(inputX, self.pool_size, pool_type=self.pool_type, mode=self.mode)
        utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmpoolte')
        return inputX


class ELMAECrossPartLayer(Layer):
    def __init__(self, C, name, n_hidden, act_mode, filter_size, pad, stride, pad_, stride_,
                 noise_type, noise_args, cross_size, pool_type, pool_size, mode, pool_args,
                 cccp_out, cccp_noise_type, cccp_noise_args, add_cccp):
        global dir_name
        dir_name = name
        self.C = C
        self.n_hidden = n_hidden
        self.act_mode = act_mode
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.stride_ = stride_
        self.pad_ = pad_
        self.noise_type = noise_type
        self.noise_args = noise_args
        self.cross_size = cross_size
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.mode = mode
        self.pool_args = pool_args
        self.cccp_out = cccp_out
        self.cccp_noise_type = cccp_noise_type
        self.cccp_noise_args = cccp_noise_args
        self.add_cccp = add_cccp

    def _get_beta(self, partX, bias_scale=10):
        assert partX.ndim == 4
        batches, channels = partX.shape[:2]
        patch_size = self.filter_size ** 2
        # 生成随机正交滤波器
        W, b = normal_random(input_unit=channels * self.filter_size ** 2, hidden_unit=self.n_hidden)
        W = orthonormalize(W)  # 正交后的幅度在-1~+1之间
        # 将所有输入的通道的patch串联
        patches = im2col_catch_compiled(partX, self.im2colfn_getbeta)
        ##########################
        patches = norm2d(patches)
        # patches, self.mean1, self.P1 = whiten2d(patches)
        ##########################
        # 在patches上加噪
        if self.noise_type in ('mn_array_edge', 'mn_array_edge_mch'):  # 需要原始图像
            self.noise_args['originX'] = partX
        noise_patches = np.copy(patches)
        if self.noise_type in ('mn_block_mch', 'mn_array_mch', 'mn_array_orient_mch', 'mn_array_edge_mch'):  # 5d
            noise_patches = noise_patches.reshape((batches, self.orows_, self.ocols_, channels, patch_size))
            noise_patches = noise_patches.transpose((0, 4, 3, 1, 2))
        elif self.noise_type in ('mn_block', 'mn_array', 'mn_array_orient', 'mn_array_edge'):  # 4d
            noise_patches = noise_patches.reshape((batches, self.orows_, self.ocols_, channels * patch_size))
            noise_patches = noise_patches.transpose((0, 3, 1, 2))
            # 直接对patch
            # noise_patches = noise_patches.reshape((patches.shape[0] * channels, -1))
        noise_patches = add_noise_decomp(noise_patches, self.noise_type, self.noise_args)
        if self.noise_type in ('mn_block_mch', 'mn_array_mch', 'mn_array_orient_mch', 'mn_array_edge_mch'):
            noise_patches = noise_patches.transpose((0, 3, 4, 2, 1))
            noise_patches = noise_patches.reshape((-1, channels * patch_size))
        elif self.noise_type in ('mn_block', 'mn_array', 'mn_array_orient', 'mn_array_edge'):
            noise_patches = noise_patches.transpose((0, 2, 3, 1))
            noise_patches = noise_patches.reshape((-1, channels * patch_size))
            # 直接对patch
            # noise_patches = noise_patches.reshape((patches.shape[0], -1))
        hiddens = np.dot(noise_patches, W)
        del noise_patches
        hmax, hmin = np.max(hiddens, axis=0), np.min(hiddens, axis=0)
        scale = (hmax - hmin) / (2 * bias_scale)
        hiddens += b * scale
        hiddens = activate(hiddens, self.act_mode)
        # 计算beta
        beta = compute_beta_direct(hiddens, patches)
        beta = beta.T
        return beta

    def forward_decomp(self, inputX, beta):
        assert inputX.ndim == 4
        batchSize = int(round(float(inputX.shape[0]) / 10))
        splits = int(np.ceil(float(inputX.shape[0]) / batchSize))
        patches = []
        for _ in xrange(splits):
            patchestmp = im2col_catch_compiled(inputX[:batchSize], self.im2colfn_forward)
            inputX = inputX[batchSize:]
            # 归一化
            patchestmp = norm2d(patchestmp)
            # patchestmp = whiten2d(patchestmp, self.mean1, self.P1)
            patchestmp = np.dot(patchestmp, beta)
            patches = np.concatenate([patches, patchestmp], axis=0) if len(patches) != 0 else patchestmp
        return patches

    def get_train_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        oshape_ = utils.basic.conv_out_shape((batches, channels, rows, cols),
                                             (self.n_hidden, channels, self.filter_size, self.filter_size),
                                             pad=self.pad_, stride=self.stride_, ignore_border=False)
        self.orows_, self.ocols_ = oshape_[-2:]
        oshape = utils.basic.conv_out_shape((batches, channels, rows, cols),
                                            (self.n_hidden, channels, self.filter_size, self.filter_size),
                                            pad=self.pad, stride=self.stride, ignore_border=False)
        self.orows, self.ocols = oshape[-2:]
        self.im2colfn_getbeta = im2col_compfn((rows, cols), self.filter_size, stride=self.stride_,
                                              pad=self.pad_, ignore_border=False)
        self.im2colfn_forward = im2col_compfn((rows, cols), self.filter_size, stride=self.stride,
                                              pad=self.pad, ignore_border=False)
        # 将输入按照通道分为多个组,每个组学习一个beta
        self.filters = []
        self.cccps = []
        output = []
        splits = int(np.ceil(float(channels) / self.cross_size))
        for num in xrange(splits):
            # 取部分通道
            partX = inputX[:, :self.cross_size, :, :]
            inputX = inputX[:, self.cross_size:, :, :]
            utils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmin')
            # 学习beta
            beta = self._get_beta(partX)
            utils.visual.save_beta_mch(beta, self.cross_size, dir_name, 'beta')
            # 前向计算
            partX = self.forward_decomp(partX, beta)
            partX = partX.reshape((batches, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
            utils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmraw')
            # 归一化
            # partX = norm4d(partX)
            # utils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmnorm')
            # 激活
            partX = activate(partX, self.act_mode)
            utils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmrelu')
            # 添加cccp层组合
            cccp = CCCPLayer(C=self.C, name=dir_name, n_out=self.cccp_out, act_mode=self.act_mode,
                             noise_type=self.cccp_noise_type, noise_args=self.cccp_noise_args)
            if self.add_cccp: partX = cccp.get_train_output_for(partX)
            # 池化
            partX = pool_op(partX, self.pool_size, self.pool_type, self.mode, self.pool_args)
            utils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmpool')
            # 组合最终结果
            output = np.concatenate([output, partX], axis=1) if len(output) != 0 else partX
            self.filters.append(copy(beta))
            self.cccps.append(deepcopy(cccp))
            print num,
            gc.collect()
        return output

    def get_test_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        output = []
        splits = int(np.ceil(float(channels) / self.cross_size))
        for num in xrange(splits):
            # 取部分通道
            partX = inputX[:, :self.cross_size, :, :]
            inputX = inputX[:, self.cross_size:, :, :]
            utils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elminte')
            # 前向计算
            partX = self.forward_decomp(partX, self.filters[num])
            partX = partX.reshape((batches, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
            utils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmrawte')
            # 归一化
            # partX = norm4d(partX)
            # utils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmnormte')
            # 激活
            partX = activate(partX, self.act_mode)
            utils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmrelute')
            # 添加cccp层组合
            if self.add_cccp: partX = self.cccps[num].get_test_output_for(partX)
            # 池化
            partX = pool_op(partX, self.pool_size, self.pool_type, self.mode, self.pool_args)
            utils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmpoolte')
            # 组合最终结果
            output = np.concatenate([output, partX], axis=1) if len(output) != 0 else partX
            print num,
            gc.collect()
        return output


########################################################################################################################


def get_indexed(inputX, idx):
    assert inputX.ndim == 4
    batches, rows, cols, patch_size = inputX.shape
    if idx.ndim == 2:  # 对应于block和neib
        idx_rows, idx_cols = idx.shape
        inputX = inputX.reshape((batches, -1, patch_size))
        idx = idx.ravel()
        inputX = inputX[:, idx, :]
        return inputX.reshape((batches, idx_rows, idx_cols, patch_size))
    elif idx.ndim == 1:  # 对应于rand
        inputX = inputX.reshape((batches, -1, patch_size))
        inputX = inputX[:, idx, :]
        return inputX[:, None, :, :]
    else:
        raise NotImplementedError


def join_result(result, idx_list):
    assert result.ndim == 3
    idx_list = map(lambda x: x.ravel(), idx_list)
    all_idx = np.concatenate(idx_list)
    sort_idx = np.argsort(all_idx)
    result = result[:, sort_idx, :]
    return result


# 分块取索引,part_size表示分割出块的多少
def get_block_idx(part_size, orows, ocols):
    nr, nc = part_size
    blockr = int(np.ceil(float(orows) / nr))
    blockc = int(np.ceil(float(ocols) / nc))
    idx = []
    for row in xrange(nr):
        row_bias = row * blockr
        for col in xrange(nc):
            col_bias = col * blockc
            base = np.arange(blockc) if col_bias + blockc < ocols else np.arange(ocols - col_bias)
            block_row = blockr if row_bias + blockr < orows else orows - row_bias
            one_block = []
            for br in xrange(block_row):
                one_row = base + orows * br + col_bias + row_bias * orows
                one_block = np.concatenate([one_block, one_row[None, :]], axis=0) \
                    if len(one_block) != 0 else one_row[None, :]
            idx.append(copy(one_block))
    return idx


# 类似池化的方式取邻域索引,part_size表示分割出邻域的多少
def get_neib_idx(part_size, orows, ocols):
    nr, nc = part_size
    idx = []
    for i in xrange(nr):
        row_idx = np.arange(i, orows, nr)
        for j in xrange(nc):
            col_idx = np.arange(j, ocols, nc)
            one_neib = []
            for row_step in row_idx:
                one_row = col_idx + row_step * orows
                one_neib = np.concatenate([one_neib, one_row[None, :]], axis=0) \
                    if len(one_neib) != 0 else one_row[None, :]
            idx.append(copy(one_neib))
    return idx


def get_rand_idx(n_rand, orows, ocols):
    size = orows * ocols
    split_size = int(round(float(size) / n_rand))
    all_idx = np.random.permutation(size)
    split_range = [split_size + split_size * i for i in xrange(n_rand - 1)]
    split_idx = np.split(all_idx, split_range)
    return split_idx


class ELMAEMBetaLayer(Layer):
    def __init__(self, C, name, n_hidden, act_mode, filter_size, pad, stride, noise_type, noise_args,
                 part_size, idx_type, pool_type, pool_size, mode, pool_args,
                 cccp_out, cccp_noise_type, cccp_noise_args, add_cccp):
        global dir_name
        dir_name = name
        self.C = C
        self.n_hidden = n_hidden
        self.act_mode = act_mode
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.noise_type = noise_type  # 不允许为mch
        self.noise_args = noise_args
        self.part_size = part_size
        assert (idx_type == 'rand') ^ \
               (noise_type in ('mn_block', 'mn_array', 'mn_block_mch', 'mn_array_mch'))
        if idx_type == 'block':
            self.get_idx = get_block_idx
        elif idx_type == 'neib':
            self.get_idx = get_neib_idx
        elif idx_type == 'rand':
            self.get_idx = get_rand_idx
        else:
            raise NameError
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.mode = mode
        self.pool_args = pool_args
        self.cccp_out = cccp_out
        self.cccp_noise_type = cccp_noise_type
        self.cccp_noise_args = cccp_noise_args
        self.add_cccp = add_cccp

    def _get_beta(self, part_patches, W, b, bias_scale=10):
        assert part_patches.ndim == 4
        batches, rows, cols, patch_size = part_patches.shape
        # 生成随机正交滤波器
        # W, b = normal_random(input_unit=self.filter_size ** 2, hidden_unit=self.n_hidden)
        W = orthonormalize(W)  # 正交后的幅度在-1~+1之间
        # 在patches上加噪
        noise_patches = np.copy(part_patches)
        part_patches = part_patches.reshape(-1, patch_size)
        if self.noise_type in ('mn_block', 'mn_array', 'mn_array_orient'):  # 4d
            noise_patches = noise_patches.transpose((0, 3, 1, 2))
        noise_patches = add_noise_decomp(noise_patches, self.noise_type, self.noise_args)
        if self.noise_type in ('mn_block', 'mn_array', 'mn_array_orient'):
            noise_patches = noise_patches.transpose((0, 2, 3, 1))
        noise_patches = noise_patches.reshape((-1, patch_size))
        hiddens = np.dot(noise_patches, W)
        del noise_patches
        hmax, hmin = np.max(hiddens, axis=0), np.min(hiddens, axis=0)
        scale = (hmax - hmin) / (2 * bias_scale)
        hiddens += b * scale
        hiddens = activate(hiddens, self.act_mode)
        # 计算beta
        beta = compute_beta_direct(hiddens, part_patches)
        beta = beta.T
        return beta

    def _train_forward(self, oneChannel):
        batches = oneChannel.shape[0]
        patch_size = self.filter_size ** 2
        patches = self.im2colfn(oneChannel)
        del oneChannel
        ##########################
        patches = norm2d(patches)
        # patches = whiten2d(patches, self.mean1, self.P1)
        ##########################
        patches = patches.reshape((batches, self.orows, self.ocols, patch_size))
        W, b = normal_random(input_unit=patch_size, hidden_unit=self.n_hidden)
        self.idx = self.get_idx(self.part_size, self.orows, self.ocols)
        filters = []
        output = []
        for num in xrange(len(self.idx)):
            part_patches = get_indexed(patches, self.idx[num])
            beta = self._get_beta(part_patches, W, b)
            utils.visual.save_beta(beta, dir_name, 'beta')
            part_patches = part_patches.reshape(-1, patch_size)
            part_patches = np.dot(part_patches, beta)
            part_patches = part_patches.reshape((batches, -1, self.n_hidden))
            output = np.concatenate([output, part_patches], axis=1) if len(output) != 0 else part_patches
            filters.append(copy(beta))
        output = join_result(output, self.idx)
        output = output.reshape((batches, self.orows, self.ocols, self.n_hidden)).transpose((0, 3, 1, 2))
        return output, filters

    def get_train_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        oshape = utils.basic.conv_out_shape((batches, channels, rows, cols),
                                            (self.n_hidden, channels, self.filter_size, self.filter_size),
                                            pad=self.pad, stride=self.stride, ignore_border=False)
        self.orows, self.ocols = oshape[-2:]
        self.im2colfn = im2col_compfn((rows, cols), self.filter_size, stride=self.stride,
                                              pad=self.pad, ignore_border=False)
        self.filters = []
        self.cccps = []
        output = []
        for ch in xrange(channels):
            oneChannel = inputX[:, [0], :, :]
            inputX = inputX[:, 1:, :, :]
            utils.visual.save_map(oneChannel[[10, 100, 1000]], dir_name, 'elmin')
            oneChannelOut, oneChannelFilter = self._train_forward(oneChannel)
            utils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmraw')
            # 归一化
            # oneChannelOut = norm4d(oneChannelOut)
            # utils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmnorm')
            # 激活
            oneChannelOut = activate(oneChannelOut, self.act_mode)
            utils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmrelu')
            # 添加cccp层组合
            cccp = CCCPLayer(C=self.C, name=dir_name, n_out=self.cccp_out, act_mode=self.act_mode,
                             noise_type=self.cccp_noise_type, noise_args=self.cccp_noise_args)
            if self.add_cccp: oneChannelOut = cccp.get_train_output_for(oneChannelOut)
            # 池化
            oneChannelOut = pool_op(oneChannelOut, self.pool_size, self.pool_type, self.mode, self.pool_args)
            utils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmpool')
            # 组合最终结果
            output = np.concatenate([output, oneChannelOut], axis=1) if len(output) != 0 else oneChannelOut
            self.filters.append(deepcopy(oneChannelFilter))
            self.cccps.append(deepcopy(cccp))
            print ch,
            gc.collect()
        return output

    def _test_forward(self, oneChannel, oneChannelFilter):
        batches = oneChannel.shape[0]
        patch_size = self.filter_size ** 2
        patches = self.im2colfn(oneChannel)
        del oneChannel
        ##########################
        patches = norm2d(patches)
        # patches = whiten2d(patches, self.mean1, self.P1)
        ##########################
        patches = patches.reshape((batches, self.orows, self.ocols, patch_size))
        output = []
        for num in xrange(len(self.idx)):
            part_patches = get_indexed(patches, self.idx[num])
            part_patches = part_patches.reshape(-1, patch_size)
            part_patches = np.dot(part_patches, oneChannelFilter[num])
            part_patches = part_patches.reshape((batches, -1, self.n_hidden))
            output = np.concatenate([output, part_patches], axis=1) if len(output) != 0 else part_patches
        output = join_result(output, self.idx)
        output = output.reshape((batches, self.orows, self.ocols, self.n_hidden)).transpose((0, 3, 1, 2))
        return output

    def get_test_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        output = []
        for ch in xrange(channels):
            oneChannel = inputX[:, [0], :, :]
            inputX = inputX[:, 1:, :, :]
            utils.visual.save_map(oneChannel[[10, 100, 1000]], dir_name, 'elminte')
            oneChannelOut = self._test_forward(oneChannel, self.filters[ch])
            utils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmrawte')
            # 归一化
            # oneChannelOut = norm4d(oneChannelOut)
            # utils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmnormte')
            # 激活
            oneChannelOut = activate(oneChannelOut, self.act_mode)
            utils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmrelute')
            # 添加cccp层组合
            if self.add_cccp: oneChannelOut = self.cccps[ch].get_test_output_for(oneChannelOut)
            # 池化
            oneChannelOut = pool_op(oneChannelOut, self.pool_size, self.pool_type, self.mode, self.pool_args)
            utils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmpoolte')
            # 组合最终结果
            output = np.concatenate([output, oneChannelOut], axis=1) if len(output) != 0 else oneChannelOut
            print ch,
            gc.collect()
        return output


########################################################################################################################


class PoolLayer(Layer):
    def __init__(self, pool_type, pool_size, mode='max', pool_args=None):
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.mode = mode
        self.pool_args = pool_args

    def get_train_output_for(self, inputX):
        output = pool_op(inputX, self.pool_size, self.pool_type, self.mode, self.pool_args)
        return output

    def get_test_output_for(self, inputX):
        output = pool_op(inputX, self.pool_size, self.pool_type, self.mode, self.pool_args)
        return output


class BNLayer(Layer):
    def get_train_output_for(self, inputX, regularization=0.1):
        inputX, self.mean, self.norm = norm4dglobal(inputX, regularization)
        return inputX

    def get_test_output_for(self, inputX, regularization=0.1):
        return norm4dglobal(inputX, self.mean, self.norm, regularization)


class GCNLayer(Layer):
    def get_train_output_for(self, inputX, regularization=0.1):
        return norm4d(inputX, regularization)

    def get_test_output_for(self, inputX, regularization=0.1):
        return norm4d(inputX, regularization)


class MergeLayer(Layer):
    def __init__(self, subnet):
        assert isinstance(subnet, OrderedDict)
        self.subnet = subnet

    def get_train_output_for(self, inputX):
        output = []
        for name, layer in self.subnet.iteritems():
            out = layer.get_train_output_for(inputX)
            output.append(np.copy(out))
            print 'add ' + name,
        return np.concatenate(output, axis=1)

    def get_test_output_for(self, inputX):
        output = []
        for name, layer in self.subnet.iteritems():
            out = layer.get_test_output_for(inputX)
            output.append(np.copy(out))
        return np.concatenate(output, axis=1)


class SPPLayer(Layer):
    def __init__(self, pool_dims):
        self.pool_dims = pool_dims

    def get_train_output_for(self, inputX):
        input_size = inputX.shape[2:]
        pool_list = []
        for pool_dim in self.pool_dims:
            win_size = tuple((i + pool_dim - 1) // pool_dim for i in input_size)
            str_size = tuple(i // pool_dim for i in input_size)
            pool = pool_fn(inputX, pool_size=win_size, stride=str_size)
            pool = pool.reshape((pool.shape[0], pool.shape[1], -1))
            pool_list.append(copy(pool))
        pooled = np.concatenate(pool_list, axis=2)
        return pooled.reshape((-1, pooled.shape[2]))

    def get_test_output_for(self, inputX):
        return self.get_train_output_for(inputX)


class CCCPLayer(Layer):
    def __init__(self, C, name, n_out, act_mode, noise_type, noise_args):
        global dir_name
        dir_name = name
        self.C = C
        self.n_out = n_out
        self.act_mode = act_mode
        self.noise_type = noise_type
        self.noise_args = noise_args

    def _get_beta(self, inputX, bias_scale=25):
        assert inputX.ndim == 4
        batches, rows, cols, n_in = inputX.shape
        W, b = normal_random(input_unit=n_in, hidden_unit=self.n_out)
        W = orthonormalize(W)
        # 在转化的矩阵上加噪
        if self.noise_type == 'mn_array_edge':  # 需要原始图像
            self.noise_args['originX'] = inputX.transpose((0, 3, 1, 2))
        noiseX = np.copy(inputX)
        inputX = inputX.reshape((-1, n_in))
        if self.noise_type in ('mn_block', 'mn_array', 'mn_array_orient', 'mn_array_edge'):  # 4d
            noiseX = noiseX.transpose((0, 3, 1, 2))
        noiseX = add_noise_decomp(noiseX, self.noise_type, self.noise_args)
        if self.noise_type in ('mn_block', 'mn_array', 'mn_array_orient', 'mn_array_edge'):
            noiseX = noiseX.transpose((0, 2, 3, 1))
        noiseX = noiseX.reshape((-1, n_in))
        H = np.dot(noiseX, W)
        del noiseX
        hmax, hmin = np.max(H, axis=0), np.min(H, axis=0)
        scale = (hmax - hmin) / (2 * bias_scale)
        H += b * scale
        H = activate(H, self.act_mode)
        beta = compute_beta_direct(H, inputX)
        beta = beta.T
        return beta

    def get_train_output_for(self, inputX):
        utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpin')
        batches, n_in, rows, cols = inputX.shape
        ##################
        inputX = inputX.transpose((0, 2, 3, 1)).reshape((-1, n_in))
        inputX = norm2d(inputX)
        ##################
        inputX = inputX.reshape((batches, rows, cols, -1))
        self.beta = self._get_beta(inputX)
        inputX = inputX.reshape((-1, n_in))
        # 前向计算
        inputX = np.dot(inputX, self.beta)
        inputX = inputX.reshape((batches, rows, cols, -1)).transpose((0, 3, 1, 2))
        utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpraw')
        # 归一化
        # inputX = norm4d(inputX)
        # utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpnorm')
        # 激活
        inputX = activate(inputX, self.act_mode)
        utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccprelu')
        return inputX

    def get_test_output_for(self, inputX):
        utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpinte')
        batches, n_in, rows, cols = inputX.shape
        ##################
        inputX = inputX.transpose((0, 2, 3, 1)).reshape((-1, n_in))
        inputX = norm2d(inputX)
        ##################
        # 前向计算
        inputX = np.dot(inputX, self.beta)
        inputX = inputX.reshape((batches, rows, cols, -1)).transpose((0, 3, 1, 2))
        utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccprawte')
        # 归一化
        # inputX = norm4d(inputX)
        # utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpnormte')
        # 激活
        inputX = activate(inputX, self.act_mode)
        utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccprelute')
        return inputX


class CCCPMBetaLayer(Layer):
    def __init__(self, C, name, n_out, act_mode, noise_type, noise_args, part_size, idx_type):
        global dir_name
        dir_name = name
        self.C = C
        self.n_out = n_out
        self.act_mode = act_mode
        self.noise_type = noise_type
        self.noise_args = noise_args
        self.part_size = part_size
        if idx_type == 'block':
            self.get_idx = get_block_idx
        elif idx_type == 'neib':
            self.get_idx = get_neib_idx
        elif idx_type == 'rand':
            self.get_idx = get_rand_idx
        else:
            raise NameError

    def _get_beta(self, partX, W, b, bias_scale=25):
        assert partX.ndim == 4
        batches, rows, cols, n_in = partX.shape
        # W, b = normal_random(input_unit=self.n_in, hidden_unit=self.n_out)
        W = orthonormalize(W)
        # 在转化的矩阵上加噪
        noiseX = np.copy(partX)
        partX = partX.reshape((-1, n_in))
        if self.noise_type in ('mn_block', 'mn_array', 'mn_array_orient'):  # 4d
            noiseX = noiseX.transpose((0, 3, 1, 2))
        noiseX = add_noise_decomp(noiseX, self.noise_type, self.noise_args)
        if self.noise_type in ('mn_block', 'mn_array', 'mn_array_orient'):
            noiseX = noiseX.transpose((0, 2, 3, 1))
        noiseX = noiseX.reshape((-1, n_in))
        H = np.dot(noiseX, W)
        del noiseX
        hmax, hmin = np.max(H, axis=0), np.min(H, axis=0)
        scale = (hmax - hmin) / (2 * bias_scale)
        H += b * scale
        H = activate(H, self.act_mode)
        beta = compute_beta_direct(H, partX)
        beta = beta.T
        return beta

    def _train_forward(self, inputX):
        batches, n_in, rows, cols = inputX.shape
        ##################
        inputX = inputX.transpose((0, 2, 3, 1)).reshape((-1, n_in))
        inputX = norm2d(inputX)
        inputX = inputX.reshape((batches, rows, cols, -1))
        ##################
        W, b = normal_random(input_unit=n_in, hidden_unit=self.n_out)
        self.idx = self.get_idx(self.part_size, rows, cols)
        self.filters = []
        output = []
        for num in xrange(len(self.idx)):
            partX = get_indexed(inputX, self.idx[num])
            beta = self._get_beta(partX, W, b)
            partX = partX.reshape((-1, n_in))
            partX = np.dot(partX, beta)
            partX = partX.reshape((batches, -1, self.n_out))
            output = np.concatenate([output, partX], axis=1) if len(output) != 0 else partX
            self.filters.append(copy(beta))
        output = join_result(output, self.idx)
        output = output.reshape((batches, rows, cols, self.n_out)).transpose((0, 3, 1, 2))
        return output

    def get_train_output_for(self, inputX):
        utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpin')
        inputX = self._train_forward(inputX)
        utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpraw')
        # 归一化
        # inputX = norm4d(inputX)
        # utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpnorm')
        # 激活
        inputX = activate(inputX, self.act_mode)
        utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccprelu')
        return inputX

    def _test_forward(self, inputX):
        batches, n_in, rows, cols = inputX.shape
        ##################
        inputX = inputX.transpose((0, 2, 3, 1)).reshape((-1, n_in))
        inputX = norm2d(inputX)
        inputX = inputX.reshape((batches, rows, cols, -1))
        ##################
        output = []
        for num in xrange(len(self.idx)):
            partX = get_indexed(inputX, self.idx[num])
            partX = partX.reshape((-1, n_in))
            partX = np.dot(partX, self.filters[num])
            partX = partX.reshape((batches, -1, self.n_out))
            output = np.concatenate([output, partX], axis=1) if len(output) != 0 else partX
        output = join_result(output, self.idx)
        output = output.reshape((batches, rows, cols, self.n_out)).transpose((0, 3, 1, 2))
        return output

    def get_test_output_for(self, inputX):
        utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpinte')
        inputX = self._test_forward(inputX)
        utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccprawte')
        # 归一化
        # inputX = norm4d(inputX)
        # utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpnormte')
        # 激活
        inputX = activate(inputX, self.act_mode)
        utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccprelute')
        return inputX
