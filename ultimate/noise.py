# coding:utf-8
import numpy as np
from copy import copy
import cv2
from skimage import feature
from util import *

__all__ = ['add_noise_decomp']


def add_noise_decomp(X, noise_type, args):
    noise_dict = {'mn': add_mn, 'gs': add_gs, 'sp': add_sp, 'gs_part': add_gs_part, 'mn_gs': add_mn_gs,
                  'mn_block': add_mn_block, 'mn_block_mch': add_mn_block_mch,
                  'mn_array': add_mn_array, 'mn_array_mch': add_mn_array_mch,
                  'mn_array_orient': add_mn_array_orient, 'mn_array_orient_mch': add_mn_array_orient_mch,
                  'mn_array_edge': add_mn_array_edge, 'mn_array_edge_mch': add_mn_array_edge_mch}
    if noise_type not in noise_dict.keys():
        raise NotImplementedError
    noise_fn = noise_dict[noise_type]
    size = X.shape[0]
    batchSize = size / 10
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    for start, end in zip(startRange, endRange):
        X[start:end] = noise_fn(X[start:end], **args)
    return X


def add_mn(X, percent=0.5):
    retain_prob = 1. - percent
    binomial = np.random.uniform(low=0., high=1., size=X.shape)
    binomial = np.asarray(binomial < retain_prob, dtype=float)
    return X * binomial


def add_sp(X, percent=0.5):
    Xmin, Xmax = np.min(X), np.max(X)
    uniform = np.random.uniform(low=0., high=1., size=X.shape)
    salt = np.where((uniform > 0) * (uniform <= percent / 2.))
    pepper = np.where((uniform > percent / 2.) * (uniform <= percent))
    X[salt] = Xmin
    X[pepper] = Xmax
    return X


def add_gs(X, scale=100., std=None):
    if std is None:
        Xmin, Xmax = np.min(X), np.max(X)
        std = (Xmax - Xmin) / (2. * scale)
    normal = np.random.normal(loc=0, scale=std, size=X.shape)
    X += normal
    return X


def add_gs_part(X, percent=0.5, scale=100., std=None):
    if std is None:
        Xmin, Xmax = np.min(X), np.max(X)
        std = (Xmax - Xmin) / (2. * scale)
    retain_prob = 1. - percent
    binomial = np.random.uniform(low=0., high=1., size=X.shape)
    binomial = np.asarray(binomial < retain_prob, dtype=float)
    normal = np.random.normal(loc=0, scale=std, size=X.shape)
    normal *= binomial
    X += normal
    return X


def add_mn_gs(X, percent=0.5, scale=100., std=None):
    if std is None:
        Xmin, Xmax = np.min(X), np.max(X)
        std = (Xmax - Xmin) / (2. * scale)
    normal = np.random.normal(loc=0, scale=std, size=X.shape)
    X += normal
    retain_prob = 1. - percent
    binomial = np.random.uniform(low=0., high=1., size=X.shape)
    binomial = np.asarray(binomial < retain_prob, dtype=float)
    X *= binomial
    return X


def block_mask(index, row_size, col_size):
    assert len(index) == 3
    batch_idx, row_idx, col_idx = index
    # 样本简单重复
    batch_idx = np.repeat(np.tile(batch_idx, row_size), col_size)
    # 行列计算相邻索引
    length = len(row_idx)
    row_idx = np.tile(np.repeat(row_idx, col_size), row_size)
    bias = np.repeat(np.arange(row_size), length * col_size)
    row_idx += bias
    col_idx = np.repeat(np.tile(col_idx, row_size), col_size)
    bias = np.tile(np.arange(col_size), length * row_size)
    col_idx += bias
    return batch_idx, row_idx, col_idx


def add_mn_block(X, percent=0.5, block_list=((1, 1), (2, 2))):
    assert X.ndim == 4
    Xshape = X.shape
    batches, channels, rows, cols = Xshape
    X = X.reshape((-1, rows, cols))
    block_list = filter(lambda x: x[0] <= rows and x[1] <= cols, block_list)
    interval = 1. / len(block_list)
    pequal = percent / sum(map(lambda x: np.prod(x), block_list))
    uniform = np.random.uniform(low=0., high=1., size=X.shape)
    start = 0.
    for block_row, block_col in block_list:
        index = np.where((uniform > start) * (uniform <= start + pequal))
        index_mask = block_mask(index, block_row, block_col)
        out_of_bond = np.where((index_mask[-2] >= rows) + (index_mask[-1] >= cols))
        index_mask = map(lambda x: np.delete(x, out_of_bond), index_mask)
        X[index_mask] = 0.
        start += interval
    X = X.reshape(Xshape)
    return X


def add_mn_block_mch(X, percent=0.5, block_list=((1, 1), (2, 2))):
    assert X.ndim == 5
    Xshape = X.shape
    batches, channels, mch, rows, cols = Xshape
    X = X.reshape((-1, mch, rows, cols))
    block_list = filter(lambda x: x[0] <= rows and x[1] <= cols, block_list)
    interval = 1. / len(block_list)
    pequal = percent / (channels * sum(map(lambda x: np.prod(x), block_list)))
    uniform = np.random.uniform(low=0., high=1., size=(batches, rows, cols))
    start = 0.
    for block_row, block_col in block_list:
        index = np.where((uniform > start) * (uniform <= start + pequal))
        index_mask = block_mask(index, block_row, block_col)
        out_of_bond = np.where((index_mask[-2] >= rows) + (index_mask[-1] >= cols))
        index_mask = map(lambda x: np.delete(x, out_of_bond), index_mask)
        X[index_mask[0], :, index_mask[1], index_mask[2]] = 0.  # 所有通道的同一位置
        start += interval
    X = X.reshape(Xshape)
    return X


########################################################################################################################


def scat_stride(orows, ocols, blockr, blockc, total_blocks):
    strider = strider_tmp = 1
    stridec = stridec_tmp = 1
    while True:
        arrayr = (orows - blockr) // strider_tmp + 1
        arrayc = (ocols - blockc) // stridec_tmp + 1
        if arrayr * arrayc < total_blocks: break
        strider = strider_tmp
        stridec = stridec_tmp
        if strider_tmp < stridec_tmp:
            strider_tmp += 1
        else:
            stridec_tmp += 1
    return strider, stridec


def assign_ch_idx_permap(channels, n_blocks):
    ch_idx = np.repeat(np.arange(channels), n_blocks)
    return ch_idx


# 均匀分配通道索引
def assign_ch_idx_uniform(channels, n_blocks):
    ch_idx = np.random.permutation(channels)[:n_blocks]
    if channels < n_blocks:
        times = int(np.ceil(float(n_blocks) / channels))
        ch_idx = np.tile(ch_idx, times)[:n_blocks]
    ch_idx.sort()
    return ch_idx


# 完全随机分配通道索引
def assign_ch_idx_rand(channels, n_blocks):
    ch_idx = np.random.randint(0, channels, n_blocks)
    ch_idx.sort()
    return ch_idx


# 只分配一部分通道索引,在这一部分中完全随机
def assign_ch_idx_partial_rand(channels, n_blocks, p_add):
    parts = int(round(channels * p_add))
    ch_idx = np.random.permutation(channels)[:parts]
    ch_idx.sort()
    rand_idx = np.random.randint(0, parts, n_blocks)
    ch_idx = ch_idx[rand_idx]
    ch_idx.sort()
    return ch_idx


# 只分配一部分通道索引,在这一部分中均匀分配
def assign_ch_idx_partial_uniform(channels, n_blocks, p_add):
    parts = int(round(channels * p_add))
    ch_idx = np.random.permutation(channels)[:parts]
    times = int(np.ceil(float(n_blocks) / parts))
    ch_idx = np.tile(ch_idx, times)[:n_blocks]
    ch_idx.sort()
    return ch_idx


# 均匀分配特征图索引,所有选中的特征图索引应该尽量均匀,相邻的索引应该尽量不同
def assign_onemap_idx_uniform(arrayr, arrayc, n_blocks):
    idx_for_array_idx = np.random.permutation(arrayr * arrayc)[:n_blocks]
    if arrayr * arrayc < n_blocks:
        times = int(np.ceil(float(n_blocks) / (arrayr * arrayc)))
        idx_for_array_idx = np.tile(idx_for_array_idx, times)[:n_blocks]
    return idx_for_array_idx


# p_center是中心占用的面积比例,是行列比例的乘积
def get_center_idx(rows, cols, p_center_of_image):
    if isinstance(p_center_of_image, (list, tuple)):
        p_row, p_col = p_center_of_image
    else:
        p_row = p_col = p_center_of_image
    centerr, centerc = int(np.floor(rows * p_row)), int(np.floor(cols * p_col))
    aroundr, aroundc = int(np.ceil((rows - centerr) / 2.)), int(np.ceil((cols - centerc) / 2.))
    center_idx = []
    for r in xrange(centerr):
        one_row = np.arange(aroundc, aroundc + centerc) + (r + aroundr) * cols
        center_idx = np.concatenate([center_idx, one_row], axis=0) if len(center_idx) != 0 else one_row
    return center_idx


# 只分配特征图外围的索引(背景),分配的比例是select
def assign_onemap_idx_around(arrayr, arrayc, n_blocks, p_center_of_image=0.5):
    center_idx = get_center_idx(arrayr, arrayc, p_center_of_image)
    total_idx = np.arange(arrayr * arrayc)
    around_idx = np.delete(total_idx, center_idx)
    idx_for_array_idx = np.random.permutation(around_idx)[:n_blocks]
    if len(around_idx) < n_blocks:
        times = int(np.ceil(float(n_blocks) / len(around_idx)))
        idx_for_array_idx = np.tile(idx_for_array_idx, times)[:n_blocks]
    return idx_for_array_idx


# 只分配特征图中心的索引(目标),分配的比例是select
def assign_onemap_idx_center(arrayr, arrayc, n_blocks, p_center_of_image=0.5):
    center_idx = get_center_idx(arrayr, arrayc, p_center_of_image)
    idx_for_array_idx = np.random.permutation(center_idx)[:n_blocks]
    if len(center_idx) < n_blocks:
        times = int(np.ceil(float(n_blocks) / len(center_idx)))
        idx_for_array_idx = np.tile(idx_for_array_idx, times)[:n_blocks]
    return idx_for_array_idx


# 交替按比例分配中心和外围,且邻近的索引尽量使中心和外围相互交替
def assign_onemap_idx_alter(arrayr, arrayc, n_blocks, p_center_of_image=0.5, p_center_of_block=0.5):
    center_idx = get_center_idx(arrayr, arrayc, p_center_of_image)
    total_idx = np.arange(arrayr * arrayc)
    around_idx = np.delete(total_idx, center_idx)
    n_centers = int(round(float(n_blocks) * p_center_of_block))
    idx_for_array_idx_center = np.random.permutation(center_idx)[:n_centers]
    if len(center_idx) < n_centers:
        times = int(np.ceil(float(n_centers) / len(center_idx)))
        idx_for_array_idx_center = np.tile(idx_for_array_idx_center, times)[:n_centers]
    n_arounds = n_blocks - n_centers
    idx_for_array_idx_around = np.random.permutation(around_idx)[:n_arounds]
    if len(around_idx) < n_arounds:
        times = int(np.ceil(float(n_arounds) / len(around_idx)))
        idx_for_array_idx_around = np.tile(idx_for_array_idx_around, times)[:n_arounds]
    long_idx, short_idx = (idx_for_array_idx_center, idx_for_array_idx_around) if n_centers > n_arounds \
        else (idx_for_array_idx_around, idx_for_array_idx_center)
    n_long, n_short = len(long_idx), len(short_idx)
    step_size = int(np.floor(float(n_long) / n_short))
    tmp_idx = long_idx[:n_short * step_size].reshape((n_short, step_size))
    short_idx = short_idx[:, None]
    tmp_idx = np.concatenate([tmp_idx, short_idx], axis=1).ravel()
    idx_for_array_idx = np.append(tmp_idx, long_idx[n_short * step_size:])
    return idx_for_array_idx


# 对每张orows*ocols的图的中心或者周围加噪blockr*blockc
# 由于orows*ocols的图中目标的位置都有平移,区分中心和四周并没有意义
class MNArray(object):
    def _add_per_map(self, X, percent, block_list):
        assert X.ndim == 3
        equal_size = self.orows * self.ocols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            map_blocks = int(round(equal_size / (blockr * blockc)))
            total_blocks = self.channels * map_blocks
            arrayr = (self.orows - blockr) // 1 + 1
            arrayc = (self.ocols - blockc) // 1 + 1
            array_idx = im2col(self.oidx, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            for b in xrange(self.batches):  # 不同样本不同噪声
                ch_idx = assign_ch_idx_permap(self.channels, map_blocks)
                idx_for_array_idx = self.assign_onemap_idx(arrayr, arrayc, total_blocks, **self.map_args)
                ch_idx = np.repeat(ch_idx, blockr * blockc)
                map_idx = array_idx[idx_for_array_idx].ravel()
                X[b][ch_idx, map_idx] = 0.
        return X

    def _add_cross_ch(self, X, percent, block_list):
        assert X.ndim == 3
        equal_size = self.channels * self.orows * self.ocols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            total_blocks = int(round(equal_size / (blockr * blockc)))
            arrayr = (self.orows - blockr) // 1 + 1  # 不考虑边界
            arrayc = (self.ocols - blockc) // 1 + 1
            array_idx = im2col(self.oidx, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            for b in xrange(self.batches):  # 不同样本不同噪声
                ch_idx = assign_ch_idx_partial_uniform(self.channels, total_blocks, **self.add_args)
                idx_for_array_idx = self.assign_onemap_idx(arrayr, arrayc, total_blocks, **self.map_args)
                ch_idx = np.repeat(ch_idx, blockr * blockc)
                map_idx = array_idx[idx_for_array_idx].ravel()
                X[b][ch_idx, map_idx] = 0.
        return X

    def _add_cross_batch(self, X, percent, block_list):
        assert X.ndim == 3
        X = X.reshape((-1, self.orows * self.ocols))
        equal_size = self.channels * self.orows * self.ocols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            total_blocks = int(round(equal_size / (blockr * blockc)))
            arrayr = (self.orows - blockr) // 1 + 1  # 不考虑边界
            arrayc = (self.ocols - blockc) // 1 + 1
            array_idx = im2col(self.oidx, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            channels = self.batches * self.channels
            total_blocks *= self.batches
            ch_idx = assign_ch_idx_partial_uniform(channels, total_blocks, **self.add_args)
            idx_for_array_idx = self.assign_onemap_idx(arrayr, arrayc, total_blocks, **self.map_args)
            ch_idx = np.repeat(ch_idx, blockr * blockc)
            map_idx = array_idx[idx_for_array_idx].ravel()
            X[ch_idx, map_idx] = 0.
        return X

    def apply(self, X, percent, block_list, mode, p_add, map_mode, p_center_of_image, p_center_of_block):
        assert X.ndim == 4
        # 添加方式
        add_fn = {'permap': self._add_per_map, 'channel': self._add_cross_ch,
                  'batch': self._add_cross_batch}
        if mode not in add_fn.keys(): raise NotImplementedError
        if mode == 'permap':
            self.add_args = {}
        elif mode in ('channel', 'batch'):
            self.add_args = {'p_add': p_add,}
        # 图索引方式
        map_fn = {'uniform': assign_onemap_idx_uniform, 'around': assign_onemap_idx_around,
                  'center': assign_onemap_idx_center, 'alter': assign_onemap_idx_alter}
        if map_mode not in map_fn.keys(): raise NotImplementedError
        self.assign_onemap_idx = map_fn[map_mode]
        if map_mode == 'uniform':
            self.map_args = {}
        elif map_mode in ('around', 'center'):
            self.map_args = {'p_center_of_image': p_center_of_image,}
        elif map_mode == 'alter':
            self.map_args = {'p_center_of_image': p_center_of_image, 'p_center_of_block': p_center_of_block}
        Xshape = X.shape
        self.batches, self.channels, self.orows, self.ocols = Xshape
        self.oidx = np.arange(self.orows * self.ocols).reshape((1, 1, self.orows, self.ocols))
        X = X.reshape((self.batches, self.channels, -1))
        block_list = filter(lambda x: x[0] <= self.orows and x[1] <= self.ocols, block_list)
        X = add_fn[mode](X, percent, block_list)
        X = X.reshape(Xshape)
        return X


# p_add是保持加噪的比例,即1-p_add的比例不加噪,p_center是中心区域面积的比例
def add_mn_array(X, percent=0.5, block_list=((1, 1),), mode='channel', p_add=0.5,
                 map_mode='uniform', p_center_of_image=0.5, p_center_of_block=0.5):
    assert X.ndim == 4
    X = MNArray().apply(X, percent, block_list, mode, p_add, map_mode, p_center_of_image, p_center_of_block)
    return X


class MNArray_mch(object):
    def _add_per_map(self, X, percent, block_list):
        assert X.ndim == 4
        equal_size = self.orows * self.ocols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            map_blocks = int(round(equal_size / (blockr * blockc)))
            total_blocks = self.channels * map_blocks
            arrayr = (self.orows - blockr) // 1 + 1
            arrayc = (self.ocols - blockc) // 1 + 1
            array_idx = im2col(self.oidx, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            for b in xrange(self.batches):  # 不同样本不同噪声
                ch_idx = assign_ch_idx_permap(self.channels, map_blocks)
                idx_for_array_idx = self.assign_onemap_idx(arrayr, arrayc, total_blocks, **self.map_args)
                ch_idx = np.repeat(ch_idx, blockr * blockc)
                map_idx = array_idx[idx_for_array_idx].ravel()
                X[b][ch_idx, :, map_idx] = 0.
        return X

    def _add_cross_ch(self, X, percent, block_list):
        assert X.ndim == 4
        equal_size = self.channels * self.orows * self.ocols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            total_blocks = int(round(equal_size / (blockr * blockc)))
            arrayr = (self.orows - blockr) // 1 + 1  # 不考虑边界
            arrayc = (self.ocols - blockc) // 1 + 1
            array_idx = im2col(self.oidx, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            for b in xrange(self.batches):  # 不同样本不同噪声
                ch_idx = assign_ch_idx_partial_uniform(self.channels, total_blocks, **self.add_args)
                idx_for_array_idx = self.assign_onemap_idx(arrayr, arrayc, total_blocks, **self.map_args)
                ch_idx = np.repeat(ch_idx, blockr * blockc)
                map_idx = array_idx[idx_for_array_idx].ravel()
                X[b][ch_idx, :, map_idx] = 0.
        return X

    def _add_cross_batch(self, X, percent, block_list):
        assert X.ndim == 4
        X = X.reshape((-1, self.mch, self.orows * self.ocols))
        equal_size = self.channels * self.orows * self.ocols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            total_blocks = int(round(equal_size / (blockr * blockc)))
            arrayr = (self.orows - blockr) // 1 + 1  # 不考虑边界
            arrayc = (self.ocols - blockc) // 1 + 1
            array_idx = im2col(self.oidx, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            channels = self.batches * self.channels
            total_blocks *= self.batches
            ch_idx = assign_ch_idx_partial_uniform(channels, total_blocks, **self.add_args)
            idx_for_array_idx = self.assign_onemap_idx(arrayr, arrayc, total_blocks, **self.map_args)
            ch_idx = np.repeat(ch_idx, blockr * blockc)
            map_idx = array_idx[idx_for_array_idx].ravel()
            X[ch_idx, :, map_idx] = 0.
        return X

    def apply(self, X, percent, block_list, mode, p_add, map_mode, p_center_of_image, p_center_of_block):
        assert X.ndim == 5
        # 添加方式
        add_fn = {'permap': self._add_per_map, 'channel': self._add_cross_ch,
                  'batch': self._add_cross_batch}
        if mode not in add_fn.keys(): raise NotImplementedError
        if mode == 'permap':
            self.add_args = {}
        elif mode in ('channel', 'batch'):
            self.add_args = {'p_add': p_add,}
        # 图索引方式
        map_fn = {'uniform': assign_onemap_idx_uniform, 'around': assign_onemap_idx_around,
                  'center': assign_onemap_idx_center, 'alter': assign_onemap_idx_alter}
        if map_mode not in map_fn.keys(): raise NotImplementedError
        self.assign_onemap_idx = map_fn[map_mode]
        if map_mode == 'uniform':
            self.map_args = {}
        elif map_mode in ('around', 'center'):
            self.map_args = {'p_center_of_image': p_center_of_image,}
        elif map_mode == 'alter':
            self.map_args = {'p_center_of_image': p_center_of_image, 'p_center_of_block': p_center_of_block}
        Xshape = X.shape
        self.batches, self.channels, self.mch, self.orows, self.ocols = Xshape
        self.oidx = np.arange(self.orows * self.ocols).reshape((1, 1, self.orows, self.ocols))
        X = X.reshape((self.batches, self.channels, self.mch, -1))
        block_list = filter(lambda x: x[0] <= self.orows and x[1] <= self.ocols, block_list)
        X = add_fn[mode](X, percent, block_list)
        X = X.reshape(Xshape)
        return X


def add_mn_array_mch(X, percent=0.5, block_list=((1, 1),), mode='channel', p_add=0.5,
                     map_mode='uniform', p_center_of_image=0.5, p_center_of_block=0.5):
    assert X.ndim == 5
    X = MNArray_mch().apply(X, percent, block_list, mode, p_add, map_mode, p_center_of_image, p_center_of_block)
    return X


########################################################################################################################


# 只分配包含目标的通道索引,origin_patches中每个patch都标注了目标
def assign_ch_idx_partial_object(origin_patches, n_blocks, p_add):
    channels = origin_patches.shape[0]
    patchsum = origin_patches.sum(axis=(1, 2))
    object_idx = np.where(patchsum > 0)[0]
    if len(object_idx) == 0:  # 如果没有目标,则通道可以完全随机分配
        object_idx = np.arange(channels)
    parts = int(round(channels * p_add))
    object_idx = np.random.permutation(object_idx)[:parts]
    times = int(np.ceil(float(n_blocks) / len(object_idx)))
    ch_idx = np.tile(object_idx, times)[:n_blocks]
    ch_idx.sort()
    return ch_idx


def get_orient_idx_all(blockr, blockc, channels, origin_patches):
    im2colfn = im2col_compfn(origin_patches.shape[-2:], (blockr, blockc), 1, 0, ignore_border=True)
    patchsize = float(np.prod(origin_patches.shape[-2:]))
    orient_idx_all = []
    for ch in xrange(channels):
        patch = origin_patches[ch][None, None, :, :]
        p_object = patch.sum() / patchsize  # 目标图像占总图像的比例,每个block中也应该有此比例的目标图像
        blocks = im2colfn(patch)
        blocksum = blocks.sum(axis=1)
        orient_idx = np.where(blocksum >= blocks.shape[1] * p_object)[0]
        if len(orient_idx) == 0:  # 如果全部是背景,则全部随机选取
            orient_idx = np.arange(len(blocks))
        orient_idx_all.append(copy(orient_idx))
    return orient_idx_all


def assign_onemap_idx_orient(ch_idx, orient_idx_all):
    idx_for_array_idx = []
    for ch in ch_idx:
        orient_idx_ch = orient_idx_all[ch]  # 对于指定的一个通道,找到对应备选的索引
        rand_idx = np.random.randint(len(orient_idx_ch))  # 随机取一个索引
        idx_for_array_idx.append(copy(orient_idx_ch[rand_idx]))
    return idx_for_array_idx


# 默认目标在原图的中心的一个矩形内,对每张orows*ocols的图包含目标的区域加噪blockr*blockc
# 由于目标并不总是在中心,大小不固定且不一定是矩形
class MNArrayOrient(object):
    def _add_cross_ch(self, X, percent, block_list):
        assert X.ndim == 3
        equal_size = self.channels * self.orows * self.ocols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            total_blocks = int(round(equal_size / (blockr * blockc)))
            array_idx = im2col(self.oidx, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            orient_idx_all = get_orient_idx_all(blockr, blockc, self.channels, self.origin_patches)
            for b in xrange(self.batches):  # 不同样本不同噪声
                ch_idx = assign_ch_idx_partial_object(self.origin_patches, total_blocks, self.p_add)
                idx_for_array_idx = assign_onemap_idx_orient(ch_idx, orient_idx_all)
                ch_idx = np.repeat(ch_idx, blockr * blockc)
                map_idx = array_idx[idx_for_array_idx].ravel()
                X[b][ch_idx, map_idx] = 0.
        return X

    def apply(self, X, pad, stride, percent, block_list, p_add, p_center_of_image):
        assert X.ndim == 4
        Xshape = X.shape
        self.batches, self.channels, self.orows, self.ocols = Xshape
        filter_size = int(np.sqrt(self.channels))
        self.p_add = p_add
        # stride和pad与获取patch一致,恢复出原始图像大小
        originr = (self.orows - 1) * stride - pad * 2 + filter_size
        originc = (self.ocols - 1) * stride - pad * 2 + filter_size
        # 默认目标处于中心区域
        center_idx = get_center_idx(originr, originc, p_center_of_image)
        origin_map = np.zeros(originr * originc, dtype=int)
        origin_map[center_idx] = 1
        origin_map = origin_map.reshape((1, 1, originr, originc))
        # 目标区域用同样的方式获取patches,shape=(channels,orows,ocols)
        origin_patches = im2col(origin_map, filter_size, stride, pad, ignore_border=False)
        self.origin_patches = origin_patches.reshape((self.orows, self.ocols, -1)).transpose((2, 0, 1))
        self.oidx = np.arange(self.orows * self.ocols).reshape((1, 1, self.orows, self.ocols))
        X = X.reshape((self.batches, self.channels, -1))
        block_list = filter(lambda x: x[0] <= self.orows and x[1] <= self.ocols, block_list)
        X = self._add_cross_ch(X, percent, block_list)
        X = X.reshape(Xshape)
        return X


# p_add是保持加噪的比例,即1-p_add的比例不加噪,p_center是中心区域面积的比例
def add_mn_array_orient(X, pad, stride, percent=0.5, block_list=((1, 1),),
                        p_add=0.5, p_center_of_image=(0.5, 0.5)):
    assert X.ndim == 4
    X = MNArrayOrient().apply(X, pad, stride, percent, block_list, p_add, p_center_of_image)
    return X


class MNArrayOrient_mch(object):
    def _add_cross_ch(self, X, percent, block_list):
        assert X.ndim == 4
        equal_size = self.channels * self.orows * self.ocols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            total_blocks = int(round(equal_size / (blockr * blockc)))
            array_idx = im2col(self.oidx, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            orient_idx_all = get_orient_idx_all(blockr, blockc, self.channels, self.origin_patches)
            for b in xrange(self.batches):  # 不同样本不同噪声
                ch_idx = assign_ch_idx_partial_object(self.origin_patches, total_blocks, self.p_add)
                idx_for_array_idx = assign_onemap_idx_orient(ch_idx, orient_idx_all)
                ch_idx = np.repeat(ch_idx, blockr * blockc)
                map_idx = array_idx[idx_for_array_idx].ravel()
                X[b][ch_idx, :, map_idx] = 0.
        return X

    def apply(self, X, pad, stride, percent, block_list, p_add, p_center_of_image):
        assert X.ndim == 5
        Xshape = X.shape
        self.batches, self.channels, self.mch, self.orows, self.ocols = Xshape
        filter_size = int(np.sqrt(self.channels))
        self.p_add = p_add
        # stride和pad与获取patch一致,恢复出原始图像大小
        originr = (self.orows - 1) * stride - pad * 2 + filter_size
        originc = (self.ocols - 1) * stride - pad * 2 + filter_size
        # 默认目标处于中心区域
        center_idx = get_center_idx(originr, originc, p_center_of_image)
        origin_map = np.zeros(originr * originc, dtype=int)
        origin_map[center_idx] = 1
        origin_map = origin_map.reshape((1, 1, originr, originc))
        # 将目标区域用同样的方式获取patches
        origin_patches = im2col(origin_map, filter_size, stride, pad, ignore_border=False)
        self.origin_patches = origin_patches.reshape((self.orows, self.ocols, -1)).transpose((2, 0, 1))
        self.oidx = np.arange(self.orows * self.ocols).reshape((1, 1, self.orows, self.ocols))
        X = X.reshape((self.batches, self.channels, self.mch, -1))
        block_list = filter(lambda x: x[0] <= self.orows and x[1] <= self.ocols, block_list)
        X = self._add_cross_ch(X, percent, block_list)
        X = X.reshape(Xshape)
        return X


# p_add是保持加噪的比例,即1-p_add的比例不加噪,p_center是中心区域面积的比例
def add_mn_array_orient_mch(X, pad, stride, percent=0.5, block_list=((1, 1),),
                            p_add=0.5, p_center_of_image=(0.5, 0.5)):
    assert X.ndim == 5
    X = MNArrayOrient_mch().apply(X, pad, stride, percent, block_list, p_add, p_center_of_image)
    return X


########################################################################################################################


# 在cv2.Canny没有高斯模糊
def canny_edge_opencv(X, p_border=0.05):
    from scipy.ndimage.filters import gaussian_filter
    def smooth_with_function_and_mask(image, function, mask):
        bleed_over = function(mask.astype(float))
        masked_image = np.zeros(image.shape, image.dtype)
        masked_image[mask] = image[mask]
        smoothed_image = function(masked_image)
        output_image = smoothed_image / (bleed_over + np.finfo(float).eps)
        return output_image

    assert X.ndim == 3  # 对灰度图像和多通道图像都可以,后者需要先转化为2维灰度图像
    X = X[0, :, :] if X.shape[0] == 1 else np.max(X, axis=0)  # 采用最大法转化灰度图像
    X = X.astype(np.float)
    rows, cols = X.shape
    mask = np.zeros(X.shape, dtype=bool)
    startr, endr = int(np.round(rows * p_border)), int(np.round(rows * (1 - p_border)))
    startc, endc = int(np.round(cols * p_border)), int(np.round(cols * (1 - p_border)))
    mask[startr:endr, startc:endc] = True
    fsmooth = lambda x: gaussian_filter(x, 0.8, mode='constant')
    X = smooth_with_function_and_mask(X, fsmooth, mask)
    X -= np.min(X)  # 最大化图像的动态范围到0~255
    X *= 255. / np.max(X)
    X = np.round(X).astype(np.uint8)
    edge = cv2.Canny(X, 180, 230)
    edge /= 255
    return edge


# 先做高斯模糊,基于numpy的速度更快
def canny_edge_skimage(X, p_border=0.05):
    assert X.ndim == 3  # 对灰度图像和多通道图像都可以,后者需要先转化为2维灰度图像
    X = X[0, :, :] if X.shape[0] == 1 else np.max(X, axis=0)
    X = X.astype(np.float)
    X -= np.min(X)  # 最大化图像的动态范围到-1~1
    X /= np.max(X)
    rows, cols = X.shape
    mask = np.zeros(X.shape, dtype=bool)
    startr, endr = int(np.round(rows * p_border)), int(np.round(rows * (1 - p_border)))
    startc, endc = int(np.round(cols * p_border)), int(np.round(cols * (1 - p_border)))
    mask[startr:endr, startc:endc] = True
    edge = feature.canny(X, sigma=0.9, low_threshold=0.5, high_threshold=0.8, mask=mask)
    return edge


def get_edge_patches(originX, orows, ocols, im2colfn):
    assert originX.ndim == 3  # originX可以是单通道也可以是多通道
    edge = canny_edge_skimage(originX)
    edge = edge[None, None, :, :]  # 对于边缘图像
    edge_patches = im2colfn(edge)  # 与原来取patch一样
    edge_patches = edge_patches.reshape((orows, ocols, -1)).transpose((2, 0, 1))
    return edge_patches


def get_edge_idx_all(edge_patches, im2colfn):
    channels = edge_patches.shape[0]
    edge_idx_all = []
    patchsize = float(np.prod(edge_patches.shape[-2:]))
    for ch in xrange(channels):  # 对所有通道获取所有含有边缘的备选索引
        patch = edge_patches[ch][None, None, :, :]
        p_edge = patch.sum() / patchsize  # 边缘图像占总图像的比例,每个block中也应该有此比例的边缘图像
        blocks = im2colfn(patch)
        blocksum = blocks.sum(axis=1)
        edge_idx = np.where(blocksum >= blocks.shape[1] * p_edge)[0]  # 含有边缘的备选索引
        if len(edge_idx) == 0:  # 如果全部是背景,则全部随机选取
            edge_idx = np.arange(len(blocks))
        edge_idx_all.append(copy(edge_idx))
    return edge_idx_all


def assign_onemap_idx_edge(ch_idx, edge_idx_all):
    idx_for_array_idx = []
    for ch in ch_idx:
        edge_idx_ch = edge_idx_all[ch]  # 对于指定的一个通道,找到对应所有备选的索引
        rand_idx = np.random.randint(len(edge_idx_ch))  # 从备选中随机取一个索引
        idx_for_array_idx.append(copy(edge_idx_ch[rand_idx]))
    return idx_for_array_idx


# 先将原图中目标的边缘找到,对每张orows*ocols的图包含目标边缘的区域加噪blockr*blockc
class MNArrayEdge(object):
    def _add_cross_ch(self, X, percent, block_list):
        assert X.ndim == 3
        equal_size = self.channels * self.orows * self.ocols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            total_blocks = int(round(equal_size / (blockr * blockc)))
            array_idx = im2col(self.oidx, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            im2colfn = im2col_compfn((self.orows, self.ocols), (blockr, blockc), 1, 0, ignore_border=True)
            for b in xrange(self.batches):  # 对每个样本的原始图像计算canny边缘
                edge_patches = get_edge_patches(self.originX[b], self.orows, self.ocols, self.im2colfn)
                edge_idx_all = get_edge_idx_all(edge_patches, im2colfn)
                ch_idx = assign_ch_idx_partial_object(edge_patches, total_blocks, self.p_add)
                idx_for_array_idx = assign_onemap_idx_edge(ch_idx, edge_idx_all)
                ch_idx = np.repeat(ch_idx, blockr * blockc)
                map_idx = array_idx[idx_for_array_idx].ravel()
                X[b][ch_idx, map_idx] = 0.
        return X

    def apply(self, X, originX, pad, stride, percent, block_list, p_add):
        assert X.ndim == 4 and originX.ndim == 4
        Xshape = X.shape
        self.batches, self.channels, self.orows, self.ocols = Xshape
        self.originX = originX
        filter_size = int(np.sqrt(self.channels))
        self.p_add = p_add
        self.oidx = np.arange(self.orows * self.ocols).reshape((1, 1, self.orows, self.ocols))
        self.im2colfn = im2col_compfn(originX.shape[-2:], filter_size, stride, pad, ignore_border=False)
        X = X.reshape((self.batches, self.channels, -1))
        block_list = filter(lambda x: x[0] <= self.orows and x[1] <= self.ocols, block_list)
        X = self._add_cross_ch(X, percent, block_list)
        X = X.reshape(Xshape)
        return X


# p_add是保持加噪的比例,即1-p_add的比例不加噪
def add_mn_array_edge(X, originX, pad, stride, percent=0.5, block_list=((1, 1),), p_add=0.5):
    assert X.ndim == 4
    X = MNArrayEdge().apply(X, originX, pad, stride, percent, block_list, p_add)
    return X


class MNArrayEdge_mch(object):
    def _add_cross_ch(self, X, percent, block_list):
        assert X.ndim == 4
        equal_size = self.channels * self.orows * self.ocols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            total_blocks = int(round(equal_size / (blockr * blockc)))
            array_idx = im2col(self.oidx, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            im2colfn = im2col_compfn((self.orows, self.ocols), (blockr, blockc), 1, 0, ignore_border=True)
            for b in xrange(self.batches):  # 对每个样本的原始图像计算canny边缘
                edge_patches = get_edge_patches(self.originX[b], self.orows, self.ocols, self.im2colfn)
                edge_idx_all = get_edge_idx_all(edge_patches, im2colfn)
                ch_idx = assign_ch_idx_partial_object(edge_patches, total_blocks, self.p_add)
                idx_for_array_idx = assign_onemap_idx_edge(ch_idx, edge_idx_all)
                ch_idx = np.repeat(ch_idx, blockr * blockc)
                map_idx = array_idx[idx_for_array_idx].ravel()
                X[b][ch_idx, :, map_idx] = 0.
        return X

    def apply(self, X, originX, pad, stride, percent, block_list, p_add):
        assert X.ndim == 5 and originX.ndim == 4
        Xshape = X.shape
        self.batches, self.channels, self.mch, self.orows, self.ocols = Xshape
        self.originX = originX
        filter_size = int(np.sqrt(self.channels))
        self.p_add = p_add
        self.oidx = np.arange(self.orows * self.ocols).reshape((1, 1, self.orows, self.ocols))
        self.im2colfn = im2col_compfn(originX.shape[-2:], filter_size, stride, pad, ignore_border=False)
        X = X.reshape((self.batches, self.channels, self.mch, -1))
        block_list = filter(lambda x: x[0] <= self.orows and x[1] <= self.ocols, block_list)
        X = self._add_cross_ch(X, percent, block_list)
        X = X.reshape(Xshape)
        return X


# p_add是保持加噪的比例,即1-p_add的比例不加噪
def add_mn_array_edge_mch(X, originX, pad, stride, percent=0.5, block_list=((1, 1),), p_add=0.5):
    assert X.ndim == 5
    X = MNArrayEdge_mch().apply(X, originX, pad, stride, percent, block_list, p_add)
    return X
