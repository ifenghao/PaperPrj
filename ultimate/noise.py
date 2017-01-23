# coding:utf-8
import numpy as np
from copy import copy
import cv2
from skimage import feature
from skimage.morphology import binary_closing
from util import *

__all__ = ['add_noise_decomp']

splits = 2


# 统一X的输入维度(batches,orows,ocols,filter_size,filter_size)
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
    batchSize = size / splits
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
    def __init__(self, mode, p_add, map_mode, p_center_of_image, p_center_of_block):
        # 添加方式
        add_fn = {'permap': self._add_per_map, 'channel': self._add_cross_ch,
                  'batch': self._add_cross_batch}
        if mode not in add_fn.keys(): raise NotImplementedError
        self.add_mn = add_fn[mode]
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

    def _add_per_map(self, X, percent, block_list):
        assert X.ndim == 4
        Xshape = X.shape
        batches, channels, rows, cols = Xshape
        X = X.reshape((batches, channels, -1))
        equal_size = rows * cols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            map_blocks = int(round(equal_size / (blockr * blockc)))
            total_blocks = channels * map_blocks
            arrayr = (rows - blockr) // 1 + 1
            arrayc = (cols - blockc) // 1 + 1
            array_idx = im2col(self.idx_map, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            for b in xrange(batches):  # 不同样本不同噪声
                ch_idx = assign_ch_idx_permap(channels, map_blocks)
                idx_for_array_idx = self.assign_onemap_idx(arrayr, arrayc, total_blocks, **self.map_args)
                ch_idx = np.repeat(ch_idx, blockr * blockc)
                map_idx = array_idx[idx_for_array_idx].ravel()
                X[b][ch_idx, map_idx] = 0.
        X = X.reshape(Xshape)
        return X

    def _add_cross_ch(self, X, percent, block_list):
        assert X.ndim == 4
        Xshape = X.shape
        batches, channels, rows, cols = Xshape
        X = X.reshape((batches, channels, -1))
        equal_size = channels * rows * cols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            total_blocks = int(round(equal_size / (blockr * blockc)))
            arrayr = (rows - blockr) // 1 + 1  # 不考虑边界
            arrayc = (cols - blockc) // 1 + 1
            array_idx = im2col(self.idx_map, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            for b in xrange(batches):  # 不同样本不同噪声
                ch_idx = assign_ch_idx_partial_uniform(channels, total_blocks, **self.add_args)
                idx_for_array_idx = self.assign_onemap_idx(arrayr, arrayc, total_blocks, **self.map_args)
                ch_idx = np.repeat(ch_idx, blockr * blockc)
                map_idx = array_idx[idx_for_array_idx].ravel()
                X[b][ch_idx, map_idx] = 0.
        X = X.reshape(Xshape)
        return X

    def _add_cross_batch(self, X, percent, block_list):
        assert X.ndim == 4
        Xshape = X.shape
        batches, channels, rows, cols = Xshape
        X = X.reshape((batches * channels, -1))
        equal_size = channels * rows * cols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            total_blocks = int(round(equal_size / (blockr * blockc)))
            arrayr = (rows - blockr) // 1 + 1  # 不考虑边界
            arrayc = (cols - blockc) // 1 + 1
            array_idx = im2col(self.idx_map, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            channels = batches * channels
            total_blocks *= batches
            ch_idx = assign_ch_idx_partial_uniform(channels, total_blocks, **self.add_args)
            idx_for_array_idx = self.assign_onemap_idx(arrayr, arrayc, total_blocks, **self.map_args)
            ch_idx = np.repeat(ch_idx, blockr * blockc)
            map_idx = array_idx[idx_for_array_idx].ravel()
            X[ch_idx, map_idx] = 0.
        X = X.reshape(Xshape)
        return X

    def apply_for_omap(self, X, percent, block_list):
        assert X.ndim == 5
        Xshape = X.shape
        batches, orows, ocols, frows, fcols = Xshape
        self.idx_map = np.arange(orows * ocols).reshape((1, 1, orows, ocols))
        X = X.reshape((batches, orows, ocols, -1)).transpose((0, 3, 1, 2))
        block_list = filter(lambda x: x[0] <= orows and x[1] <= ocols, block_list)
        X = self.add_mn(X, percent, block_list)
        X = X.transpose((0, 2, 3, 1)).reshape(Xshape)
        return X

    def apply_for_patch(self, X, percent, block_list):
        assert X.ndim == 5
        Xshape = X.shape
        batches, orows, ocols, frows, fcols = Xshape
        self.idx_map = np.arange(frows * fcols).reshape((1, 1, frows, fcols))
        X = X.reshape((batches, -1, frows, fcols))
        block_list = filter(lambda x: x[0] <= frows and x[1] <= fcols, block_list)
        X = self.add_mn(X, percent, block_list)
        X = X.reshape(Xshape)
        return X

    def apply_for_cccp(self, X, percent, block_list):
        assert X.ndim == 4
        Xshape = X.shape
        batches, rows, cols, channels = Xshape
        self.idx_map = np.arange(rows * cols).reshape((1, 1, rows, cols))
        X = X.transpose((0, 3, 1, 2))
        block_list = filter(lambda x: x[0] <= rows and x[1] <= cols, block_list)
        X = self.add_mn(X, percent, block_list)
        X = X.transpose((0, 2, 3, 1))
        return X


# p_add是保持加噪的比例,即1-p_add的比例不加噪,p_center是中心区域面积的比例
def add_mn_array(X, percent=0.5, block_list=((1, 1),), mode='channel', p_add=0.5,
                 map_mode='uniform', p_center_of_image=0.5, p_center_of_block=0.5, apply_mode='omap'):
    mna = MNArray(mode, p_add, map_mode, p_center_of_image, p_center_of_block)
    apply_fn = {'omap': mna.apply_for_omap, 'patch': mna.apply_for_patch, 'cccp': mna.apply_for_cccp}
    if apply_mode not in apply_fn.keys(): raise NotImplementedError
    X = apply_fn[apply_mode](X, percent, block_list)
    return X


class MNArray_mch(object):
    def __init__(self, mode, p_add, map_mode, p_center_of_image, p_center_of_block):
        # 添加方式
        add_fn = {'permap': self._add_per_map, 'channel': self._add_cross_ch,
                  'batch': self._add_cross_batch}
        if mode not in add_fn.keys(): raise NotImplementedError
        self.add_mn = add_fn[mode]
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

    def _add_per_map(self, X, percent, block_list):
        assert X.ndim == 5
        Xshape = X.shape
        batches, channels, mch, rows, cols = Xshape
        X = X.reshape((batches, channels, mch, -1))
        equal_size = rows * cols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            map_blocks = int(round(equal_size / (blockr * blockc)))
            total_blocks = channels * map_blocks
            arrayr = (rows - blockr) // 1 + 1
            arrayc = (cols - blockc) // 1 + 1
            array_idx = im2col(self.idx_map, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            for b in xrange(batches):  # 不同样本不同噪声
                ch_idx = assign_ch_idx_permap(channels, map_blocks)
                idx_for_array_idx = self.assign_onemap_idx(arrayr, arrayc, total_blocks, **self.map_args)
                ch_idx = np.repeat(ch_idx, blockr * blockc)
                map_idx = array_idx[idx_for_array_idx].ravel()
                X[b][ch_idx, :, map_idx] = 0.
        X = X.reshape(Xshape)
        return X

    def _add_cross_ch(self, X, percent, block_list):
        assert X.ndim == 5
        Xshape = X.shape
        batches, channels, mch, rows, cols = Xshape
        X = X.reshape((batches, channels, mch, -1))
        equal_size = channels * rows * cols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            total_blocks = int(round(equal_size / (blockr * blockc)))
            arrayr = (rows - blockr) // 1 + 1  # 不考虑边界
            arrayc = (cols - blockc) // 1 + 1
            array_idx = im2col(self.idx_map, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            for b in xrange(batches):  # 不同样本不同噪声
                ch_idx = assign_ch_idx_partial_uniform(channels, total_blocks, **self.add_args)
                idx_for_array_idx = self.assign_onemap_idx(arrayr, arrayc, total_blocks, **self.map_args)
                ch_idx = np.repeat(ch_idx, blockr * blockc)
                map_idx = array_idx[idx_for_array_idx].ravel()
                X[b][ch_idx, :, map_idx] = 0.
        X = X.reshape(Xshape)
        return X

    def _add_cross_batch(self, X, percent, block_list):
        assert X.ndim == 5
        Xshape = X.shape
        batches, channels, mch, rows, cols = Xshape
        X = X.reshape((batches * channels, mch, -1))
        equal_size = channels * rows * cols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            total_blocks = int(round(equal_size / (blockr * blockc)))
            arrayr = (rows - blockr) // 1 + 1  # 不考虑边界
            arrayc = (cols - blockc) // 1 + 1
            array_idx = im2col(self.idx_map, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            channels = batches * channels
            total_blocks *= batches
            ch_idx = assign_ch_idx_partial_uniform(channels, total_blocks, **self.add_args)
            idx_for_array_idx = self.assign_onemap_idx(arrayr, arrayc, total_blocks, **self.map_args)
            ch_idx = np.repeat(ch_idx, blockr * blockc)
            map_idx = array_idx[idx_for_array_idx].ravel()
            X[ch_idx, :, map_idx] = 0.
        X = X.reshape(Xshape)
        return X

    def apply_for_omap(self, X, percent, block_list):
        assert X.ndim == 6
        Xshape = X.shape
        batches, orows, ocols, mch, frows, fcols = Xshape
        self.idx_map = np.arange(orows * ocols).reshape((1, 1, orows, ocols))
        X = X.reshape((batches, orows, ocols, mch, -1)).transpose((0, 4, 3, 1, 2))
        block_list = filter(lambda x: x[0] <= orows and x[1] <= ocols, block_list)
        X = self.add_mn(X, percent, block_list)
        X = X.transpose((0, 3, 4, 2, 1)).reshape(Xshape)
        return X

    def apply_for_patch(self, X, percent, block_list):
        assert X.ndim == 6
        Xshape = X.shape
        batches, orows, ocols, mch, frows, fcols = Xshape
        self.idx_map = np.arange(frows * fcols).reshape((1, 1, frows, fcols))
        X = X.reshape((batches, -1, mch, frows, fcols))
        block_list = filter(lambda x: x[0] <= frows and x[1] <= fcols, block_list)
        X = self.add_mn(X, percent, block_list)
        X = X.reshape(Xshape)
        return X


# p_add是保持加噪的比例,即1-p_add的比例不加噪,p_center是中心区域面积的比例
def add_mn_array_mch(X, percent=0.5, block_list=((1, 1),), mode='channel', p_add=0.5,
                     map_mode='uniform', p_center_of_image=0.5, p_center_of_block=0.5, apply_mode='omap'):
    mna = MNArray_mch(mode, p_add, map_mode, p_center_of_image, p_center_of_block)
    apply_fn = {'omap': mna.apply_for_omap, 'patch': mna.apply_for_patch}
    if apply_mode not in apply_fn.keys(): raise NotImplementedError
    X = apply_fn[apply_mode](X, percent, block_list)
    return X


########################################################################################################################


# 只分配包含目标的通道索引,origin_patches中每个patch都标注了目标
def assign_ch_idx_partial_justfor_object(origin_maps, n_blocks, p_add):
    assert origin_maps.ndim == 3
    channels = origin_maps.shape[0]
    patchsum = origin_maps.sum(axis=(1, 2))
    object_idx = np.where(patchsum > 0)[0]
    if len(object_idx) == 0:  # 如果没有目标,则通道可以完全随机分配
        object_idx = np.arange(channels)
    parts = int(round(channels * p_add))
    object_idx = np.random.permutation(object_idx)[:parts]
    times = int(np.ceil(float(n_blocks) / len(object_idx)))
    ch_idx = np.tile(object_idx, times)[:n_blocks]
    ch_idx.sort()
    return ch_idx


# 默认目标处于中心区域,目标区域为1,背景区域为0
def get_origin_patches(orows, ocols, frows, fcols, pad, stride, p_center_of_image):
    # stride和pad与获取patch一致,恢复出原始图像大小
    originr = (orows - 1) * stride - pad * 2 + frows
    originc = (ocols - 1) * stride - pad * 2 + fcols
    center_idx = get_center_idx(originr, originc, p_center_of_image)
    origin_map = np.zeros(originr * originc, dtype=int)
    origin_map[center_idx] = 1
    origin_map = origin_map.reshape((1, 1, originr, originc))
    # 目标区域用同样的方式获取(orows*ocols, frows*fcols)的patches
    origin_patches = im2col(origin_map, (frows, fcols), stride, pad, ignore_border=False)
    return origin_patches


# 获取每个通道patch的所有的block中包含目标的索引列表
def get_orient_idx_all(blockr, blockc, origin_patches):
    im2colfn = im2col_compfn(origin_patches.shape[-2:], (blockr, blockc), 1, 0, ignore_border=True)
    patchsize = float(np.prod(origin_patches.shape[-2:]))
    channels = origin_patches.shape[0]
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


# 默认目标处于中心区域,目标区域为1,背景区域为0
def get_origin_blocks_cccp(rows, cols, blockr, blockc, p_center_of_image):
    center_idx = get_center_idx(rows, cols, p_center_of_image)
    origin_map = np.zeros(rows * cols, dtype=int)
    origin_map[center_idx] = 1
    origin_map = origin_map.reshape((1, 1, rows, cols))
    origin_blocks = im2col(origin_map, (blockr, blockc), 1, 0, ignore_border=True)
    return origin_blocks


# 获取原始图像的所有blocks中包含目标的索引列表
def get_orient_idx_all_cccp(rows, cols, blockr, blockc, p_center_of_image):
    origin_blocks = get_origin_blocks_cccp(rows, cols, blockr, blockc, p_center_of_image)
    blocksum = origin_blocks.sum(axis=1)
    orient_idx_all = np.where(blocksum > 0)[0]
    if len(orient_idx_all) == 0:  # 如果全部是背景,则全部随机选取
        orient_idx_all = np.arange(len(origin_blocks))
    return orient_idx_all


# 对于每个通道,可选择的包含目标的索引列表都是一样的
def assign_onemap_idx_orient_cccp(n_blocks, orient_idx_all):
    idx_for_array_idx = np.random.permutation(len(orient_idx_all))[:n_blocks]
    if len(idx_for_array_idx) < n_blocks:
        times = int(np.ceil(float(n_blocks) / len(idx_for_array_idx)))
        idx_for_array_idx = np.tile(idx_for_array_idx, times)[:n_blocks]
    return idx_for_array_idx


# 默认目标在原图的中心的一个矩形内,对每张orows*ocols的图包含目标的区域加噪blockr*blockc
# 由于目标并不总是在中心,大小不固定且不一定是矩形
class MNArrayOrient(object):
    def _add_cross_ch_ae(self, X, percent, block_list):
        assert X.ndim == 4
        Xshape = X.shape
        batches, channels, rows, cols = Xshape
        X = X.reshape((batches, channels, -1))
        equal_size = channels * rows * cols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            total_blocks = int(round(equal_size / (blockr * blockc)))
            array_idx = im2col(self.idx_map, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            orient_idx_all = get_orient_idx_all(blockr, blockc, self.origin_patches)
            for b in xrange(batches):  # 不同样本不同噪声
                ch_idx = assign_ch_idx_partial_justfor_object(self.origin_patches, total_blocks, self.p_add)
                idx_for_array_idx = assign_onemap_idx_orient(ch_idx, orient_idx_all)
                ch_idx = np.repeat(ch_idx, blockr * blockc)
                map_idx = array_idx[idx_for_array_idx].ravel()
                X[b][ch_idx, map_idx] = 0.
        X = X.reshape(Xshape)
        return X

    def _add_cross_ch_cccp(self, X, percent, block_list):
        assert X.ndim == 4
        Xshape = X.shape
        batches, channels, rows, cols = Xshape
        X = X.reshape((batches, channels, -1))
        equal_size = channels * rows * cols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            total_blocks = int(round(equal_size / (blockr * blockc)))
            array_idx = im2col(self.idx_map, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            orient_idx_all = get_orient_idx_all_cccp(rows, cols, blockr, blockc, self.p_center_of_image)
            for b in xrange(batches):  # 不同样本不同噪声
                ch_idx = assign_ch_idx_partial_uniform(channels, total_blocks, self.p_add)  # 目标位置默认都为中心,随机分配
                idx_for_array_idx = assign_onemap_idx_orient_cccp(total_blocks, orient_idx_all)
                ch_idx = np.repeat(ch_idx, blockr * blockc)
                map_idx = array_idx[idx_for_array_idx].ravel()
                X[b][ch_idx, map_idx] = 0.
        X = X.reshape(Xshape)
        return X

    def apply_for_omap(self, X, pad, stride, percent, block_list, p_add, p_center_of_image):
        assert X.ndim == 5
        Xshape = X.shape
        batches, orows, ocols, frows, fcols = Xshape
        self.p_add = p_add
        origin_patches = get_origin_patches(orows, ocols, frows, fcols, pad, stride, p_center_of_image)
        self.origin_patches = origin_patches.reshape((orows, ocols, -1)).transpose((2, 0, 1))
        self.idx_map = np.arange(orows * ocols).reshape((1, 1, orows, ocols))
        X = X.reshape((batches, orows, ocols, -1)).transpose((0, 3, 1, 2))
        block_list = filter(lambda x: x[0] <= orows and x[1] <= ocols, block_list)
        X = self._add_cross_ch_ae(X, percent, block_list)
        X = X.transpose((0, 2, 3, 1)).reshape(Xshape)
        return X

    def apply_for_patch(self, X, pad, stride, percent, block_list, p_add, p_center_of_image):
        assert X.ndim == 5
        Xshape = X.shape
        batches, orows, ocols, frows, fcols = Xshape
        self.p_add = p_add
        origin_patches = get_origin_patches(orows, ocols, frows, fcols, pad, stride, p_center_of_image)
        self.origin_patches = origin_patches.reshape((-1, frows, fcols))
        self.idx_map = np.arange(frows * fcols).reshape((1, 1, frows, fcols))
        X = X.reshape((batches, -1, frows, fcols))
        block_list = filter(lambda x: x[0] <= frows and x[1] <= fcols, block_list)
        X = self._add_cross_ch_ae(X, percent, block_list)
        X = X.reshape(Xshape)
        return X

    def apply_for_cccp(self, X, pad, stride, percent, block_list, p_add, p_center_of_image):
        assert X.ndim == 4
        assert pad is None and stride is None
        Xshape = X.shape
        batches, rows, cols, channels = Xshape
        self.p_add = p_add
        self.p_center_of_image = p_center_of_image
        self.idx_map = np.arange(rows * cols).reshape((1, 1, rows, cols))
        X = X.transpose((0, 3, 1, 2))
        block_list = filter(lambda x: x[0] <= rows and x[1] <= cols, block_list)
        X = self._add_cross_ch_cccp(X, percent, block_list)
        X = X.transpose((0, 2, 3, 1))
        return X


# p_add是保持加噪的比例,即1-p_add的比例不加噪,p_center是中心区域面积的比例
def add_mn_array_orient(X, pad, stride, percent=0.5, block_list=((1, 1),),
                        p_add=0.5, p_center_of_image=(0.5, 0.5), apply_mode='omap'):
    mnao = MNArrayOrient()
    apply_fn = {'omap': mnao.apply_for_omap, 'patch': mnao.apply_for_patch, 'cccp': mnao.apply_for_cccp}
    if apply_mode not in apply_fn.keys(): raise NotImplementedError
    if apply_mode == 'cccp': pad = stride = None
    X = apply_fn[apply_mode](X, pad, stride, percent, block_list, p_add, p_center_of_image)
    return X


class MNArrayOrient_mch(object):
    def _add_cross_ch_ae(self, X, percent, block_list):
        assert X.ndim == 5
        Xshape = X.shape
        batches, channels, mch, rows, cols = Xshape
        X = X.reshape((batches, channels, mch, -1))
        equal_size = channels * rows * cols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            total_blocks = int(round(equal_size / (blockr * blockc)))
            array_idx = im2col(self.idx_map, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            orient_idx_all = get_orient_idx_all(blockr, blockc, self.origin_patches)
            for b in xrange(batches):  # 不同样本不同噪声
                ch_idx = assign_ch_idx_partial_justfor_object(self.origin_patches, total_blocks, self.p_add)
                idx_for_array_idx = assign_onemap_idx_orient(ch_idx, orient_idx_all)
                ch_idx = np.repeat(ch_idx, blockr * blockc)
                map_idx = array_idx[idx_for_array_idx].ravel()
                X[b][ch_idx, :, map_idx] = 0.
        X = X.reshape(Xshape)
        return X

    def apply_for_omap(self, X, pad, stride, percent, block_list, p_add, p_center_of_image):
        assert X.ndim == 6
        Xshape = X.shape
        batches, orows, ocols, mch, frows, fcols = Xshape
        self.p_add = p_add
        origin_patches = get_origin_patches(orows, ocols, frows, fcols, pad, stride, p_center_of_image)
        self.origin_patches = origin_patches.reshape((orows, ocols, -1)).transpose((2, 0, 1))
        self.idx_map = np.arange(orows * ocols).reshape((1, 1, orows, ocols))
        X = X.reshape((batches, orows, ocols, mch, -1)).transpose((0, 4, 3, 1, 2))
        block_list = filter(lambda x: x[0] <= orows and x[1] <= ocols, block_list)
        X = self._add_cross_ch_ae(X, percent, block_list)
        X = X.transpose((0, 3, 4, 2, 1)).reshape(Xshape)
        return X

    def apply_for_patch(self, X, pad, stride, percent, block_list, p_add, p_center_of_image):
        assert X.ndim == 6
        Xshape = X.shape
        batches, orows, ocols, mch, frows, fcols = Xshape
        self.p_add = p_add
        origin_patches = get_origin_patches(orows, ocols, frows, fcols, pad, stride, p_center_of_image)
        self.origin_patches = origin_patches.reshape((-1, frows, fcols))
        self.idx_map = np.arange(frows * fcols).reshape((1, 1, frows, fcols))
        X = X.reshape((batches, -1, mch, frows, fcols))
        block_list = filter(lambda x: x[0] <= frows and x[1] <= fcols, block_list)
        X = self._add_cross_ch_ae(X, percent, block_list)
        X = X.reshape(Xshape)
        return X


# p_add是保持加噪的比例,即1-p_add的比例不加噪,p_center是中心区域面积的比例
def add_mn_array_orient_mch(X, pad, stride, percent=0.5, block_list=((1, 1),),
                            p_add=0.5, p_center_of_image=(0.5, 0.5), apply_mode='omap'):
    mnao = MNArrayOrient_mch()
    apply_fn = {'omap': mnao.apply_for_omap, 'patch': mnao.apply_for_patch}
    if apply_mode not in apply_fn.keys(): raise NotImplementedError
    X = apply_fn[apply_mode](X, pad, stride, percent, block_list, p_add, p_center_of_image)
    return X


########################################################################################################################


# 在cv2.Canny没有高斯模糊
def canny_edge_opencv(X, border=0.05, sigma=1., lth=0.5, hth=0.8):
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
    startr, endr = int(np.round(rows * border)), int(np.round(rows * (1 - border)))
    startc, endc = int(np.round(cols * border)), int(np.round(cols * (1 - border)))
    mask[startr:endr, startc:endc] = True
    fsmooth = lambda x: gaussian_filter(x, sigma, mode='constant')
    X = smooth_with_function_and_mask(X, fsmooth, mask)
    X -= np.min(X)  # 最大化图像的动态范围到0~255
    X *= 255. / np.max(X)
    X = np.round(X).astype(np.uint8)
    lth, hth = int(255 * lth), int(255 * hth)
    edge = cv2.Canny(X, threshold1=lth, threshold2=hth)
    edge /= 255
    edge = binary_closing(edge)  # 填充空白空洞
    return edge


# 先做高斯模糊,基于numpy的速度更快
def canny_edge_skimage(X, border=0.05, sigma=1., lth=0.5, hth=0.8):
    assert X.ndim == 3  # 对灰度图像和多通道图像都可以,后者需要先转化为2维灰度图像
    X = X[0, :, :] if X.shape[0] == 1 else np.max(X, axis=0)
    X = X.astype(np.float)
    X -= np.min(X)  # 最大化图像的动态范围到-1~1
    X /= np.max(X)
    rows, cols = X.shape
    mask = np.zeros(X.shape, dtype=bool)
    startr, endr = int(np.round(rows * border)), int(np.round(rows * (1 - border)))
    startc, endc = int(np.round(cols * border)), int(np.round(cols * (1 - border)))
    mask[startr:endr, startc:endc] = True
    edge = feature.canny(X, sigma=sigma, low_threshold=lth, high_threshold=hth, mask=mask)
    edge = binary_closing(edge)  # 填充空白空洞
    return edge


# 直接对图像二值化提取目标位置,只对MNIST使用
def binary_object(X):
    assert X.ndim == 3  # 对灰度图像和多通道图像都可以,后者需要先转化为2维灰度图像
    X = X[0, :, :] if X.shape[0] == 1 else np.max(X, axis=0)
    X = X.astype(np.float)
    X -= np.min(X)  # 最大化图像的动态范围到-1~1
    X *= 255. / np.max(X)
    X = X.astype(np.uint8)
    X = X > 0
    return X


# originX是一个原始图像
# im2colfn要将原始图像经过fsize的滤波器得到(orows*ocols,fsize*fsize)的patch(与对oneChannel取的patch相同)
def get_edge_patches(originX, edge_args, im2colfn):
    assert originX.ndim == 3  # originX可以是单通道也可以是多通道
    edge = canny_edge_skimage(originX, **edge_args)
    edge = edge[None, None, :, :]  # 对于边缘图像
    edge_patches = im2colfn(edge)
    return edge_patches


def get_binary_patches(originX, im2colfn):
    assert originX.ndim == 3
    binary = binary_object(originX)
    binary = binary[None, None, :, :]
    binary_patches = im2colfn(binary)
    return binary_patches


# 获取每个通道patch的所有的block中包含目标的索引列表
# im2colfn要将patch经过(blockr,blockc)的block得到(arrayr*arrayc,blockr*blockc)的blocks
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


# 获取每个通道的边缘图像
def get_edges_allchannels_cccp(originX, edge_args):
    assert originX.ndim == 3
    channels = originX.shape[0]
    edges = []
    for ch in xrange(channels):  # 每个通道寻找边缘图像
        edge = canny_edge_skimage(originX[[ch]], **edge_args)
        edges.append(copy(edge))
    return np.array(edges)


def get_binary_allchannels_cccp(originX):
    assert originX.ndim == 3
    channels = originX.shape[0]
    binaries = []
    for ch in xrange(channels):
        binary = binary_object(originX[[ch]])
        binaries.append(copy(binary))
    return np.array(binaries)


# 先将原图中目标的边缘找到,对每张orows*ocols的图包含目标边缘的区域加噪blockr*blockc
class MNArrayEdge(object):
    def _add_cross_ch_ae(self, X, percent, block_list):
        assert X.ndim == 4
        Xshape = X.shape
        batches, channels, rows, cols = Xshape
        X = X.reshape((batches, channels, -1))
        equal_size = channels * rows * cols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            total_blocks = int(round(equal_size / (blockr * blockc)))
            array_idx = im2col(self.idx_map, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            im2colfn = im2col_compfn((rows, cols), (blockr, blockc), 1, 0, ignore_border=True)
            for b in xrange(batches):  # 对每个样本的原始图像计算canny边缘
                edge_patches = get_edge_patches(self.originX[b], self.edge_args, self.im2colfn) \
                    if self.edge_or_binary else get_binary_patches(self.originX[b], self.im2colfn)
                edge_patches = edge_patches.reshape((rows, cols, -1)).transpose((2, 0, 1)) \
                    if self.apply_mode == 'omap' else edge_patches.reshape((-1, rows, cols))
                edge_idx_all = get_edge_idx_all(edge_patches, im2colfn)
                ch_idx = assign_ch_idx_partial_justfor_object(edge_patches, total_blocks, self.p_add)
                idx_for_array_idx = assign_onemap_idx_edge(ch_idx, edge_idx_all)
                ch_idx = np.repeat(ch_idx, blockr * blockc)
                map_idx = array_idx[idx_for_array_idx].ravel()
                X[b][ch_idx, map_idx] = 0.
        X = X.reshape(Xshape)
        return X

    def _add_cross_ch_cccp(self, X, percent, block_list):
        assert X.ndim == 4
        Xshape = X.shape
        batches, channels, rows, cols = Xshape
        X = X.reshape((batches, channels, -1))
        equal_size = channels * rows * cols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            total_blocks = int(round(equal_size / (blockr * blockc)))
            array_idx = im2col(self.idx_map, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            im2colfn = im2col_compfn((rows, cols), (blockr, blockc), 1, 0, ignore_border=True)
            for b in xrange(batches):  # 不同样本不同噪声
                edges = get_edges_allchannels_cccp(self.originX[b], self.edge_args) \
                    if self.edge_or_binary else get_binary_allchannels_cccp(self.originX[b])
                edge_idx_all = get_edge_idx_all(edges, im2colfn)
                ch_idx = assign_ch_idx_partial_justfor_object(edges, total_blocks, self.p_add)
                idx_for_array_idx = assign_onemap_idx_edge(ch_idx, edge_idx_all)
                ch_idx = np.repeat(ch_idx, blockr * blockc)
                map_idx = array_idx[idx_for_array_idx].ravel()
                X[b][ch_idx, map_idx] = 0.
        X = X.reshape(Xshape)
        return X

    def apply_for_omap(self, X, originX, pad, stride, percent, block_list, p_add, edge_args, edge_or_binary):
        assert X.ndim == 5 and originX.ndim == 4
        Xshape = X.shape
        batches, orows, ocols, frows, fcols = Xshape
        self.originX = originX
        self.p_add = p_add
        self.edge_args = edge_args
        self.edge_or_binary = edge_or_binary
        self.idx_map = np.arange(orows * ocols).reshape((1, 1, orows, ocols))
        self.im2colfn = im2col_compfn(originX.shape[-2:], (frows, fcols), stride, pad, ignore_border=False)
        self.apply_mode = 'omap'
        X = X.reshape((batches, orows, ocols, -1)).transpose((0, 3, 1, 2))
        block_list = filter(lambda x: x[0] <= orows and x[1] <= ocols, block_list)
        X = self._add_cross_ch_ae(X, percent, block_list)
        X = X.transpose((0, 2, 3, 1)).reshape(Xshape)
        return X

    def apply_for_patch(self, X, originX, pad, stride, percent, block_list, p_add, edge_args, edge_or_binary):
        assert X.ndim == 5 and originX.ndim == 4
        Xshape = X.shape
        batches, orows, ocols, frows, fcols = Xshape
        self.originX = originX
        self.p_add = p_add
        self.edge_args = edge_args
        self.edge_or_binary = edge_or_binary
        self.idx_map = np.arange(frows * fcols).reshape((1, 1, frows, fcols))
        self.im2colfn = im2col_compfn(originX.shape[-2:], (frows, fcols), stride, pad, ignore_border=False)
        self.apply_mode = 'patch'
        X = X.reshape((batches, -1, frows, fcols))
        block_list = filter(lambda x: x[0] <= frows and x[1] <= fcols, block_list)
        X = self._add_cross_ch_ae(X, percent, block_list)
        X = X.reshape(Xshape)
        return X

    def apply_for_cccp(self, X, originX, pad, stride, percent, block_list, p_add, edge_args, edge_or_binary):
        assert X.ndim == 4 and originX.ndim == 4
        assert pad is None and stride is None
        Xshape = X.shape
        batches, rows, cols, channels = Xshape
        self.originX = originX
        self.p_add = p_add
        self.edge_args = edge_args
        self.edge_or_binary = edge_or_binary
        self.idx_map = np.arange(rows * cols).reshape((1, 1, rows, cols))
        X = X.transpose((0, 3, 1, 2))
        block_list = filter(lambda x: x[0] <= rows and x[1] <= cols, block_list)
        X = self._add_cross_ch_cccp(X, percent, block_list)
        X = X.transpose((0, 2, 3, 1))
        return X


# p_add是保持加噪的比例,即1-p_add的比例不加噪
def add_mn_array_edge(X, originX, pad, stride, percent=0.5, block_list=((1, 1),),
                      p_add=0.5, edge_args=None, apply_mode='omap', edge_or_binary=True):
    mnae = MNArrayEdge()
    apply_fn = {'omap': mnae.apply_for_omap, 'patch': mnae.apply_for_patch, 'cccp': mnae.apply_for_cccp}
    if apply_mode not in apply_fn.keys(): raise NotImplementedError
    if apply_mode == 'cccp': pad = stride = None
    X = apply_fn[apply_mode](X, originX, pad, stride, percent, block_list, p_add, edge_args, edge_or_binary)
    return X


class MNArrayEdge_mch(object):
    def _add_cross_ch_ae(self, X, percent, block_list):
        assert X.ndim == 5
        Xshape = X.shape
        batches, channels, mch, rows, cols = Xshape
        X = X.reshape((batches, channels, mch, -1))
        equal_size = channels * rows * cols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            total_blocks = int(round(equal_size / (blockr * blockc)))
            array_idx = im2col(self.idx_map, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
            im2colfn = im2col_compfn((rows, cols), (blockr, blockc), 1, 0, ignore_border=True)
            for b in xrange(batches):  # 对每个样本的原始图像计算canny边缘
                edge_patches = get_edge_patches(self.originX[b], self.edge_args, self.im2colfn) \
                    if self.edge_or_binary else get_binary_patches(self.originX[b], self.im2colfn)
                edge_patches = edge_patches.reshape((rows, cols, -1)).transpose((2, 0, 1)) \
                    if self.mode == 'omap' else edge_patches.reshape((-1, rows, cols))
                edge_idx_all = get_edge_idx_all(edge_patches, im2colfn)
                ch_idx = assign_ch_idx_partial_justfor_object(edge_patches, total_blocks, self.p_add)
                idx_for_array_idx = assign_onemap_idx_edge(ch_idx, edge_idx_all)
                ch_idx = np.repeat(ch_idx, blockr * blockc)
                map_idx = array_idx[idx_for_array_idx].ravel()
                X[b][ch_idx, :, map_idx] = 0.
        X = X.reshape(Xshape)
        return X

    def apply_for_omap(self, X, originX, pad, stride, percent, block_list, p_add, edge_args, edge_or_binary):
        assert X.ndim == 6 and originX.ndim == 4
        Xshape = X.shape
        batches, orows, ocols, mch, frows, fcols = Xshape
        self.originX = originX
        self.p_add = p_add
        self.edge_args = edge_args
        self.edge_or_binary = edge_or_binary
        self.idx_map = np.arange(orows * ocols).reshape((1, 1, orows, ocols))
        self.im2colfn = im2col_compfn(originX.shape[-2:], (frows, fcols), stride, pad, ignore_border=False)
        self.mode = 'omap'
        X = X.reshape((batches, orows, ocols, mch, -1)).transpose((0, 4, 3, 1, 2))
        block_list = filter(lambda x: x[0] <= orows and x[1] <= ocols, block_list)
        X = self._add_cross_ch_ae(X, percent, block_list)
        X = X.transpose((0, 3, 4, 2, 1)).reshape(Xshape)
        return X

    def apply_for_patch(self, X, originX, pad, stride, percent, block_list, p_add, edge_args, edge_or_binary):
        assert X.ndim == 6 and originX.ndim == 4
        Xshape = X.shape
        batches, orows, ocols, mch, frows, fcols = Xshape
        self.originX = originX
        self.p_add = p_add
        self.edge_args = edge_args
        self.edge_or_binary = edge_or_binary
        self.idx_map = np.arange(frows * fcols).reshape((1, 1, frows, fcols))
        self.im2colfn = im2col_compfn(originX.shape[-2:], (frows, fcols), stride, pad, ignore_border=False)
        self.mode = 'patch'
        X = X.reshape((batches, -1, mch, frows, fcols))
        block_list = filter(lambda x: x[0] <= frows and x[1] <= fcols, block_list)
        X = self._add_cross_ch_ae(X, percent, block_list)
        X = X.reshape(Xshape)
        return X


# p_add是保持加噪的比例,即1-p_add的比例不加噪
def add_mn_array_edge_mch(X, originX, pad, stride, percent=0.5, block_list=((1, 1),),
                          p_add=0.5, edge_args=None, apply_mode='omap', edge_or_binary=True):
    mnae = MNArrayEdge_mch()
    apply_fn = {'omap': mnae.apply_for_omap, 'patch': mnae.apply_for_patch}
    if apply_mode not in apply_fn.keys(): raise NotImplementedError
    X = apply_fn[apply_mode](X, originX, pad, stride, percent, block_list, p_add, edge_args, edge_or_binary)
    return X
