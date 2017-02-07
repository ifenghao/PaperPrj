# coding:utf-8

import cv2
from skimage import feature
import numpy as np
from matplotlib import pylab
import utils
from ultimate.util import *
from ultimate.noise import *

fsize = 5
pad = 0
stride = 1
noise_type = 'mn_array_edge'

tr_X, te_X, tr_y, te_y = utils.load.mnist_vary('basic', onehot=False)
tr_X = utils.pre.norm4d_per_sample(tr_X)
batches, channels, rows, cols = tr_X.shape
im2colfn = im2col_compfn((rows, cols), fsize, stride=stride, pad=pad, ignore_border=False)
patches = im2colfn(tr_X)
noise_patches = np.copy(patches)
args1 = {'pad': 0, 'stride': 1, 'percent': 0.25, 'block_list': ((15, 15),), 'edge_or_binary': True,
         'p_add': 0.9, 'edge_args': {'border': 0.1, 'sigma': 0.9, 'lth': 0.5, 'hth': 0.8},
         'apply_mode': 'omap', 'originX': tr_X}
args2 = {'percent': 0.25}
oshape = utils.basic.conv_out_shape((batches, channels, rows, cols),
                                    (1, channels, fsize, fsize),
                                    pad=pad, stride=stride, ignore_border=False)
orows, ocols = oshape[-2:]
noise_patches = noise_patches.reshape((batches, orows, ocols, fsize, fsize))
noise_patches = add_noise_decomp(noise_patches, noise_type, args1)
noise_patches = noise_patches.transpose((0, 3, 4, 1, 2)).reshape((batches, -1, orows, ocols))

utils.visual.show_map(noise_patches[[10, 100, 1000]])
