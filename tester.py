# coding:utf-8

import lasagne
import theano
from theano import tensor as T
import numpy as np
import cPickle
import os
import utils
from copy import copy

# size=50
# batchSize=3
# startRange = range(0, size - batchSize+1, batchSize)
# endRange = range(batchSize, size+1, batchSize)
# if size % batchSize != 0:
#     startRange.append(size - size % batchSize)
#     endRange.append(size)
# print startRange
# print endRange
# for start, end in zip(startRange, endRange):
#     print slice(start, end)

# class DotLayer(lasagne.layers.Layer):
#     def __init__(self, incoming, num_units, W=lasagne.init.Normal(0.01), **kwargs):
#         super(DotLayer, self).__init__(incoming, **kwargs)
#         num_inputs = self.input_shape[1]
#         self.num_units = num_units
#         self.W = self.add_param(W, (num_inputs, num_units), name='W')
#
#     def get_output_for(self, input, **kwargs):
#         return T.dot(input, self.W)
#
#     def get_output_shape_for(self, input_shape):
#         return (input_shape[0], self.num_units)
#
# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
# _srng = RandomStreams()
#
# class DropoutLayer(lasagne.layers.Layer):
#     def __init__(self, incoming, p=0.5, **kwargs):
#         super(DropoutLayer, self).__init__(incoming, **kwargs)
#         self.p = p
#
#     def get_output_for(self, input, deterministic=False, **kwargs):
#         if deterministic:  # do nothing in the deterministic case
#             return input
#         else:  # add dropout noise otherwise
#             retain_prob = 1 - self.p
#             input /= retain_prob
#             return input * _srng.binomial(input.shape, p=retain_prob,
#                                           dtype=theano.config.floatX)

# x=np.arange(6).reshape((2,3))
# y=np.arange(30).reshape((3,10))
#
# print np.dot(x,y)
#
# a=np.dot(x,y).transpose().reshape((5,2,2))
# a=np.max(a,axis=1).transpose()
# print a
#
# b=np.dot(x,y).reshape((2,5,2))
# b=np.max(b,axis=2)
# print b

# def listFiles(path, numPerDir=None):
#     fileList = []
#     try:
#         dirs=os.listdir(path)
#     except Exception:
#         return []
#     dirs=dirs[:numPerDir]
#     for n,file in enumerate(dirs):
#         subFile=os.path.join(path,file)
#         if os.path.isdir(subFile):
#             fileList.extend(listFiles(subFile, numPerDir))
#         else:
#             fileList.append(subFile)
#     return fileList
#
# for file in listFiles('/home/zfh/PycharmProjects',3):
#     print file

# from theano.tensor.signal.conv import conv2d
# x=np.arange(50).reshape(2,5,5)
# f=np.ones((3,3))
#
# X=T.tensor3()
# F=theano.shared(f)
# out=conv2d(X,F,border_mode='valid')
# func=theano.function([X],out)
# print x
# print func(x)

# import numpy as np
# import pycuda.autoinit
# import pycuda.gpuarray as gpuarray
# import skcuda.magma as magma
# np.random.seed(0)
# n = 1000
#
# # Note that the matrices have to either be stored in column-major form,
# # otherwise MAGMA will compute the solution for the system dot(a.T, x) == b:
# a = np.asarray(np.random.rand(n, n), order='F')
# b = np.asarray(np.random.rand(n), order='F')
#
# # Compute solution on the CPU:
# x = np.linalg.solve(a, b)
#
# # Compute the solution on the GPU - b_gpu is subsequently overwritten with the solution:
# a_gpu = gpuarray.to_gpu(a)
# b_gpu = gpuarray.to_gpu(b)
# magma.magma_init()
# magma.magma_dgesv_nopiv_gpu(n, 1, int(a_gpu.gpudata), n, int(b_gpu.gpudata), n)
# magma.magma_finalize()
#
# # Check that the solutions match:
# print np.allclose(x, b_gpu.get())

# def softmax(X):
#     e_x = np.exp(X - X.max(axis=1).reshape(-1,1))
#     return e_x / e_x.sum(axis=1).reshape(-1,1)
#
# x=np.random.randn(10,5)
# tinyfloat = np.finfo(theano.config.floatX).tiny
# Tmat = np.full_like(x, np.log(tinyfloat), dtype=theano.config.floatX, order='A')
# Tmat[np.arange(len(x)),np.argmax(x, axis=1)] = 0
# print x,Tmat
# print np.asarray(softmax(Tmat),np.float32)

# import utils
# import pylab as plt
# from numpy.linalg import pinv, solve
# tr_X, te_X, tr_y, te_y = utils.load.mnist()
# img = tr_X[0, 0, :, :].reshape(1, 784)
# w=np.random.randn(784, 784)
# y=img.dot(w)
# wpinv=pinv(w.T)
# xrec=(wpinv.dot(y.T)).T
# # xrec=solve(w.T,y.T).T
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(img.reshape(28,28))
# plt.subplot(1,2,2)
# plt.imshow(xrec.reshape(28,28))
# plt.show()

# from lasagne import layers
# x = T.tensor4()
# l_in = layers.InputLayer((2,2,3, 3), input_var=x)
# # gp = layers.GlobalPoolLayer(l_in, T.mean)
# # gp=layers.NonlinearityLayer(l_in,nonlinearity=lasagne.nonlinearities.softmax)
# # gpinv=layers.InverseLayer(gp,gp)
# # out1 = lasagne.layers.get_output(gp)
# # out2 = lasagne.layers.get_output(gpinv)
#
# uppool=layers.Upscale2DLayer(l_in,scale_factor=2)
# out1=lasagne.layers.get_output(uppool)
#
# func1 = theano.function([x], out1)
# a=np.random.randn(2,2,3,3)
# b=func1(a)
#
# print a
# print b

# from lasagne.theano_extensions.padding import pad
#
# x = T.tensor4()
# y = T.tensor4()
# index = utils.basic.patchIndex((10, 1, 5, 5), (3, 3))
# patch = T.flatten(x, 1)[index]
# patch = patch.reshape((-1, 9))
# f = y.reshape((8, 9)).T
# out1 = T.dot(patch, f)
# out2 = T.nnet.conv2d(x, y, border_mode='valid', filter_flip=False)
# func = theano.function([x, y], [out1, out2], allow_input_downcast=True)
#
# a = np.random.randn(10, 1, 5, 5)
# b = np.random.randn(8, 1, 3, 3)
# o1, o2 = func(a, b)
# o2 = o2.transpose((0, 2, 3, 1)).reshape((-1, 8))
# print np.allclose(o1, o2)

# a = np.random.randn(3,2,5, 5)
# print np.pad(a,((0,0),(0,0),(1,1),(1,1)),mode='constant',constant_values=0).shape

# from decaf import base
# from decaf.layers import core_layers, fillers

# ourblob=base.Blob()
# inblob=base.Blob(shape=(1,5,5,1),filler=fillers.GaussianRandFiller())
# layer=core_layers.Im2colLayer(name='im2col',psize=3,stride=1)
# layer.forward([inblob],[ourblob])
# print inblob.data().transpose((0,3,1,2)), ourblob.data()

# from decaf.layers.cpp import wrapper
# def get_new_shape(features, psize, stride):
#     """Gets the new shape of the im2col operation."""
#     if features.ndim != 4:
#         raise ValueError('Input features should be 4-dimensional.')
#     num, height, width, channels = features.shape
#     return (num,
#             (height - psize) / stride + 1,
#             (width - psize) / stride + 1,
#             channels * psize * psize)
#
# features=np.arange(50,dtype='float').reshape(2,5,5,1)
# print features.dtype
# shape=get_new_shape(features, 4, 2)
# output=np.zeros(shape,dtype='float')
# wrapper.im2col_forward(features, output, 4, 2)
# print features.transpose((0,3,1,2)), output

# features=np.arange(50).reshape(2,1,5,5)
# im2col=utils.basic.Im2ColOp(psize=3, stride=1)
# a=im2col.transform(features).reshape((-1,9))
# index=utils.basic.patchIndex((2,1,5,5),(3,3),(1,1))
# b=features.flat[index].reshape((-1,9))
# print a,b

# import pylearn2.scripts.datasets.make_cifar10_gcn_whitened
# from keras.preprocessing.image import ImageDataGenerator

# a=np.random.randn(10,8)
# b=np.random.randn(8,10)
# c1=a.dot(b)
# asplit=np.split(a,2,axis=1)
# bsplit=np.split(b,2,axis=0)
# c2=asplit[0].dot(bsplit[0])+asplit[1].dot(bsplit[1])
# print np.allclose(c1,c2)

# xnp=np.random.randn(20,150)
# ynp=np.random.randn(150,20)
# subsize=13
# maxdim = np.max(xnp.shape)
# parts = maxdim // subsize + 1  # 分块矩阵的分块数
# index = [subsize * i for i in range(1, parts)]
# print index
# xparts = np.split(xnp, index, axis=1)
# yparts = np.split(ynp, index, axis=0)
# partsum = []
# for x, y in zip(xparts, yparts):
#     partsum.append(np.dot(x, y))
# print np.allclose(sum(partsum), np.dot(xnp,ynp))

# def convactfn(pad, stride):
#     xt = T.ftensor4()
#     ft = T.ftensor4()
#     convxf = T.nnet.conv2d(xt, ft, border_mode=pad, subsample=stride, filter_flip=False)
#     convxf = T.tanh(convxf)
#     conv2d = theano.function([xt, ft], convxf, allow_input_downcast=True)
#     return conv2d
#
# def dotfn():
#     xt = T.fmatrix()
#     yt = T.fmatrix()
#     dotxy = T.dot(xt, yt)
#     dotact = theano.function([xt, yt], dotxy, allow_input_downcast=True)
#     return dotact
#
#
# dot = dotfn()
# conv=convactfn(3, (1,1))
#
# filters=np.random.randn(49,32)
# image=np.random.randn(2,2,28,28)
# output=0.
# for ch in range(2):
#     oneChannel = image[:, ch, :, :].reshape((2, 1, 28, 28))
#     padding = ((0, 0), (0, 0), (3, 3), (3, 3))
#     oneChannel = np.pad(oneChannel, padding, mode='constant', constant_values=0)
#     im2col = utils.basic.Im2ColOp(psize=7, stride=1)
#     patches = im2col.transform(oneChannel)
#     patches = patches.reshape((-1, 49))
#     output+=dot(patches,filters).reshape((2, 28, 28, 32)).transpose((0, 3, 1, 2))
# output=np.tanh(output)
#
# filters=filters.reshape((7,7,1,32)).transpose((3,2,0,1))
# convout=conv(image,filters)
#
# print np.allclose(output,convout)

# def get(inputX):
#     im2col = utils.basic.Im2ColOp(psize=3, stride=1)
#     out = im2col.transform(inputX).reshape((-1, 9))
#     return out
#
# features=np.arange(375).reshape(5,3,5,5)
#
# for i in range(3):
#     onechannel =features[:,i,:,:].reshape(5,1,5,5)
#     print onechannel, get(onechannel)
# index=utils.basic.patchIndex((5,3,5,5),(3,3),(1,1))
# b=features.flat[index].reshape((-1,9))
# print b

# from theano.sandbox.neighbours import images2neibs
# images = T.tensor4('images')
# neibs = images2neibs(images, neib_shape=(3, 3), neib_step=(1,1))
# f = theano.function([images], neibs)
# features=np.arange(225).reshape(3,3,5,5)
# l=[]
# for i in range(3):
#     l.append(f(features[:,i,:,:].reshape((3,1,5,5))))
# print np.concatenate(l), f(features)

# a=np.arange(200).reshape((10,5,2,2))
# index1=np.arange(10)
# index2=np.random.randint(5,size=10)
# print a[index1,index2,...]

# from sklearn.linear_model import MultiTaskLasso, Lasso, LassoLars
#
# rng = np.random.RandomState(42)
#
# # Generate some 2D coefficients with sine waves with random frequency and phase
# n_samples, n_features, n_tasks = 100, 30, 40
# n_relevant_features = 5
# coef = np.zeros((n_tasks, n_features))
# times = np.linspace(0, 2 * np.pi, n_tasks)
# for k in range(n_relevant_features):
#     coef[:, k] = np.sin((1. + rng.randn(1)) * times + 3 * rng.randn(1))
#
# X = rng.randn(n_samples, n_features)
# Y = np.dot(X, coef.T) + rng.randn(n_samples, n_tasks)
#
# # coef_lasso_ = np.array([Lasso(alpha=0.5).fit(X, y).coef_ for y in Y.T])
# coef_lasso_2 = Lasso(alpha=0.5).fit(X, Y).coef_
# coef_lasso_3 = LassoLars(alpha=0.01).fit(X, Y).coef_
# coef_lasso_4 = LassoLars(alpha=0.1).fit(X, Y).coef_
# coef_multi_task_lasso_ = MultiTaskLasso(alpha=1.).fit(X, Y).coef_
# print coef_lasso_2,coef_lasso_3,coef_lasso_4, coef_multi_task_lasso_

# a=np.arange(10)
# a=np.tile(a,5).reshape((5,10))
# map(np.random.shuffle,a)
# print a

# from sklearn.neighbors import NearestNeighbors
# import numpy as np
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
# distances, indices = nbrs.kneighbors()
# print indices

# a=np.arange(5,25)
# i=np.random.randint(20, size=(5,3))
# x=np.take(a,i)
# y=x-np.repeat(x[:,0].reshape((-1,1)),3,axis=1)
# y[np.where(y!=0)]=-1
# aa=np.where(y!=0)
# print y

# w=np.zeros((5,5))
# index=np.arange(5)
# index=np.repeat(index,3,axis=0)
# nn=np.random.randint(5,size=15)
# graph=np.arange(15)
# w[index,nn] = graph
# print index, nn,graph,w


# from scipy.sparse import csr_matrix,dok_matrix
# a=dok_matrix((10,10))
# idx1=np.random.randint(10,size=15)
# idx2=np.random.randint(10, size=15)
# a[idx1,idx2]=1
# a[idx1,idx2]=1
# b=dok_matrix((10,10))
# b[idx1,idx2]=1
# b[idx1,idx2]=1
# print a,b,a.dot(b)

# def cartesian(arrays, out=None):
#     """
#     Generate a cartesian product of input arrays.
#
#     Parameters
#     ----------
#     arrays : list of array-like
#         1-D arrays to form the cartesian product of.
#     out : ndarray
#         Array to place the cartesian product in.
#
#     Returns
#     -------
#     out : ndarray
#         2-D array of shape (M, len(arrays)) containing cartesian products
#         formed of input arrays.
#
#     Examples
#     --------
#     >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
#     array([[1, 4, 6],
#            [1, 4, 7],
#            [1, 5, 6],
#            [1, 5, 7],
#            [2, 4, 6],
#            [2, 4, 7],
#            [2, 5, 6],
#            [2, 5, 7],
#            [3, 4, 6],
#            [3, 4, 7],
#            [3, 5, 6],
#            [3, 5, 7]])
#
#     """
#
#     arrays = [np.asarray(x) for x in arrays]
#     dtype = arrays[0].dtype
#
#     n = np.prod([x.size for x in arrays])
#     if out is None:
#         out = np.zeros([n, len(arrays)], dtype=dtype)
#
#     m = n / arrays[0].size
#     out[:,0] = np.repeat(arrays[0], m)
#     if arrays[1:]:
#         cartesian(arrays[1:], out=out[0:m,1:])
#         for j in xrange(1, arrays[0].size):
#             out[j*m:(j+1)*m,1:] = out[0:m,1:]
#     return out
#
# idx= cartesian([np.array([1,2,3]),np.array([1,2,3])])

# a=np.arange(10).reshape((5,2))
# print (a>3)*(a<8)
# print np.where((a>3) *(a<8))

# a=np.random.uniform(size=(15,15))
# index=np.where(a<0.1)
# idx1=index[0]
# idx2=index[1]
# print idx1
# print idx2
# length=len(idx1)
#
# idx1=np.tile(np.repeat(idx1,3),3)
# tmp1=np.repeat(np.array([0,1,2]),length*3)
# idx1+=tmp1
#
# idx2=np.repeat(np.tile(idx2, 3),3)
# tmp2=np.tile(np.array([0,1,2]),length*3)
# idx2+=tmp2
#
# print np.vstack((idx1,idx2))
#
# bond= np.where((idx2>=15)+(idx1>=15))
# idx1, idx2= np.delete(idx1, bond), np.delete(idx2,bond)
# a[idx1,idx2]=0
# print a

# def mask(row_idx, col_idx, row_size, col_size):
#     assert len(row_idx)==len(col_idx)
#     length = len(row_idx)
#     row_idx = np.tile(np.repeat(row_idx, col_size), row_size)
#     bias = np.repeat(np.arange(row_size), length * col_size)
#     row_idx += bias
#     col_idx = np.repeat(np.tile(col_idx, row_size), col_size)
#     bias = np.tile(np.arange(col_size), length * row_size)
#     col_idx += bias
#     return row_idx, col_idx
#
# block_row_size1=3
# block_col_size2=4
# X=np.ones((5,1,15,15))
# uniform = np.random.uniform(low=0., high=1., size=X.shape)
# index = np.where((uniform > 0) * (uniform <= 0.01))
# index_b_ch = map(lambda x: np.repeat(np.tile(x, block_row_size1), block_col_size2), index[:2])
# index_r, index_c = mask(index[2], index[3], block_row_size1, block_col_size2)
# out_of_bond = np.where((index_r >= 15) + (index_c >= 15))
# index = map(lambda x: np.delete(x, out_of_bond), index_b_ch + [index_r, index_c])
# X[index] = 0.
# print X

# from theano.sandbox.neighbours import images2neibs
# from lasagne.theano_extensions.padding import pad as lasagnepad
# def im2col(inputX, fsize, stride, pad):
#     assert inputX.ndim == 4
#     Xrows, Xcols = inputX.shape[-2:]
#     X = T.tensor4()
#     if pad is None:  # 保持下和右的边界
#         rowpad = colpad = 0
#         rowrem = (Xrows - fsize) % stride
#         if rowrem: rowpad = stride - rowrem
#         colrem = (Xcols - fsize) % stride
#         if colrem: colpad = stride - colrem
#         pad = ((0, rowpad), (0, colpad))
#     Xpad = lasagnepad(X, pad, batch_ndim=2)
#     neibs = images2neibs(Xpad, (fsize, fsize), (stride, stride), 'ignore_borders')
#     im2colfn = theano.function([X], neibs, allow_input_downcast=True)
#     return im2colfn(inputX)
#
# x=np.arange(100).reshape((2,2,5,5))
# a=im2col(x,3,1,None)
# print a

# for i in range(28):
#     print np.ceil(1.414*(i+0.5)),np.ceil(1.414*(i+1.5))

# def maxpool(xnp, pool_size, ignore_border=False, stride=None, pad=(0, 0)):
#     batch, channel, img_row, img_col = xnp.shape
#     if stride is None:
#         stride = pool_size
#     if pad != (0, 0):
#         img_row += 2 * pad[0]
#         img_col += 2 * pad[1]
#         ynp = np.zeros((xnp.shape[0], xnp.shape[1], img_row, img_col), dtype=xnp.dtype)
#         ynp[:, :, pad[0]:(img_row - pad[0]), pad[1]:(img_col - pad[1])] = xnp
#         print ynp
#     else:
#         ynp = xnp
#     fsize = (channel, channel) + pool_size
#     out_shape = utils.basic.conv_out_shape(xnp.shape, fsize, pad, stride)
#     out_shape = list(out_shape)
#     if not ignore_border:
#         out_shape[2] += (img_row - pool_size[0]) % stride[0]
#         out_shape[3] += (img_col - pool_size[1]) % stride[1]
#     out = np.empty(out_shape, dtype=xnp.dtype)
#     for b in xrange(batch):
#         for ch in xrange(channel):
#             for r in xrange(out_shape[2]):
#                 row_start = r * stride[0]
#                 row_end = min(row_start + pool_size[0], img_row)
#                 for c in xrange(out_shape[3]):
#                     col_start = c * stride[1]
#                     col_end = min(col_start + pool_size[1], img_col)
#                     patch = ynp[b, ch, row_start:row_end, col_start:col_end]
#                     out[b, ch, r, c] = np.max(patch)
#     return out
#
#
# x = np.arange(196).reshape((2, 2, 7, 7))
# print x, maxpool(x, (2, 2), True)

# def fmp_overlap(xnp, pool_ratio, constant, overlap=True):
#     batch, channel, img_row, img_col = xnp.shape
#     out_row = int((float(img_row) / pool_ratio))
#     out_col = int((float(img_col) / pool_ratio))
#     row_idx = [int(pool_ratio * (i + constant)) for i in xrange(out_row + 1)]
#     if row_idx[-1] != img_row:
#         row_idx.append(img_row)
#         out_row += 1
#     row_idx = np.array(row_idx, dtype=np.int)
#     col_idx = [int(pool_ratio * (i + constant)) for i in xrange(out_col + 1)]
#     if col_idx[-1] != img_col:
#         col_idx.append(img_col)
#         out_col += 1
#     col_idx = np.array(col_idx, dtype=np.int)
#     out_shape = (batch, channel, out_row, out_col)
#     out = np.empty(out_shape, dtype=xnp.dtype)
#     for b in xrange(batch):
#         for ch in xrange(channel):
#             for r in xrange(out_row):
#                 row_start = row_idx[r]
#                 row_end = row_idx[r + 1] + 1 if overlap else row_idx[r + 1]
#                 for c in xrange(out_col):
#                     col_start = col_idx[c]
#                     col_end = col_idx[c + 1] + 1 if overlap else col_idx[c + 1]
#                     patch = xnp[b, ch, row_start:row_end, col_start:col_end]
#                     out[b, ch, r, c] = np.max(patch)
#     return out
#
#
# x = np.arange(484).reshape((2, 2, 11, 11))
# print x, fmp_overlap(x, 1.414, 0.5, True)

# # Sparse
# def sparse_random(input_unit, hidden_unit):
#     filters = np.random.uniform(low=0, high=1, size=(input_unit, hidden_unit))
#     neg_idx = np.where((filters >= 0) * (filters < 1. / 6.))
#     zero_idx = np.where((filters >= 1. / 6.) * (filters < 5. / 6.))
#     pos_idx = np.where((filters >= 5. / 6.) * (filters < 1.))
#     filters[neg_idx] = -1.
#     filters[zero_idx] = 0.
#     filters[pos_idx] = 1.
#     ranges = np.sqrt(2.0 / input_unit)
#     bias = np.random.uniform(low=-ranges, high=ranges, size=hidden_unit)
#     return filters, bias


# x=np.random.rand(10,5)
# c=np.cov(x,rowvar=0)
# y=x-x.mean(0)
# c1=y.T.dot(y)/(10-1)
# print c, c1

# def partition_channels(channels, n_jobs):
#     if n_jobs < 0:
#         n_jobs = max(mp.cpu_count() + 1 + n_jobs, 1)
#     n_jobs = min(n_jobs, channels)
#     n_channels_per_job = (channels // n_jobs) * np.ones(n_jobs, dtype=np.int)
#     n_channels_per_job[:channels % n_jobs] += 1
#     starts = np.cumsum(n_channels_per_job)
#     return n_jobs, [0] + starts.tolist()
#
#
# def train_lin_job(inputX, func, n_hidden, filter_size, stride, pool_size):
#     batches, channels, rows, cols = inputX.shape
#     _, _, orows, ocols = utils.basic.conv_out_shape((batches, channels, rows, cols),
#                                                       (n_hidden, channels, filter_size, filter_size),
#                                                       pad=filter_size // 2, stride=stride)
#     output = 0.
#     filters = []
#     for ch in xrange(channels):
#         oneChannel = inputX[:, [ch], :, :]
#         beta = func(oneChannel)
#         patches = im2col(oneChannel, filter_size, pad=filter_size // 2, stride=stride)
#         patches = patches.reshape((batches, orows, ocols, -1))
#         output += dot_pool(patches, beta, pool_size)
#         filters.append(beta)
#     return output, filters

# n_jobs, starts = partition_channels(inputX.shape[1], -1)
# result = Parallel(n_jobs=n_jobs)(
#     delayed(train_lin_job)(inputX[:, starts[i]:starts[i + 1], :, :], self._get_beta,
#                  self.n_hidden, self.filter_size, self.stride, self.pool_size)
#     for i in xrange(n_jobs))

# x= np.random.rand(10,10,3,3)
# y=np.sum(x,axis=(0,2,3))
# idx=np.argsort(y)
# idx1=idx[-5:]
# # z=np.array([x[b,ch] for b,ch in enumerate(idx)])
# # a=np.sum(z,axis=(2,3))
# # zz=np.array([x[b,ch] for b,ch in enumerate(idx1)])
# # aa=np.sum(zz,axis=(2,3))
# # print a,aa
# print idx,y

# def split_map(x, split_size):
#     assert x.ndim == 4
#     batches, channels, rows, cols = x.shape
#     splited_rows = int(np.ceil(float(rows) / split_size))
#     splited_cols = int(np.ceil(float(cols) / split_size))
#     rowpad = splited_rows * split_size - rows
#     colpad = splited_cols * split_size - cols
#     pad = (0, rowpad, 0, colpad)
#     x = utils.basic.pad2d(x, pad)
#     result = []
#     for i in xrange(split_size):
#         for j in xrange(split_size):
#             result.append(x[:, :,
#                           i * splited_rows:(i + 1) * splited_rows,
#                           j * splited_cols:(j + 1) * splited_cols])
#     return result
#
# def join_map(x_list, split_size):
#     result=[]
#     for i in xrange(split_size):
#         result.append(np.concatenate(x_list[i*split_size:(i+1)*split_size], axis=3))
#     return np.concatenate(result, axis=2)
#
# def split_neib(x, split_size):
#     assert x.ndim == 4
#     batches, channels, rows, cols = x.shape
#     splited_rows = int(np.ceil(float(rows) / split_size))
#     splited_cols = int(np.ceil(float(cols) / split_size))
#     rowpad = splited_rows * split_size - rows
#     colpad = splited_cols * split_size - cols
#     pad = (0, rowpad, 0, colpad)
#     x = utils.basic.pad2d(x, pad)
#     result = []
#     for i in xrange(split_size):
#         for j in xrange(split_size):
#             result.append(x[:, :, i::split_size, j::split_size])
#     return result
#
#
# def join_neib(x_list, split_size):
#     result = []
#     for i in xrange(split_size):
#         x_row = x_list[i * split_size:(i + 1) * split_size]
#         x_row = map(lambda x: x[:, :, :, np.newaxis, :, np.newaxis], x_row)
#         x_row = np.concatenate(x_row, axis=5)
#         x_row = x_row.reshape(x_row.shape[:-2] + (-1,))
#         result.append(x_row)
#     result = np.concatenate(result, axis=3)
#     result = result.reshape(result.shape[:2] + (-1, result.shape[-1]))
#     return result
#
# x=np.arange(100*2).reshape((2,1,10,10))
# y=split_neib(x, 3)
# print x,y
# z=join_neib(y, 3)
# print z


# class ELMAECrossAllMBetaLayer(Layer):
#     def __init__(self, C, n_hidden, filter_size, pad, stride, noise_level, part_size, idx_type,
#                  pool_type, pool_size, mode, cccp_out, cccp_noise_level):
#         self.C = C
#         self.n_hidden = n_hidden
#         self.filter_size = filter_size
#         self.stride = stride
#         self.pad = pad
#         self.noise_level = noise_level
#         self.part_size = part_size
#         if idx_type == 'block':
#             self.get_idx = get_block_idx
#         elif idx_type == 'neib':
#             self.get_idx = get_neib_idx
#         elif idx_type == 'rand':
#             self.get_idx = get_rand_idx
#         else:
#             raise NameError
#         self.pool_type = pool_type
#         self.pool_size = pool_size
#         self.mode = mode
#         self.cccp_out = cccp_out
#         self.cccp_noise_level = cccp_noise_level
#
#     def _get_beta(self, patches, W, b, bias_scale=25):
#         # 生成随机正交滤波器
#         # W, b = normal_random(input_unit=self.filter_size ** 2, hidden_unit=self.n_hidden)
#         W = orthonormalize(W)  # 正交后的幅度在-1~+1之间
#         # 在patches上加噪
#         noise_patches = np.copy(patches)
#         noise_patches = add_noise_decomp(noise_patches, add_mn, self.noise_level)
#         # noisePatch = add_mn_row(patches, p=0.25)
#         # noisePatch = add_sp(patches, p=0.25)
#         # noisePatch = add_gs(patches, p=0.25)
#         hiddens = np.dot(noise_patches, W)
#         del noise_patches
#         hmax, hmin = np.max(hiddens, axis=0), np.min(hiddens, axis=0)
#         scale = (hmax - hmin) / (2 * bias_scale)
#         hiddens += b * scale
#         hiddens = relu(hiddens)
#         # 计算beta
#         beta = compute_beta_direct(hiddens, patches)
#         beta = beta.T
#         return beta
#
#     def _train_forward(self, inputX):
#         batches, channels, rows, cols = inputX.shape
#         _, _, orows, ocols = utils.basic.conv_out_shape((batches, channels, rows, cols),
#                                                           (self.n_hidden, channels, self.filter_size, self.filter_size),
#                                                           pad=self.pad, stride=self.stride)
#         patches = im2col_catch(inputX, self.filter_size, pad=self.pad, stride=self.stride)
#         del inputX
#         patches = patches.reshape((batches, orows * ocols, channels * self.filter_size ** 2))
#         W, b = normal_random(input_unit=self.filter_size ** 2, hidden_unit=self.n_hidden)
#         self.idx = self.get_idx(self.part_size, orows, ocols)
#         self.filters = []
#         output = []
#         for num in xrange(len(self.idx)):
#             one_part = get_indexed(patches, self.idx[num])
#             ##########################
#             one_part = norm2d(one_part)
#             # patches = whiten2d(patches, self.mean1, self.P1)
#             ##########################
#             beta = self._get_beta(one_part, W, b)
#             utils.visual.save_beta(beta, dir_name, 'beta')
#             one_part = np.dot(one_part, beta)
#             one_part = one_part.reshape((batches, -1, self.n_hidden))
#             output = np.concatenate([output, one_part], axis=1) if len(output) != 0 else one_part
#             self.filters.append(copy(beta))
#         output = join_result(output, self.idx, orows, ocols)
#         return output
#
#     def get_train_output_for(self, inputX):
#         inputX = self._train_forward(inputX)
#         utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmraw')
#         # 池化
#         inputX = pool_op(inputX, self.pool_size, pool_type=self.pool_type, mode=self.mode)
#         utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmpool')
#         # 归一化
#         # patches = norm4d(patches)
#         # utils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmnorm')
#         # 激活
#         inputX = relu(inputX)
#         utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmrelu')
#         # 添加cccp层组合
#         self.cccp = CCCPLayer(C=self.C, n_out=self.cccp_out, noise_level=self.cccp_noise_level)
#         inputX = self.cccp.get_train_output_for(inputX)
#         return inputX
#
#     def _test_forward(self, inputX):
#         batches, channels, rows, cols = inputX.shape
#         _, _, orows, ocols = utils.basic.conv_out_shape((batches, channels, rows, cols),
#                                                           (self.n_hidden, channels, self.filter_size, self.filter_size),
#                                                           pad=self.pad, stride=self.stride)
#         patches = im2col_catch(inputX, self.filter_size, pad=self.pad, stride=self.stride)
#         del inputX
#         patches = patches.reshape((batches, orows * ocols, channels * self.filter_size ** 2))
#         output = []
#         for num in xrange(len(self.idx)):
#             one_part = get_indexed(patches, self.idx[num])
#             ##########################
#             one_part = norm2d(one_part)
#             # patches = whiten2d(patches, self.mean1, self.P1)
#             ##########################
#             one_part = np.dot(one_part, self.filters[num])
#             one_part = one_part.reshape((batches, -1, self.n_hidden))
#             output = np.concatenate([output, one_part], axis=1) if len(output) != 0 else one_part
#         output = join_result(output, self.idx, orows, ocols)
#         return output
#
#     def get_test_output_for(self, inputX):
#         inputX = self._test_forward(inputX)
#         utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmrawte')
#         # 池化
#         inputX = pool_op(inputX, self.pool_size, pool_type=self.pool_type, mode=self.mode)
#         utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmpoolte')
#         # 归一化
#         # patches = norm4d(patches)
#         # utils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmnormte')
#         # 激活
#         inputX = relu(inputX)
#         utils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmrelute')
#         # 添加cccp层组合
#         inputX = self.cccp.get_test_output_for(inputX)
#         return inputX
#
#
# class ELMAECrossPartMBetaLayer(Layer):
#     def __init__(self, C, n_hidden, filter_size, pad, stride, noise_level, part_size, idx_type, cross_size,
#                  pool_type, pool_size, mode, cccp_out, cccp_noise_level):
#         self.C = C
#         self.n_hidden = n_hidden
#         self.filter_size = filter_size
#         self.stride = stride
#         self.pad = pad
#         self.noise_level = noise_level
#         self.part_size = part_size
#         self.cross_size = cross_size
#         if idx_type == 'block':
#             self.get_idx = get_block_idx
#         elif idx_type == 'neib':
#             self.get_idx = get_neib_idx
#         elif idx_type == 'rand':
#             self.get_idx = get_rand_idx
#         else:
#             raise NameError
#         self.pool_type = pool_type
#         self.pool_size = pool_size
#         self.mode = mode
#         self.cccp_out = cccp_out
#         self.cccp_noise_level = cccp_noise_level
#
#     def _get_beta(self, patches, W, b, bias_scale=25):
#         # 生成随机正交滤波器
#         # W, b = normal_random(input_unit=self.filter_size ** 2, hidden_unit=self.n_hidden)
#         W = orthonormalize(W)  # 正交后的幅度在-1~+1之间
#         # 在patches上加噪
#         noise_patches = np.copy(patches)
#         noise_patches = add_noise_decomp(noise_patches, add_mn, self.noise_level)
#         # noisePatch = add_mn_row(patches, p=0.25)
#         # noisePatch = add_sp(patches, p=0.25)
#         # noisePatch = add_gs(patches, p=0.25)
#         hiddens = np.dot(noise_patches, W)
#         del noise_patches
#         hmax, hmin = np.max(hiddens, axis=0), np.min(hiddens, axis=0)
#         scale = (hmax - hmin) / (2 * bias_scale)
#         hiddens += b * scale
#         hiddens = relu(hiddens)
#         # 计算beta
#         beta = compute_beta_direct(hiddens, patches)
#         beta = beta.T
#         return beta
#
#     def _train_forward(self, partX):
#         batches, channels, rows, cols = partX.shape
#         _, _, orows, ocols = utils.basic.conv_out_shape((batches, channels, rows, cols),
#                                                           (self.n_hidden, channels, self.filter_size, self.filter_size),
#                                                           pad=self.pad, stride=self.stride)
#         patches = im2col_catch(partX, self.filter_size, pad=self.pad, stride=self.stride)
#         del partX
#         patches = patches.reshape((batches, orows * ocols, channels * self.filter_size ** 2))
#         W, b = normal_random(input_unit=self.filter_size ** 2, hidden_unit=self.n_hidden)
#         self.idx = self.get_idx(self.part_size, orows, ocols)
#         filters = []
#         output = []
#         for num in xrange(len(self.idx)):
#             one_part = get_indexed(patches, self.idx[num])
#             ##########################
#             one_part = norm2d(one_part)
#             # patches = whiten2d(patches, self.mean1, self.P1)
#             ##########################
#             beta = self._get_beta(one_part, W, b)
#             utils.visual.save_beta(beta, dir_name, 'beta')
#             one_part = np.dot(one_part, beta)
#             one_part = one_part.reshape((batches, -1, self.n_hidden))
#             output = np.concatenate([output, one_part], axis=1) if len(output) != 0 else one_part
#             filters.append(copy(beta))
#         output = join_result(output, self.idx, orows, ocols)
#         return output, filters
#
#     def get_train_output_for(self, inputX):
#         batches, channels, rows, cols = inputX.shape
#         # 将输入按照通道分为多个组,每个组学习一个beta
#         self.filters = []
#         self.cccps = []
#         output = []
#         splits = int(np.ceil(float(channels) / self.cross_size))
#         for num in xrange(splits):
#             # 取部分通道
#             partX = inputX[:, :self.cross_size, :, :]
#             inputX = inputX[:, self.cross_size:, :, :]
#             partX, partXFilter = self._train_forward(partX)
#             utils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmraw')
#             # 池化
#             partX = pool_op(partX, self.pool_size, pool_type=self.pool_type, mode=self.mode)
#             utils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmpool')
#             # 归一化
#             # patches = norm4d(patches)
#             # utils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmnorm')
#             # 激活
#             partX = relu(partX)
#             utils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmrelu')
#             # 添加cccp层组合
#             cccp = CCCPLayer(C=self.C, n_out=self.cccp_out, noise_level=self.cccp_noise_level)
#             partX = cccp.get_train_output_for(partX)
#             output = np.concatenate([output, partX], axis=1) if len(output) != 0 else partX
#             self.filters.append(deepcopy(partXFilter))
#             self.cccps.append(deepcopy(cccp))
#             print num,
#             gc.collect()
#         return output
#
#     def _test_forward(self, partX, partXFilter):
#         batches, channels, rows, cols = partX.shape
#         _, _, orows, ocols = utils.basic.conv_out_shape((batches, channels, rows, cols),
#                                                           (self.n_hidden, channels, self.filter_size, self.filter_size),
#                                                           pad=self.pad, stride=self.stride)
#         patches = im2col_catch(partX, self.filter_size, pad=self.pad, stride=self.stride)
#         del partX
#         patches = patches.reshape((batches, orows * ocols, channels * self.filter_size ** 2))
#         output = []
#         for num in xrange(len(self.idx)):
#             one_part = get_indexed(patches, self.idx[num])
#             ##########################
#             one_part = norm2d(one_part)
#             # patches = whiten2d(patches, self.mean1, self.P1)
#             ##########################
#             one_part = np.dot(one_part, partXFilter[num])
#             one_part = one_part.reshape((batches, -1, self.n_hidden))
#             output = np.concatenate([output, one_part], axis=1) if len(output) != 0 else one_part
#         output = join_result(output, self.idx, orows, ocols)
#         return output
#
#     def get_test_output_for(self, inputX):
#         batches, channels, rows, cols = inputX.shape
#         output = []
#         splits = int(np.ceil(float(channels) / self.cross_size))
#         for num in xrange(splits):
#             # 取部分通道
#             partX = inputX[:, :self.cross_size, :, :]
#             inputX = inputX[:, self.cross_size:, :, :]
#             partX = self._test_forward(partX, self.filters[num])
#             utils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmrawte')
#             # 池化
#             partX = pool_op(partX, self.pool_size, pool_type=self.pool_type, mode=self.mode)
#             utils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmpoolte')
#             # 归一化
#             # patches = norm4d(patches)
#             # utils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmnormte')
#             # 激活
#             partX = relu(partX)
#             utils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmrelute')
#             # 添加cccp层组合
#             partX = self.cccps[num].get_test_output_for(partX)
#             output = np.concatenate([output, partX], axis=1) if len(output) != 0 else partX
#             print num,
#             gc.collect()
#         return output


# import pylearn2.scripts.datasets.make_cifar10_gcn_whitened

# x=np.array([[1,2,3,4,5,6,7,8],[0,0,0,0,0,0,0,0]],dtype='float')
# print np.var(x,axis=1),((x-x.mean(axis=1)[:,np.newaxis]) ** 2).mean(axis=1)

# X=np.random.randn(10,5)
# cov = np.dot(X.T, X) / X.shape[0]
# D, V = np.linalg.eig(cov)
# P = V.dot(np.diag(np.sqrt(1 / (D + 0.1)))).dot(V.T)
# print D,V,P
#
# X/=10
# cov = np.dot(X.T, X) / X.shape[0]
# D, V = np.linalg.eig(cov)
# P = V.dot(np.diag(np.sqrt(1 / (D + 0.1)))).dot(V.T)
# print D,V,P

# def _get_block_idx(blockr, blockc,orows, ocols):
#     nr = int(np.ceil(float(orows) / blockr))
#     nc = int(np.ceil(float(ocols) / blockc))
#     idx = []
#     for row in xrange(nr):
#         row_bias = row * blockr
#         for col in xrange(nc):
#             col_bias = col * blockc
#             base = np.arange(blockc) if col_bias + blockc < ocols else np.arange(ocols - col_bias)
#             block_row = blockr if row_bias + blockr < orows else orows - row_bias
#             one_block = []
#             for br in xrange(block_row):
#                 one_row = base + orows * br + col_bias + row_bias * orows
#                 one_block = np.concatenate([one_block, one_row]) if len(one_block) != 0 else one_row
#             idx.append(one_block)
#     return idx
#
# def join_result(blocks, blockr, blockc, orows, ocols):
#     batches = blocks[0].shape[0] / (blockr * blockc)
#     channels = blocks[0].shape[1]
#     nr = int(np.ceil(float(orows) / blockr))
#     nc = int(np.ceil(float(ocols) / blockc))
#     output = []
#     for row in xrange(nr):
#         one_row = []
#         for col in xrange(nc):
#             one_block = blocks.pop(0)
#             if col == nc - 1 and row != nr - 1:
#                 one_block = one_block.reshape((batches, blockr, -1, channels))
#             elif row == nr - 1 and col != nc - 1:
#                 one_block = one_block.reshape((batches, -1, blockc, channels))
#             elif row == nr - 1 and col == nc - 1:
#                 rsize = orows % blockr if orows % blockr else blockr
#                 csize = ocols % blockc if ocols % blockc else blockc
#                 one_block = one_block.reshape((batches, rsize, csize, channels))
#             else:
#                 one_block = one_block.reshape((batches, blockr, blockc, channels))
#             one_block = one_block.transpose((0, 3, 1, 2))
#             one_row = np.concatenate([one_row, one_block], axis=3) if len(one_row) != 0 else one_block
#         output = np.concatenate([output, one_row], axis=2) if len(output) != 0 else one_row
#     return output
#
# idx=_get_block_idx(5,5,10,10)
# print idx
# idx=map(lambda x:x[:,None],idx)
# print join_result(idx,5,5, 10,10)


# def get_rand_idx(n_rand, orows, ocols):
#     size = orows * ocols
#     split_size = int(round(float(size) / n_rand))
#     all_idx = np.random.permutation(size)
#     split_range = [split_size + split_size * i for i in xrange(n_rand - 1)]
#     split_idx = np.split(all_idx, split_range)
#     return split_idx
#
#
# print get_rand_idx(10, 7, 7)

# def get_neib_idx(neibr, neibc, orows, ocols):
#     idx = []
#     for i in xrange(neibr):
#         row_idx = np.arange(i, orows, neibr)
#         for j in xrange(neibc):
#             col_idx = np.arange(j, ocols, neibc)
#             one_neib = []
#             for row_step in row_idx:
#                 one_row = col_idx + row_step * orows
#                 one_neib = np.concatenate([one_neib, one_row]) if len(one_neib) != 0 else one_row
#             idx.append(one_neib)
#     return idx
#
# print np.arange(100).reshape((10,10)),get_neib_idx(4,4,10,10)


# x = np.zeros((5, 36, 23, 23))
# block_list = [(8, 8), (7, 7),(15,15)]
# batches, channels, orows, ocols = x.shape
# x = x.reshape((5, 36, -1))
# oidx = np.arange(orows * ocols).reshape((1, 1, orows, ocols))
# block_list.sort(lambda x,y:np.prod(x)-np.prod(y))
# # block_list = filter(lambda x: x[0] * x[1]*len(block_list) < orows * ocols * 0.5, block_list)
# length=len(block_list)
# for _ in xrange(length): # 将不符合的大block剔除掉
#     block=block_list[-1]
#     if block[0]*block[1]*length<=orows * ocols * 0.5: break
#     block_list.pop(-1)
#     length-=1
# equal_size = orows * ocols * 0.5 / float(len(block_list))
# for block_row, block_col in block_list:
#     onemap_blocks = int(round(equal_size / (block_row * block_col)))
#     total_blocks = channels * onemap_blocks
#     row_stride = row_stride_tmp = 1
#     col_stride = col_stride_tmp = 1
#     while True:
#         array_row = (orows - block_row) // row_stride_tmp + 1
#         array_col = (ocols - block_col) // col_stride_tmp + 1
#         if array_row * array_col < total_blocks: break
#         row_stride = row_stride_tmp
#         col_stride = col_stride_tmp
#         if row_stride_tmp < col_stride_tmp:
#             row_stride_tmp += 1
#         else:
#             col_stride_tmp += 1
#     array_row = (orows - block_row) // row_stride + 1
#     array_col = (ocols - block_col) // col_stride + 1
#     array_idx = im2col(oidx, (block_row, block_col), (row_stride, col_stride), 0, ignore_border=True)
#     array_idx = array_idx.astype(int)
#     ch_idx = np.repeat(np.arange(channels), onemap_blocks * block_row * block_col)
#     for b in xrange(batches):# 不同样本不同噪声
#         idx_for_array_idx = np.random.permutation(array_row * array_col)[:total_blocks]
#         if array_row * array_col < total_blocks:
#             times = np.ceil(float(total_blocks) / (array_row * array_col))
#             idx_for_array_idx = np.random.permutation(np.tile(idx_for_array_idx, times)[:total_blocks])
#         map_idx = array_idx[idx_for_array_idx].reshape(-1)
#         x[b][ch_idx, map_idx] = 1.
#         tmp = x[b].reshape((36, 23, 23)).astype(int)
#         # split_idx=np.split(rand_idx_idx, channels)
#         # for ch, idx in enumerate(split_idx):
#         #     ch_array_idx=array_idx[idx].reshape(-1)
#         #     x[b][ch][ch_array_idx] = 1.
#         #     print x[b][ch].reshape((23,23)).astype(int)


# x = 20
# for i in range(5):
#     x = int(float(x) / 5.5)
#     print x

# from theano.sandbox.neighbours import images2neibs
# from lasagne.theano_extensions.padding import pad as lasagnepad
#
#
# def im2col(inputX, fsize, stride, pad, ignore_border=False):
#     assert inputX.ndim == 4
#     if isinstance(fsize, (int, float)): fsize = (int(fsize), int(fsize))
#     if isinstance(stride, (int, float)): stride = (int(stride), int(stride))
#     Xrows, Xcols = inputX.shape[-2:]
#     X = T.tensor4()
#     if not ignore_border:  # 保持下和右的边界
#         rowpad = colpad = 0
#         rowrem = (Xrows - fsize[0]) % stride[0]
#         if rowrem: rowpad = stride[0] - rowrem
#         colrem = (Xcols - fsize[1]) % stride[1]
#         if colrem: colpad = stride[1] - colrem
#         pad = ((0, rowpad), (0, colpad))
#     Xpad = lasagnepad(X, pad, batch_ndim=2)
#     neibs = images2neibs(Xpad, fsize, stride, 'ignore_borders')
#     im2colfn = theano.function([X], neibs, allow_input_downcast=True)
#     return im2colfn(inputX)
#
# class MNArray(object):
#     def _scat_stride(self, blockr, blockc, total_blocks):
#         strider = strider_tmp = 1
#         stridec = stridec_tmp = 1
#         while True:
#             arrayr = (self.orows - blockr) // strider_tmp + 1
#             arrayc = (self.ocols - blockc) // stridec_tmp + 1
#             if arrayr * arrayc < total_blocks: break
#             strider = strider_tmp
#             stridec = stridec_tmp
#             if strider_tmp < stridec_tmp:
#                 strider_tmp += 1
#             else:
#                 stridec_tmp += 1
#         return strider, stridec
#
#     def _assign_ch_idx_permap(self, channels, block_size, n_blocks):
#         ch_idx = np.repeat(np.arange(channels), n_blocks * block_size)
#         return ch_idx
#
#     def _assign_ch_idx_uniform(self, channels, block_size, n_blocks):
#         ch_idx = np.random.permutation(channels)[:n_blocks]
#         if channels < n_blocks:
#             times = np.ceil(float(n_blocks) / channels)
#             ch_idx = np.tile(ch_idx, times)[:n_blocks]
#         ch_idx.sort()
#         ch_idx = np.repeat(ch_idx, block_size)
#         return ch_idx
#
#     def _assign_ch_idx_rand(self, channels, block_size, n_blocks):
#         ch_idx = np.random.randint(0, channels, n_blocks)
#         ch_idx.sort()
#         ch_idx = np.repeat(ch_idx, block_size)
#         return ch_idx
#
#     def _assign_ch_idx_retain(self, channels, block_size, n_blocks):
#         retains = round(channels * self.retain)
#         ch_idx = np.random.permutation(channels)[:retains]
#         ch_idx.sort()
#         rand_idx = np.random.randint(0, retains, n_blocks)
#         ch_idx = ch_idx[rand_idx]
#         ch_idx.sort()
#         ch_idx = np.repeat(ch_idx, block_size)
#         return ch_idx
#
#     def _assign_onemap_idx(self, array_idx, array_size, n_blocks):
#         idx_for_array_idx = np.random.permutation(array_size)[:n_blocks]  # 一定要均匀的分配map索引
#         if array_size < n_blocks:
#             times = np.ceil(float(n_blocks) / (array_size))
#             idx_for_array_idx = np.random.permutation(np.tile(idx_for_array_idx, times)[:n_blocks])
#         map_idx = array_idx[idx_for_array_idx].reshape(-1)
#         return map_idx
#
#     def _add_per_map(self, X, percent, block_list):
#         assert X.ndim == 3
#         equal_size = self.orows * self.ocols * percent / float(len(block_list))
#         for blockr, blockc in block_list:
#             map_blocks = int(round(equal_size / (blockr * blockc)))
#             total_blocks = self.channels * map_blocks
#             strider, stridec = self._scat_stride(blockr, blockc, total_blocks)
#             arrayr = (self.orows - blockr) // strider + 1
#             arrayc = (self.ocols - blockc) // stridec + 1
#             array_idx = im2col(self.oidx, (blockr, blockc), (strider, stridec), 0, ignore_border=True).astype(int)
#             for b in xrange(self.batches):  # 不同样本不同噪声
#                 ch_idx = self._assign_ch_idx_permap(self.channels, blockr * blockc, map_blocks)
#                 map_idx = self._assign_onemap_idx(array_idx, arrayr * arrayc, total_blocks)
#                 X[b][ch_idx, map_idx] = 0.
#         return X
#
#     def _add_cross_ch_ae(self, X, percent, block_list):
#         assert X.ndim == 3
#         equal_size = self.channels * self.orows * self.ocols * percent / float(len(block_list))
#         for blockr, blockc in block_list:
#             total_blocks = int(round(equal_size / (blockr * blockc)))
#             strider, stridec = self._scat_stride(blockr, blockc, total_blocks)  # 尝试最分散的stride
#             arrayr = (self.orows - blockr) // strider + 1  # 不考虑边界
#             arrayc = (self.ocols - blockc) // stridec + 1
#             array_idx = im2col(self.oidx, (blockr, blockc), (strider, stridec), 0, ignore_border=True).astype(int)
#             for b in xrange(self.batches):  # 不同样本不同噪声
#                 ch_idx = self._assign_ch_idx_retain(self.channels, blockr * blockc, total_blocks)
#                 map_idx = self._assign_onemap_idx(array_idx, arrayr * arrayc, total_blocks)
#                 X[b][ch_idx, map_idx] = 0.
#                 tmp=X[b].reshape(36,23,23).astype(int)
#                 pass
#         return X
#
#     def _add_cross_batch(self, X, percent, block_list):
#         assert X.ndim == 3
#         X = X.reshape((-1, self.orows * self.ocols))
#         equal_size = self.channels * self.orows * self.ocols * percent / float(len(block_list))
#         for blockr, blockc in block_list:
#             total_blocks = int(round(equal_size / (blockr * blockc)))
#             strider, stridec = self._scat_stride(blockr, blockc, total_blocks)  # 尝试最分散的stride
#             arrayr = (self.orows - blockr) // strider + 1  # 不考虑边界
#             arrayc = (self.ocols - blockc) // stridec + 1
#             array_idx = im2col(self.oidx, (blockr, blockc), (strider, stridec), 0, ignore_border=True).astype(int)
#             channels = self.batches * self.channels
#             total_blocks *= self.batches
#             ch_idx = self._assign_ch_idx_retain(channels, blockr * blockc, total_blocks)
#             map_idx = self._assign_onemap_idx(array_idx, arrayr * arrayc, total_blocks)
#             X[ch_idx, map_idx] = 0.
#         return X
#
#     def apply_for_omap(self, X, percent, block_list, mode, retain):
#         assert X.ndim == 4
#         add_fn = {'permap': self._add_per_map, 'channel': self._add_cross_ch_ae, 'batch': self._add_cross_batch}
#         if mode not in add_fn.keys():
#             raise NotImplementedError
#         Xshape = X.shape
#         self.batches, self.channels, self.orows, self.ocols = Xshape
#         self.retain = retain
#         self.oidx = np.arange(self.orows * self.ocols).reshape((1, 1, self.orows, self.ocols))
#         X = X.reshape((self.batches, self.channels, -1))
#         block_list = filter(lambda x: x[0] <= self.orows and x[1] <= self.ocols, block_list)
#         X = add_fn[mode](X, percent, block_list)
#         X = X.reshape(Xshape)
#         return X
#
#
# # retain是保持加噪的比例,即1-retain的比例不加噪
# def add_mn_array(X, percent=0.5, block_list=((1, 1), (2, 2)), mode='channel', retain=0.5):
#     assert X.ndim == 4
#     X = MNArray().apply_for_omap(X, percent, block_list, mode, retain)
#     return X
#
# x = np.ones((5, 36, 23, 23))
# x=add_mn_array(x,0.3, ((15,15),),'channel',0.8)


# def get_center_idx(arrayr, arrayc, p_center):
#     p_center = np.sqrt(p_center)
#     centerr, centerc = int(np.floor(arrayr * p_center)), int(np.floor(arrayc * p_center))
#     aroundr, aroundc = int(np.ceil((arrayr - centerr) / 2.)), int(np.ceil((arrayc - centerc) / 2.))
#     center_idx = []
#     for r in xrange(centerr):
#         one_row = np.arange(aroundc, aroundc + centerc) + (r + aroundr) * arrayc
#         center_idx = np.concatenate([center_idx, one_row], axis=0) if len(center_idx) != 0 else one_row
#     return center_idx
#
#
# # 只分配特征图外围的索引(背景)
# def assign_onemap_idx_around(arrayr, arrayc, n_blocks, select=0.5):
#     center_idx=get_center_idx(arrayr, arrayc, select)
#     print np.arange(arrayr*arrayc).reshape((arrayr,arrayc)),center_idx
#     total_idx=np.arange(arrayr*arrayc)
#     around_idx=np.delete(total_idx, center_idx)
#     rand_idx = np.random.randint(0, len(around_idx), n_blocks)
#     idx_for_array_idx = around_idx[rand_idx]
#     return idx_for_array_idx
#
# def assign_onemap_idx_alter(arrayr, arrayc, n_blocks, p_center=0.5, center_vs_around=0.5):
#     center_idx = get_center_idx(arrayr, arrayc, p_center)
#     print np.arange(arrayr * arrayc).reshape((arrayr, arrayc)), center_idx
#     total_idx = np.arange(arrayr * arrayc)
#     around_idx = np.delete(total_idx, center_idx)
#     n_centers = int(round(float(n_blocks) * center_vs_around))
#     idx_for_array_idx_center = np.random.permutation(center_idx)[:n_centers]
#     if len(center_idx) < n_centers:
#         times = int(np.ceil(float(n_centers) / len(center_idx)))
#         idx_for_array_idx_center = np.tile(idx_for_array_idx_center, times)[:n_centers]
#     n_arounds = n_blocks - n_centers
#     idx_for_array_idx_around = np.random.permutation(around_idx)[:n_arounds]
#     if len(around_idx) < n_arounds:
#         times = int(np.ceil(float(n_arounds) / len(around_idx)))
#         idx_for_array_idx_around = np.tile(idx_for_array_idx_around, times)[:n_arounds]
#     long_idx, short_idx = (idx_for_array_idx_center, idx_for_array_idx_around) if n_centers > n_arounds \
#         else (idx_for_array_idx_around, idx_for_array_idx_center)
#     n_long, n_short = len(long_idx), len(short_idx)
#     step_size = int(np.floor(float(n_long) / n_short))
#     tmp_idx = long_idx[:n_short * step_size].reshape((n_short, step_size))
#     short_idx = short_idx[:, None]
#     tmp_idx = np.concatenate([tmp_idx, short_idx], axis=1).ravel()
#     idx_for_array_idx = np.append(tmp_idx, long_idx[n_arounds * step_size:])
#     return idx_for_array_idx
#
# print assign_onemap_idx_alter(5,5,30, 0.5,0.8)


# def scat_stride(orows, ocols, blockr, blockc, total_blocks):
#     strider = strider_tmp = 1
#     stridec = stridec_tmp = 1
#     while True:
#         arrayr = (orows - blockr) // strider_tmp + 1
#         arrayc = (ocols - blockc) // stridec_tmp + 1
#         if arrayr * arrayc < total_blocks: break
#         strider = strider_tmp
#         stridec = stridec_tmp
#         strider_tmp += 1
#         stridec_tmp += 1
#     return strider, stridec
#
#
# def get_block_idx(arrayr, arrayc, p_block):
#     p_block = np.sqrt(p_block)
#     blockr, blockc = int(np.floor(arrayr * p_block)), int(np.floor(arrayc * p_block))
#     block_idx = []
#     for r in xrange(blockr):
#         one_row = np.arange(0, blockc) + r * arrayc
#         one_row = one_row[None, :]
#         block_idx = np.concatenate([block_idx, one_row], axis=0) if len(block_idx) != 0 else one_row
#     return block_idx
#
# def get_shift_range(need, real):
#     if need>real:
#         lack=need-real
#         step=real//lack
#         start= (real - step * lack) // 2
#         repeat=range(start,real-start, step)
#         shift_range = repeat+range(0, real, 1)
#         shift_range.sort()
#     else:
#         step = real // need
#         start = (real - step * need) // 2
#         shift_range = range(start, real - start, step)
#     return shift_range
#
#
# def shift_block_idx(num, block_idx, patchr, patchc, arrayr, arrayc):
#     assert block_idx.ndim == 2
#     blockr, blockc = block_idx.shape
#     realr, realc = arrayr - blockr + 1, arrayc - blockc + 1
#     shiftr = num // patchc
#     shiftc = num % patchc
#     shiftr_range=get_shift_range(patchr, realr)
#     shiftc_range=get_shift_range(patchc, realc)
#     shiftr=shiftr_range[shiftr]
#     shiftc=shiftc_range[shiftc]
#     block_idx_shift = block_idx + (shiftr * arrayc + shiftc)
#     print block_idx_shift.ravel()
#
#
# block_idx = get_block_idx(9, 9, 0.5)
# print np.arange(81).reshape((9, 9))
# for i in range(49):
#     shift_block_idx(i, block_idx, 7, 7, 9, 9)

# def assign_ch_idx_retain_uniform(channels, n_blocks, p_add):
#     retains = int(round(channels * p_add))
#     ch_idx = np.random.permutation(channels)[:retains]
#     times = int(np.ceil(float(n_blocks) / retains))
#     ch_idx = np.tile(ch_idx, times)[:n_blocks]
#     ch_idx.sort()
#     return ch_idx
#
# def get_center_idx(rows, cols, p_center):
#     if isinstance(p_center, (list, tuple)):
#         p_row, p_col=p_center
#     else:
#         p_row= p_col = p_center
#     centerr, centerc = int(np.floor(rows * p_row)), int(np.floor(cols * p_col))
#     aroundr, aroundc = int(np.ceil((rows - centerr) / 2.)), int(np.ceil((cols - centerc) / 2.))
#     center_idx = []
#     for r in xrange(centerr):
#         one_row = np.arange(aroundc, aroundc + centerc) + (r + aroundr) * cols
#         center_idx = np.concatenate([center_idx, one_row], axis=0) if len(center_idx) != 0 else one_row
#     return center_idx
#
# def assign_onemap_idx_orient(ch_idx, orient_idx_allch):
#     idx_for_array_idx = []
#     for ch in ch_idx:
#         orient_idx_onech = orient_idx_allch[ch]  # 找到对应通道的索引
#         rand_idx = np.random.randint(len(orient_idx_onech))  # 随机取一个索引
#         idx_for_array_idx.append((orient_idx_onech[0]))
#     return idx_for_array_idx
#
#
# class MNArrayOrient(object):
#     def _get_orient_idx_allch(self, blockr, blockc):
#         orient_idx_allch = []
#         for ch in xrange(self.channels):
#             patch = self.patches[ch][None, None, :, :]
#             blocks = im2col(patch, (blockr, blockc), 1, 0, ignore_border=True)
#             blocksum = blocks.sum(axis=1)
#             orient_idx = np.where(blocksum >= blockr * blockc * self.center_vs_around)[0]
#             if len(orient_idx) == 0:  # 如果全部是背景,则全部随机选取
#                 orient_idx = np.arange(len(blocks))
#             orient_idx_allch.append((orient_idx))
#         return orient_idx_allch
#
#     def _add_cross_ch_ae(self, X, percent, block_list):
#         assert X.ndim == 3
#         equal_size = self.channels * self.orows * self.ocols * percent / float(len(block_list))
#         for blockr, blockc in block_list:
#             total_blocks = int(round(equal_size / (blockr * blockc)))
#             array_idx = im2col(self.oidx, (blockr, blockc), 1, 0, ignore_border=True).astype(int)
#             orient_idx_allch = self._get_orient_idx_allch(blockr, blockc)
#             for b in xrange(self.batches):  # 不同样本不同噪声
#                 ch_idx = assign_ch_idx_retain_uniform(self.channels, total_blocks, self.p_add)
#                 print ch_idx
#                 idx_for_array_idx = assign_onemap_idx_orient(ch_idx, orient_idx_allch)
#                 ch_idx = np.repeat(ch_idx, blockr * blockc)
#                 map_idx = array_idx[idx_for_array_idx].ravel()
#                 X[b][ch_idx, map_idx] = 0.
#                 tmp= X[b].reshape((36,23,23)).astype(int)
#                 pass
#         return X
#
#     def apply_for_omap(self, X, pad, stride, percent, block_list, p_add, p_center, center_vs_around):
#         assert X.ndim == 4
#         Xshape = X.shape
#         self.batches, self.channels, self.orows, self.ocols = Xshape
#         filter_size = int(np.sqrt(self.channels))
#         self.p_add = p_add
#         self.center_vs_around = center_vs_around
#         # stride和pad与获取patch一致,恢复出原始图像大小
#         originr = (self.orows - 1) * stride - pad * 2 + filter_size
#         originc = (self.ocols - 1) * stride - pad * 2 + filter_size
#         # 默认目标处于中心区域
#         center_idx = get_center_idx(originr, originc, p_center)
#         origin_map = np.zeros(originr * originc, dtype=int)
#         origin_map[center_idx] = 1
#         origin_map = origin_map.reshape((1, 1, originr, originc))
#         # 将目标区域用同样的方式获取patches
#         self.patches = im2col(origin_map, filter_size, stride, pad, ignore_border=True)
#         self.patches = self.patches.reshape((self.orows, self.ocols, -1)).transpose((2, 0, 1))
#         self.oidx = np.arange(self.orows * self.ocols).reshape((1, 1, self.orows, self.ocols))
#         X = X.reshape((self.batches, self.channels, -1))
#         block_list = filter(lambda x: x[0] <= self.orows and x[1] <= self.ocols, block_list)
#         X = self._add_cross_ch_ae(X, percent, block_list)
#         X = X.reshape(Xshape)
#         return X
#
#
# # p_add是保持加噪的比例,即1-p_add的比例不加噪,p_center是中心区域面积的比例
# def add_mn_array_orient(X, pad, stride, percent=0.5, block_list=((1, 1),),
#                         p_add=0.5, p_center=0.5, center_vs_around=0.5):
#     assert X.ndim == 4
#     X = MNArrayOrient().apply_for_omap(X, pad, stride, percent, block_list, p_add, p_center, center_vs_around)
#     return X
#
# x = np.ones((5, 36, 6, 6))
# x=add_mn_array_orient(x,0,1,0.3, ((4,4),),0.5,(0.3, 0.6),0.75)

# x=10
#
# class tst():
#     def __init__(self):
#         global x
#         x=20
#         print x
#
#     def fn(self):
#         print x
#
# c=tst()
# c.fn()

# from skimage import feature
# from skimage import io, color
# from matplotlib import pylab
#
# x=io.imread('/home/zfh/a.png')
# x=color.rgb2gray(x)
# edger, edgec = x.shape
# p_border=0.01
# mask = np.zeros(x.shape, dtype=bool)
# startr, endr = np.round(edger * p_border), np.round(edger * (1 - p_border))
# startc, endc = np.round(edgec * p_border), np.round(edgec * (1 - p_border))
# mask[startr:endr, startc:endc] = True
# y=feature.canny(x,2,0.8,0.9,mask,use_quantiles=True)
# pylab.figure()
# pylab.gray()
# pylab.imshow(x, interpolation=None)
# pylab.figure()
# pylab.gray()
# pylab.imshow(y, interpolation=None)
# pylab.show()

# x=np.arange(1*49*9)
# print x.reshape((1,49,3,3))
# x=x.reshape((1,49,9)).transpose((0,2,1)).reshape((1,9,7,7))
# x[:,1,1:3,1:3]=0
# x[:,4,3:6,3:6]=0
# x[:,7,3:5,3:5]=0
# print x
# x=x.reshape((1,9,49)).transpose((0,2,1))
# print x.reshape((1,49,3,3))

# from ultimate.noise import add_mn_array_orient
# x=np.ones((1,23,23,6,6), dtype=int)
# x=add_mn_array_orient(x, 0, 1, 0.3, ((5,5),), 0.8, 0.5, 'patch')
# pass

# from numpy.linalg import solve
# Hmat=np.random.randn(20,10)
# Tmat=np.random.randn(20,5)
# beta1 = np.dot(Hmat.T, solve(np.dot(Hmat, Hmat.T)+ np.eye(20)/1e6, Tmat))
# beta2 = solve(np.dot(Hmat.T, Hmat)+np.eye(10)/1e6, np.dot(Hmat.T, Tmat))
# beta3 = np.dot(Hmat.T, solve(np.dot(Hmat, Hmat.T), Tmat))
# beta4 = solve(np.dot(Hmat.T, Hmat), np.dot(Hmat.T, Tmat))
# print beta1, beta2, beta3, beta4
# print Hmat.dot(beta1),Hmat.dot(beta2),Hmat.dot(beta3),Hmat.dot(beta4)


# x1=np.random.randn(1,10)
# y1=np.random.randn(10,1)
#
# x2=np.random.randn(1,10)
# y2=np.random.randn(10,1)
#
# print x1.dot(y1)*x2.dot(y2)
# print x1.dot(y1.dot(y2.T)).dot(x2.T)

# x=np.random.randn(2,7)
# w=np.random.randn(4,2)
#
# h=w.dot(x)
#
# hh=h.dot(h.T)
# print hh
# xx=x.dot(x.T)
# hh1=np.zeros((4,4))
# for i in range(4):
#     for j in range(4):
#         hh1[i,j]=w[i,:].dot(xx).dot(w[j,:].T)
# print hh1
# ww=np.zeros((w.shape[0],w.shape[1]*w.shape[0]))
# for i in range(w.shape[0]):
#     ww[i,i*w.shape[1]:(i+1)*w.shape[1]]=w[i,:]
# hh2=w.dot(np.tile(xx,(1,4))).dot(ww.T)
# print hh2
#
# xh=x.dot(h.T)
# print xh
# xx1=np.zeros((x.shape[0],x.shape[0]))
# for i in range(x.shape[0]):
#     xx1[i,:]=x[i,:].dot(x.T)
# xh1=xx1.dot(w.T)
# print xh1

# x=np.random.randn(7,2)
# w=np.random.randn(2,4)
#
# h=x.dot(w)
#
# hh=h.T.dot(h)
# print hh
# xx=x.T.dot(x)
# Q = None
# left = np.dot(w.T, xx)
# for i in xrange(4):
#     right = np.dot(left, w[:, [i]])
#     Q = np.concatenate((Q, right), axis=1) if Q != None else right
# print Q

# from numpy.linalg import solve
# class mDAELMLayer(object):
#     def __init__(self, C, n_hidden, p_noise):
#         self.C = C
#         self.n_hidden = n_hidden
#         self.p_noise = p_noise
#
#     def _get_beta(self, inputX):
#         n_features = inputX.shape[1]
#         S_X = np.dot(inputX.T, inputX)
#         q_noise = np.ones((n_features, 1)) * (1 - self.p_noise)
#         q_noise[-1] = 1.
#         q1 = np.dot(q_noise, q_noise.T)
#         diag_idx = np.diag_indices(n_features - 1)
#         q1[diag_idx] = 1 - self.p_noise
#         S_X_noise1 = S_X * q1
#         W = np.random.normal(loc=0, scale=1., size=(n_features, self.n_hidden))
#         Q = None
#         left = np.dot(W.T, S_X_noise1)
#         for i in xrange(self.n_hidden):
#             right = np.dot(left, W[:, [i]])
#             Q = np.concatenate((Q, right), axis=1) if Q != None else right
#         Q *= 0.25
#         diag_idx = np.diag_indices_from(Q)
#         Q[diag_idx] *= 2
#         q2 = np.tile(q_noise.T, (n_features, 1))
#         S_X_noise2 = S_X * q2
#         P = np.dot(W.T, S_X_noise2) * 0.5
#         beta = solve(np.eye(self.n_hidden) / self.C + Q, P)
#         return beta.T
#
#     def get_train_output_for(self, inputX):
#         n_samples, n_features = inputX.shape
#         bias = np.ones((n_samples, 1), dtype=float)
#         inputX = np.hstack((inputX, bias))
#         self.beta = self._get_beta(inputX)
#         output = np.dot(inputX, self.beta)
#         # output = relu(output)
#         return output
#
#     def get_test_output_for(self, inputX):
#         n_samples, n_features = inputX.shape
#         bias = np.ones((n_samples, 1), dtype=float)
#         inputX = np.hstack((inputX, bias))
#         output = np.dot(inputX, self.beta)
#         # output = relu(output)
#         return output
#
#
# obj = mDAELMLayer(1e5, 4, 0.2)
# X = np.random.randn(7, 2)
# y = obj.get_train_output_for(X)
# print y


# def dottrans_decomp(X, splits=(1, 1)):
#     rows, cols = X.shape
#     rsplit, csplit = splits
#     assert rows % rsplit == 0 and cols % csplit == 0
#     prows = rows / rsplit
#     pcols = cols / csplit
#     out = np.zeros((rows, rows), dtype=float)
#     for j in xrange(csplit):
#         col_list = [X[i * prows:(i + 1) * prows, j * pcols:(j + 1) * pcols] for i in xrange(rsplit)]
#         for i in xrange(rsplit):
#             for k in xrange(i, rsplit):
#                 part_dot = np.dot(col_list[i], col_list[k].T)
#                 out[i * prows:(i + 1) * prows, k * prows:(k + 1) * prows] += part_dot
#                 if i != k:
#                     out[k * prows:(k + 1) * prows, i * prows:(i + 1) * prows] += part_dot.T
#     return out
#
# def dot_decomp(X, Y, splits=1):
#     Xrows, Xcols = X.shape
#     Yrows, Ycols = Y.shape
#     assert Xcols % splits == 0 and Yrows % splits == 0
#     psize = Xcols / splits
#     out = np.zeros((Xrows, Ycols), dtype=float)
#     for i in xrange(splits):
#         part_dot = np.dot(X[:, i * psize:(i + 1) * psize], Y[i * psize:(i + 1) * psize, :])
#         out += part_dot
#     return out
#
# # X=np.random.randn(10, 30)
# # a=X.dot(X.T)
# # b=dottrans_decomp(X, (5,5))
# # print np.allclose(a,b,1e-15,1e-15), a,b
#
# X=np.random.randn(10, 30)
# Y=np.random.randn(30, 20)
# a=X.dot(Y)
# b=dot_decomp(X,Y,5)
# print np.allclose(a,b,1e-15,1e-15), a,b

# import heapq

# def push_active_2d(hp, x, x_probs, y, y_probs, pos, x_active, y_active):
#     z = x[pos[0]] + y[pos[1]]
#     prob = x_probs[pos[0]] * y_probs[pos[1]]
#     heapq.heappush(hp, (z, prob, pos))
#     x_active[pos[0]] = 1
#     y_active[pos[1]] = 1
#
#
# def pop_update_2d(hp, x, x_probs, y, y_probs, x_active, y_active):
#     z, prob, pos = heapq.heappop(hp)
#     x_active[pos[0]] = 0
#     y_active[pos[1]] = 0
#     new_pos = (pos[0] + 1, pos[1])
#     if new_pos[0] < len(x) and x_active[new_pos[0]] == 0 and y_active[new_pos[1]] == 0:
#         push_active_2d(hp, x, x_probs, y, y_probs, new_pos, x_active, y_active)
#     new_pos = (pos[0], pos[1] + 1)
#     if new_pos[1] < len(y) and x_active[new_pos[0]] == 0 and y_active[new_pos[1]] == 0:
#         push_active_2d(hp, x, x_probs, y, y_probs, new_pos, x_active, y_active)
#     return z, prob
#
#
# def moving_heap_2d(x, x_probs, y, y_probs):  # x,y都是升序排列
#     hp = []
#     z = []
#     z_probs = []
#     x_active = np.zeros(len(x))
#     y_active = np.zeros(len(y))
#     push_active_2d(hp, x, x_probs, y, y_probs, (0, 0), x_active, y_active)
#     while len(hp) != 0:
#         new_z, prob = pop_update_2d(hp, x, x_probs, y, y_probs, x_active, y_active)
#         if len(z) == 0:
#             z.append(new_z)
#             z_probs.append(prob)
#             pre_z = new_z
#         else:
#             if np.allclose(pre_z, new_z):  # 相同的元素概率合并
#                 z_probs[-1] += prob
#             else:
#                 z.append(new_z)
#                 z_probs.append(prob)
#                 pre_z = new_z
#     return z, z_probs
#
#
# def push_active_md(hp, x_list, pos, x_active):
#     z = 0.
#     prob = 1.
#     for i, x in zip(pos, x_list):
#         z += x[0][i]
#         prob *= x[1][i]
#     heapq.heappush(hp, (z, prob, pos))
#     for i, active in zip(pos, x_active):
#         active[i] = 1
#
#
# def isinactive(pos, x_active):
#     for i, active in zip(pos, x_active):
#         if active[i] == 1:
#             return False
#     return True
#
#
# def isinbound(pos, x_list):
#     for i in xrange(len(pos)):
#         if pos[i] >= len(x_list[i][0]):
#             return False
#     return True
#
#
# def gen_pos(ndim):
#     pos_list = []
#
#     def dfs(lst):
#         if len(lst) == ndim:
#             pos_list.append(lst)
#         else:
#             for i in xrange(2):
#                 new_lst = copy(lst)
#                 new_lst.append(i)
#                 dfs(new_lst)
#
#     dfs([])
#     return pos_list
#
#
# def pop_update_md(hp, x_list, x_active):
#     z, prob, pos = heapq.heappop(hp)
#     for i, active in zip(pos, x_active):
#         active[i] = 0
#     all_pos = gen_pos(len(x_list))[1:-1]
#     for i, new_pos in enumerate(all_pos):
#         if isinbound(new_pos, x_list) and isinactive(new_pos, x_active):
#             push_active_md(hp, x_list, new_pos, x_active)
#     return z, prob
#
#
# def moving_heap_md(x_list):  # 升序排列
#     hp = []
#     z = []
#     z_probs = []
#     ndim = len(x_list)
#     x_active = []
#     for i in xrange(ndim):
#         x_active.append(np.zeros(len(x_list[i][0])))
#     push_active_md(hp, x_list, [0, ] * ndim, x_active)
#     while len(hp) != 0:
#         new_z, prob = pop_update_md(hp, x_list, x_active)
#         if len(z) == 0:
#             z.append(new_z)
#             z_probs.append(prob)
#             pre_z = new_z
#         else:
#             if np.allclose(pre_z, new_z):  # 相同的元素概率合并
#                 z_probs[-1] += prob
#             else:
#                 z.append(new_z)
#                 z_probs.append(prob)
#                 pre_z = new_z
#     return z, z_probs
#
#
# x = [-3, -1, 2, 6, 8]
# x_probs = [0.15, 0.25, 0.1, 0.3, 0.2]
# y = [-2, 1, 5, 8]
# y_probs = [0.2, 0.1, 0.3, 0.4]
# w = [-4, -2, 0, 4, 8]
# w_probs = [0.15, 0.25, 0.1, 0.3, 0.2]
# z, z_probs = moving_heap_md([[x, x_probs], [y, y_probs], [w, w_probs]])
# print z, z_probs
#
# z, z_probs = moving_heap_2d(x, x_probs, y, y_probs)
# u, u_probs = moving_heap_2d(z, z_probs, w, w_probs)
# print u, u_probs

# from copy import copy
# depth=10
# def dfs(lst):
#     if len(lst)==depth:
#         print lst
#     else:
#         for i in xrange(2):
#             new_lst = copy(lst)
#             new_lst.append(i)
#             dfs(new_lst)
#
# dfs([])

# def combine(ndim_list):
#     depth = len(ndim_list)
#     idx_list = []
#
#     def dfs(lst):
#         idim = len(lst)
#         if idim == depth:
#             idx_list.append(lst)
#         else:
#             for i in xrange(ndim_list[idim]):
#                 new_lst = copy(lst)
#                 new_lst.append(i)
#                 dfs(new_lst)
#
#     dfs([])
#     return idx_list
#
#
# def brute_force(x_lists, x_probs_lists):
#     ndim_list = np.array(map(lambda x: len(x), x_lists))
#     idx_list = combine(ndim_list)
#     idx_iter = iter(idx_list)
#     counter = np.zeros(len(x_lists))
#     positive_prob = 0
#     while sum(np.array(ndim_list) - np.array(counter)-1) >= 0:
#         z = 0
#         counter = idx_iter.next()
#         for num, idx in enumerate(counter):
#             z += x_lists[num][idx]
#         if z > 0:
#             z_prob = 1
#             for num, idx in enumerate(counter):
#                 z_prob *= x_probs_lists[num][idx]
#             positive_prob += z_prob
#             print positive_prob
#     return positive_prob
#
# x = [-3, -1, 2, 6, 8]
# x_probs = [0.15, 0.25, 0.1, 0.3, 0.2]
# y = [-2, 1, 5, 8]
# y_probs = [0.2, 0.1, 0.3, 0.4]
# print brute_force([x,y],[x_probs,y_probs])

def bit_add(bit, result, ndim_list):
    if bit <0:
        return
    result[bit]+=1
    if result[bit]>=ndim_list[bit]:
        result[bit]=0
        bit_add(bit - 1, result, ndim_list)


def combine_gen(ndim_list):
    nbit = len(ndim_list)
    result = [0,]*nbit
    while True:
        yield result
        bit_add(nbit - 1, result, ndim_list)
        if sum(result) == 0: break
#
# # gen=combine_gen([2,3,4,5])
# # for i,num in enumerate(gen):
# #     print i,num
#
# def brute_force(x_lists, x_probs_lists):
#     ndim_list = map(lambda x: len(x), x_lists)
#     idx_list_gen=combine_gen(ndim_list)
#     positive_prob = 0
#     for idx_list in idx_list_gen:
#         z = 0
#         for num, idx in enumerate(idx_list):
#             z += x_lists[num][idx]
#         if z > 0:
#             z_prob = 1
#             for num, idx in enumerate(idx_list):
#                 z_prob *= x_probs_lists[num][idx]
#             positive_prob += z_prob
#             # print positive_prob
#     return positive_prob
#
# def depth_first(x_lists, prob):
#     ndim = len(x_lists)
#     global positive_prob
#     positive_prob = 0
#
#     def dfs(x_sum, n_retain, idim):
#         global positive_prob
#         if idim == ndim:
#             if x_sum > 0:
#                 positive_prob += ((1-prob)**n_retain)*(prob**(ndim-n_retain))
#         else:
#             for i, x in enumerate(x_lists[idim]):
#                 new_sum = x_sum + x
#                 new_retain=n_retain+1 if i==0 else n_retain
#                 dfs(new_sum, new_retain, idim + 1)
#
#     dfs(0., 0, 0)
#     return positive_prob
#
# x = [-3, -1, 2, 6, 8]
# x_probs = [0.15, 0.25, 0.1, 0.3, 0.2]
# y = [-2, 1, 5, 8]
# y_probs = [0.2, 0.1, 0.3, 0.4]
# print depth_first([x,y],[x_probs,y_probs])
#
# x_list=[]
# x_probs_list=[]
# for i in range(25):
#     x_list.append([np.random.normal(), 0])
#     x_probs_list.append([0.8, 0.2])
#
# import time
# start=time.time()
# print depth_first(x_list, 0.2)
# end1=time.time()
# print end1-start
# print brute_force(x_list, x_probs_list)
# end2=time.time()
# print end2-end1
#
# from scipy import stats
# xx=[]
# for xl in x_list:
#     xx.append(xl[0])
# xx=np.array(xx)
# mu=np.sum(xx)*0.8
# sigma=np.sqrt(np.sum(xx**2)*0.8*0.2)
# print 1-stats.norm.cdf(-mu/sigma)

# from scipy import stats
# def relu_probs_mn(X, W, p_noise):
#     n_feature, n_hidden = W.shape
#     hidden_positive_prob = None
#     for i in xrange(n_hidden):
#         X_hidden = X * W[:, i]
#         mu = np.sum(X_hidden, axis=1) * (1. - p_noise)
#         sigma = np.sqrt(np.sum(X_hidden ** 2, axis=1) * (1. - p_noise) * p_noise)
#         col_positive_prob = 1. - stats.norm.cdf(-mu / sigma)
#         hidden_positive_prob = np.concatenate([hidden_positive_prob, col_positive_prob[:, None]], axis=1) \
#             if hidden_positive_prob is not None else col_positive_prob[:, None]
#     return hidden_positive_prob
#
# x=np.random.randn(100, 10)
# w=np.random.randn(10,20)
# print relu_probs_mn(x,w,0.2)

# def design_Q_noise(n_feature, p_noise):
#     q = np.ones((n_feature, 1)) * (1 - p_noise)
#     q[-1] = 1.
#     q_noise = np.dot(q, q.T)
#     diag_idx = np.diag_indices(n_feature - 1)
#     q_noise[diag_idx] = 1 - p_noise
#     return q_noise
#
#
# def design_Q_mn_gpu(X, W, P_positive, p_noise):
#     n_feature, n_hidden = W.shape
#     Q_noise = design_Q_noise(n_feature, p_noise)
#     Q = np.zeros((n_hidden, n_hidden), dtype=float)
#     for i in xrange(n_hidden):
#         for j in xrange(i, n_hidden):
#             if i == j:
#                 P_col = np.repeat(P_positive[:, [j]], n_feature, axis=1)
#                 S_X = np.dot(X.T, X * P_col)
#                 S_X *= Q_noise
#                 Q[i, j] = W[:, [i]].T.dot(S_X).dot(W[:, j])
#             else:
#                 P_row = np.repeat(P_positive[:, [i]].T, n_feature, axis=0)
#                 P_col = np.repeat(P_positive[:, [j]], n_feature, axis=1)
#                 S_X = np.dot(X.T * P_row, X * P_col)
#                 S_X *= Q_noise
#                 Q[i, j] = W[:, [i]].T.dot(S_X).dot(W[:, j])
#                 Q[j, i] = Q[i, j]
#     return Q
#
#
# def design_P_noise(n_feature, p_noise):
#     q = np.ones((n_feature, 1)) * (1 - p_noise)
#     q[-1] = 1.
#     q_noise = np.tile(q.T, (n_feature, 1))
#     return q_noise
#
#
# def design_P_mn_gpu(X, W, P_positive, p_noise):
#     n_batch, n_feature = X.shape
#     n_feature, n_hidden = W.shape
#     q_noise = design_P_noise(n_feature, p_noise)
#     P = np.zeros((n_feature, n_hidden), dtype=float)
#     for i in xrange(n_batch):
#         X_row = np.repeat(X[[i], :], n_feature, axis=0)
#         X_col = np.repeat(X[[i], :].T, n_feature, axis=1)
#         P_row = np.repeat(P_positive[[i], :], n_feature, axis=0)
#         P += np.dot(X_col * X_row * q_noise, W * P_row)
#     return P
#
# x=np.repeat(np.arange(7)[:,None],2,axis=1)
# x=np.hstack([x,np.ones((7,1))])
# w=np.ones((3,4))
# p=np.zeros((7,4))
# for i in range(7):
#     for j in range(4):
#         p[i,j]=(i+j)*0.05
#
# print design_Q_mn_gpu(x,w,p,0.2)
# print design_P_mn_gpu(x,w,p,0.2)

# from sklearn.externals.joblib import Parallel, delayed
#
# # def one_job(x):
# #     x+=1
# #     return x
# #
# # x=np.zeros((10,10))
# # px=Parallel(n_jobs=2, verbose=1)(
# #     delayed(one_job)(x[i*5:(i+1)*5]) for i in range(2))
# # print sum(px)
#
# def _partition_estimators(n_estimators, n_jobs):
#     """Private function used to partition estimators between jobs."""
#     # Partition estimators between jobs
#     n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs,
#                                                               dtype=np.int)
#     n_estimators_per_job[:n_estimators % n_jobs] += 1
#     starts = np.cumsum(n_estimators_per_job)
#
#     return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()
#
# print _partition_estimators(100,12)

# x=np.zeros((10,10))
# y=np.ones((10,10))
# for i in range(10):
#     tmp=copy(x[i,:])
#     # tmp = x[i, :]
#     y=y+i
#     x[i,:]=y[i,:]
#     x[i,:]=tmp
#
# print x

# from sklearn.externals.joblib import Parallel, delayed
# import time
# def func(n):
#     time.sleep(5-n)
#     return n
#
# result=Parallel(n_jobs=-1)(
#     delayed(func)(n) for n in range(4))
# print result

# from scipy.linalg import orth
#
# def orthonormalize(filters):
#     ndim = filters.ndim
#     if ndim != 2:
#         filters = np.expand_dims(filters, axis=0)
#     rows, cols = filters.shape
#     if rows >= cols:
#         orthonormal = orth(filters)
#     else:
#         orthonormal = orth(filters.T).T
#     if ndim != 2:
#         orthonormal = np.squeeze(orthonormal, axis=0)
#     return orthonormal
#
# def add_mn(X, percent=0.5):
#     retain_prob = 1. - percent
#     binomial = np.random.uniform(low=0., high=1., size=X.shape)
#     binomial = np.asarray(binomial < retain_prob, dtype=float)
#     return X * binomial
#
# def add_gs(X, scale=100., std=None):
#     if std is None:
#         Xmin, Xmax = np.min(X), np.max(X)
#         std = (Xmax - Xmin) / (2. * scale)
#     normal = np.random.normal(loc=0, scale=std, size=X.shape)
#     X += normal
#     return X
#
# w=np.random.randn(1000)
# # x=np.random.randn(1000)
# # w=orthonormalize(w)
# # w=w/1000
# result=[]
# for i in range(1000):
#     x = np.random.randn(1000)
#     # y = copy(x)
#     # y=add_gs(y, 0.5)
#     result.append(np.dot(x,w))
#
# print result
# mean=np.mean(result)
# print mean, sum(map(lambda x:x>0, result))

# import multiprocessing
# import time
#
#
# def func(msg):
#     for i in xrange(3):
#         print msg
#         time.sleep(1)
#     return "done " + msg
#
#
# if __name__ == "__main__":
#     pool = multiprocessing.Pool(processes=4)
#     result = []
#     for i in xrange(10):
#         msg = "hello %d" % (i)
#         result.append(pool.apply_async(func, (msg,)))
#     pool.close()
#     pool.join()
#     for res in result:
#         print res.get()
#     print "Sub-process(es) done."

# x=np.arange(25).reshape((5,5)).astype(theano.config.floatX)
# t1=T.matrix()
# t2=t1+1
# diag=T.arange(t2.shape[0])
# tmp=t2[diag,diag]
# w=T.set_subtensor(tmp, tmp/10)
# func=theano.function([t1],[w, t2], allow_input_downcast=True)
# print func(x)

# def design_Q_mn(X, W, P_positive):
#     n_batch, n_feature = X.shape
#     n_feature, n_hidden = W.shape
#     Q = 0.  # np.zeros((n_hidden, n_hidden), dtype=float)
#     for i in xrange(n_batch):
#         X_row = X[[i], :]
#         S_X = np.dot(X_row.T, X_row)
#         P_row = P_positive[i, :]
#         W_p = W * P_row
#         half_p = np.dot(W_p.T, S_X)
#         Q_i = np.dot(half_p, W_p)
#         Q_i_diag = np.sum(half_p * W.T, axis=1)
#         diag_idx = np.diag_indices(n_hidden)
#         Q_i[diag_idx] = Q_i_diag
#         Q += Q_i
#     return Q
#
# def design_Q_gs(X, W, P_positive):  # gs和mn相比仅仅不需要add_Q_noise
#     n_batch, n_feature = X.shape
#     n_feature, n_hidden = W.shape
#     Q = 0.  # np.zeros((n_hidden, n_hidden), dtype=float)
#     for i in xrange(n_batch):
#         X_row = X[[i], :]
#         S_X = np.dot(X_row.T, X_row)
#         P_row = P_positive[i, :]
#         W_p = W * P_row
#         half_p = np.dot(W_p.T, S_X)
#         half_nop = np.dot(W.T, S_X)
#         Q_i = []
#         for j in xrange(n_hidden):  # 要避免除以概率值,有可能很小甚至接近于0
#             half_copy = copy(half_p[j, :])  # 保留原值
#             half_p[j, :] = half_nop[j, :]  # 每次替换一行,对应于最终结果的对角线上
#             Q_i_col = np.dot(half_p, W_p[:, [j]])
#             Q_i.append(Q_i_col)
#             half_p[j, :] = half_copy  # 恢复原值
#         Q_i = np.concatenate(Q_i, axis=1)
#         Q += Q_i
#     return Q
#
# X=np.random.randn(20,10)
# W=np.random.rand(10, 15)
# P=np.random.rand(20, 15)
# print design_Q_mn(X,W,P)-design_Q_gs(X,W,P)

# import sys
# sys.path.append('/home/zfh/downloadcode/liblinear-2.1/python')
# from liblinear.python.liblinearutil import *
# # y, x = [1,-1], [[1,0,1], [-1,0,-1]]
# y,x=np.array([1,-1]), np.array([[1,0,1], [-1,0,-1]])
# prob  = problem(y.tolist(), x.tolist())
# param = parameter('-s 0 -c 4 -B 1')
# m = train(prob, param)

# import sys
# sys.path.append('/home/zfh/downloadcode/liblinear-2.1/python')
# import liblinearutil
# class Classifier_SVMlin():
#     def __init__(self, C):
#         self.C = C
#
#     def get_train_acc(self, inputX, inputy):
#         inputX, inputy = inputX.tolist(), inputy.tolist()
#         prob = liblinearutil.problem(inputy, inputX)
#         param = liblinearutil.parameter('-q -s 2 -B 1.0 -c ' + str(self.C))
#         self.clf = liblinearutil.train(prob, param)
#         _, p_acc, _ = liblinearutil.predict(inputy, inputX, self.clf)
#         return p_acc[0]
#
#     def get_test_acc(self, inputX, inputy):
#         inputX, inputy = inputX.tolist(), inputy.tolist()
#         _, p_acc, _ = liblinearutil.predict(inputy, inputX, self.clf)
#         return p_acc[0]
#
# clf=Classifier_SVMlin(1)
# y,x=np.random.randint(0,10, 100), np.random.rand(100,4)
# print clf.get_train_acc(x,y)
# print clf.get_test_acc(x,y)

# def add_Q_noise(S_X, p_noise):
#     n_feature = S_X.shape[0]
#     S_X *= (1. - p_noise) ** 2
#     diag_idx = np.diag_indices(n_feature - 1)
#     S_X[diag_idx] /= 1. - p_noise
#     S_X[-1, :] /= 1. - p_noise
#     S_X[:, -1] /= 1. - p_noise
#     return S_X
#
# x=np.arange(100).astype(float).reshape((10,10))
# print x
# y=add_Q_noise(x, 0.2)
# print x,y

# def deploy(batch, n_jobs):
#     dis_batch = (batch // n_jobs) * np.ones(n_jobs, dtype=np.int)
#     dis_batch[:batch % n_jobs] += 1
#     starts = np.cumsum(dis_batch)
#     starts = [0] + starts.tolist()
#     return starts
#
# def dottrans_decomp(X, splits=(1, 1)):
#     rows, cols = X.shape
#     rsplit, csplit = splits
#     assert rows % rsplit == 0 and cols % csplit == 0
#     prows = rows / rsplit
#     pcols = cols / csplit
#     out = np.zeros((rows, rows), dtype=float)
#     for j in xrange(csplit):
#         col_list = [X[i * prows:(i + 1) * prows, j * pcols:(j + 1) * pcols] for i in xrange(rsplit)]
#         for i in xrange(rsplit):
#             for k in xrange(i, rsplit):
#                 part_dot = np.dot(col_list[i], col_list[k].T)
#                 out[i * prows:(i + 1) * prows, k * prows:(k + 1) * prows] += part_dot
#                 if i != k:
#                     out[k * prows:(k + 1) * prows, i * prows:(i + 1) * prows] += part_dot.T
#     return out
#
#
# # compute X dot Y
# def dot_decomp_dim1(X, Y, splits=1):
#     size = X.shape[0]
#     starts = deploy(size, splits)
#     result = None
#     for i in xrange(splits):
#         tmp = np.dot(X[starts[i]:starts[i + 1]], Y)
#         result = np.concatenate([result, tmp], axis=0) if result is not None else tmp
#     return result
#
#
# def dot_decomp_dim2(X, Y, splits=1):
#     Xrows, Xcols = X.shape
#     Yrows, Ycols = Y.shape
#     assert Xcols == Yrows
#     starts = deploy(Xcols, splits)
#     out = np.zeros((Xrows, Ycols), dtype=float)
#     for i in xrange(splits):
#         part_dot = np.dot(X[:, starts[i]:starts[i + 1]], Y[starts[i]:starts[i + 1], :])
#         out += part_dot
#     return out
#
# x=np.random.randn(20,10)
# y=np.random.randn(10,5)
# r1=x.T.dot(x)
# r2=dottrans_decomp(x.T, splits=(5,5))
# print r1-r2
#
# r3=dot_decomp_dim1(x,y,5)
# r4=dot_decomp_dim2(x,y,5)
# r5=x.dot(y)
# print r5-r3, r5-r4

# def conv_out_shape(inputShape, filterShape, pad, stride, ignore_border=False):
#     batch, channel, mrows, mcols = inputShape
#     channelout, channelin, frows, fcols = filterShape
#     assert channel == channelin
#     if isinstance(pad, tuple):
#         rowpad, colpad = pad
#     else:
#         rowpad = colpad = pad
#     if isinstance(stride, tuple):
#         rowstride, colstride = stride
#     else:
#         rowstride = colstride = stride
#     mrows, mcols = mrows + 2 * rowpad, mcols + 2 * colpad
#     if not ignore_border:  # 保持下和右的边界
#         rowrem = (mrows - frows) % rowstride
#         if rowrem: mrows += rowstride - rowrem
#         colrem = (mcols - fcols) % colstride
#         if colrem: mcols += colstride - colrem
#     orow = (mrows - frows) // rowstride + 1
#     ocol = (mcols - fcols) // colstride + 1
#     return batch, channelout, orow, ocol
#
# oshape = conv_out_shape((0, 0, 96, 96),
#                         (0, 0, 5, 5),
#                         pad=2, stride=4, ignore_border=False)
# print oshape
