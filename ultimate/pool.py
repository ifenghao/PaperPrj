# coding:utf-8
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
import numpy as np
import utils

__all__ = ['pool_op', 'pool_fn', 'pool_l2_fn', 'fp_fn', 'fp_l2_fn']


def pool_op(xnp, psize, pool_type, mode, args=None):
    if pool_type not in ('pool', 'fp'):
        raise NotImplementedError
    if mode not in ('max', 'sum', 'avg', 'l2'):
        raise NotImplementedError
    if pool_type == 'pool':
        if mode in ('max', 'sum', 'avg'):
            xnp = pool_fn(xnp, psize, mode=mode) if args is None \
                else pool_fn(xnp, psize, mode=mode, **args)
        else:
            xnp = pool_l2_fn(xnp, psize) if args is None \
                else pool_l2_fn(xnp, psize, **args)
    else:
        if mode in ('max', 'sum', 'avg'):
            xnp = fp_fn(xnp, psize, mode=mode) if args is None \
                else fp_fn(xnp, psize, mode=mode, **args)
        else:
            xnp = fp_l2_fn(xnp, psize) if args is None \
                else fp_l2_fn(xnp, psize, **args)
    return xnp


def pool_theanofn(pool_size, ignore_border, stride, pad, mode):
    xt = T.tensor4()
    poolx = pool_2d(xt, pool_size, ignore_border=ignore_border, st=stride, padding=pad, mode=mode)
    pool = theano.function([xt], poolx, allow_input_downcast=True)
    return pool


def pool_fn(xnp, pool_size, ignore_border=False, stride=None, pad=(0, 0), mode='max'):
    if mode == 'avg': mode = 'average_exc_pad'
    if isinstance(pool_size, (int, float)): pool_size = (int(pool_size), int(pool_size))
    pool = pool_theanofn(pool_size, ignore_border, stride, pad, mode)
    return pool(xnp)


def pool_l2_fn(xnp, pool_size, ignore_border=False, stride=None, pad=(0, 0)):
    if isinstance(pool_size, (int, float)): pool_size = (int(pool_size), int(pool_size))
    pool = pool_theanofn(pool_size, ignore_border, stride, pad, 'sum')
    xnp = np.square(xnp)
    xnp = pool(xnp)
    xnp = np.sqrt(xnp)
    return xnp


def fp_theanofn(pool_ratio, constant, overlap, mode):
    xt = T.tensor4()
    fpx = utils.pool.fp(xt, pool_ratio, constant, overlap, mode)
    fp = theano.function([xt], fpx, allow_input_downcast=True)
    return fp


def fp_fn(xnp, pool_ratio, constant=0.5, overlap=True, mode='max'):
    fp = fp_theanofn(pool_ratio, constant, overlap, mode)
    return fp(xnp)


def fp_l2_fn(xnp, pool_ratio, constant=0.5, overlap=True):
    fp = fp_theanofn(pool_ratio, constant, overlap, 'sum')
    xnp = np.square(xnp)
    xnp = fp(xnp)
    xnp = np.sqrt(xnp)
    return xnp
