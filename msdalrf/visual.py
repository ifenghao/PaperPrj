# coding:utf-8
import numpy as np
from matplotlib import pylab
import os
import time

username = 'zhufenghao'
interpolation = 'nearest'
img_count = 0

__all__ = ['save_beta_fc', 'save_beta_lrf', 'save_beta_lrfchs', 'save_map_lrf']


def addPad(map2d, padWidth):
    row, col = map2d.shape
    hPad = np.zeros((row, padWidth))
    map2d = np.hstack((hPad, map2d, hPad))
    vPad = np.zeros((padWidth, col + 2 * padWidth))
    map2d = np.vstack((vPad, map2d, vPad))
    return map2d


def squareStack(map3d):
    mapNum = map3d.shape[0]
    row, col = map3d.shape[1:]
    side = int(np.ceil(np.sqrt(mapNum)))
    lack = side ** 2 - mapNum
    map3d = np.vstack((map3d, np.zeros((lack, row, col))))
    map2ds = [addPad(map3d[i], 1) for i in range(side ** 2)]
    return np.vstack([np.hstack(map2ds[i:i + side])
                      for i in range(0, side ** 2, side)])


def save_beta_fc(beta):
    save_path = os.path.join('/home', username, 'images', time.asctime())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    n_feature, n_hidden = beta.shape
    num = int(np.ceil(n_hidden / 100.))
    img_side = int(np.sqrt(n_feature))
    img = beta.T.reshape((-1, img_side, img_side))
    pylab.figure()
    pylab.gray()
    for i in xrange(num):
        one_img = img[:100]
        img = img[100:]
        pylab.imshow(squareStack(one_img), interpolation=interpolation)
        pic_path = os.path.join(save_path, str(i) + '.png')
        pylab.savefig(pic_path)
    pylab.close()


def save_beta_lrf(beta, dir, name):
    global img_count
    save_path = os.path.join('/home', username, 'images', dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pic_path = os.path.join(save_path, str(img_count) + name + '.png')
    img_count += 1
    size, channels = beta.shape
    beta = beta.T.reshape((channels, int(np.sqrt(size)), int(np.sqrt(size))))
    pylab.figure()
    pylab.gray()
    pylab.imshow(squareStack(beta), interpolation=interpolation)
    pylab.savefig(pic_path)
    pylab.close()


def save_beta_lrfchs(beta, chs, dir, name):
    global img_count
    save_path = os.path.join('/home', username, 'images', dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pic_path = os.path.join(save_path, str(img_count) + name + '.png')
    img_count += 1
    size, channels = beta.shape
    size /= chs
    beta = np.split(beta.T, chs, axis=1)
    beta = np.concatenate(beta, axis=0)
    beta = beta.reshape((chs * channels, int(np.sqrt(size)), int(np.sqrt(size))))
    pylab.figure()
    pylab.gray()
    pylab.imshow(squareStack(beta), interpolation=interpolation)
    pylab.savefig(pic_path)
    pylab.close()


def save_map_lrf(map, dir, name):
    global img_count
    save_path = os.path.join('/home', username, 'images', dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pic_path = os.path.join(save_path, str(img_count) + name + '.png')
    img_count += 1
    num = len(map)
    pylab.figure()
    pylab.gray()
    for i in xrange(num):
        pylab.subplot(1, num, i + 1)
        pylab.imshow(squareStack(map[i]), interpolation=interpolation)
    pylab.savefig(pic_path)
    pylab.close()