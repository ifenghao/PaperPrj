# coding:utf-8

import cv2
from skimage import feature
import numpy as np
from matplotlib import pylab
import utils


def canny_edge(X, p_border=0.05):
    assert X.ndim == 3  # 对灰度图像和多通道图像都可以,后者需要先转化为2维灰度图像
    X = X[0, :, :] if X.shape[0] == 1 else np.max(X, axis=0)
    X = X.astype(np.float)
    X -= np.min(X)  # 最大化图像的动态范围到0~255
    X *= 255 / np.max(X)
    X = np.round(X).astype(np.uint8)
    edge = cv2.Canny(X, 185, 225)
    edge /= 255
    edger, edgec = edge.shape
    mask = np.zeros_like(edge, dtype=np.uint8)
    startr, endr = np.round(edger * p_border), np.round(edger * (1 - p_border))
    startc, endc = np.round(edgec * p_border), np.round(edgec * (1 - p_border))
    mask[startr:endr, startc:endc] = 1
    edge *= mask
    return edge

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
    edge = cv2.Canny(X, 50, 205)
    edge /= 255
    return edge

def canny_edge2(X, p_border=0.05):
    assert X.ndim == 3  # 对灰度图像和多通道图像都可以,后者需要先转化为2维灰度图像
    X = X[0, :, :] if X.shape[0] == 1 else np.max(X, axis=0)
    X = X.astype(np.float)
    X -= np.min(X)  # 最大化图像的动态范围到-1~1
    X /= np.max(X)
    rows, cols = X.shape
    mask = np.zeros(X.shape, dtype=bool)
    startr, endr = np.round(rows * p_border), np.round(rows * (1 - p_border))
    startc, endc = np.round(cols * p_border), np.round(cols * (1 - p_border))
    mask[startr:endr, startc:endc] = True
    edge = feature.canny(X, sigma=1., low_threshold=0.6, high_threshold=0.8, mask=mask)
    return edge

tr_X, te_X, tr_y, te_y = utils.load.cifar(onehot=False)
images=[]
edges=[]
for rand in np.random.randint(0,len(tr_X),400):
    img=tr_X[rand]
    edge = canny_edge_opencv(img)
    images.append(img[0])
    edges.append(edge)
images=np.array(images)
edges=np.array(edges)
disp_img=utils.visual.squareStack(images)
disp_edge=utils.visual.squareStack(edges)

pylab.figure()
pylab.gray()
pylab.imshow(disp_img, interpolation=None)
pylab.figure()
pylab.gray()
pylab.imshow(disp_edge, interpolation=None)
pylab.show()



