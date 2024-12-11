import os, pathlib
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.sparse
import torch
from pylab import *
from scipy.signal import convolve2d
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve
import re


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def image_name_sort(image_names):
    num_list = []
    for image_name in image_names:
        if image_name.endswith('png') or image_name.endswith('jpg'):
            a = image_name.split('.')[-2]
            # num = int(a[len(no_num_image_name):])
            num = int(re.sub('\D', '', a))
            num_list.append(num)
    # print('old_image_names:', image_names, len(image_names))
    ids = np.argsort(num_list)
    new_image_names = []
    for i in range(len(num_list)):
        new_image_names.append(image_names[ids[i]])
    # print('new_image_names:', new_image_names), len(new_image_names)
    return new_image_names


def save_output(image_dir, image_path, y_pred, dsize=None, save_dir=r'./'):


    save_path = pathlib.PurePath(save_dir, pathlib.PurePath(image_path.strip(image_dir + os.sep)))
    save_path = str(pathlib.PurePath(save_path))
    # exit()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if dsize is None:
        image_shape = cv.imread(image_path).shape
        y_pred = cv.resize(y_pred, dsize=(image_shape[1], image_shape[0]), interpolation=cv.INTER_CUBIC)
    else:
        y_pred = cv.resize(y_pred, dsize=(dsize[1], dsize[0]), interpolation=cv.INTER_CUBIC)
    cv.imwrite(save_path, y_pred)


def flow_to_image_from_dir(image_dir, save_dir, dsize=None):
    print('开始进行步态轮廓图提取...')

    os.makedirs(save_dir, exist_ok=True)

    t1 = time.time()
    ImgTypeList = ['jpg', 'JPG', 'bmp', 'png', 'jpeg', 'rgb', 'tif']
    for dir_path, dir_names, image_names in os.walk(image_dir):
        # print(dir_path, dir_names, image_names)
        image_path_list = [os.path.join(dir_path, image_name) for image_name in image_names if
                           image_name.split('.')[-1] in ImgTypeList]
        image_path_list = image_name_sort(image_path_list)
        # print('image_path_list:', image_path_list)
        if len(image_path_list) > 0:
            # 开始转换
            print('开始转换:%s' % dir_path)
            for i in range(len(image_path_list)):

                if i < len(image_path_list) - 1:
                    image_path = image_path_list[i]
                    image_path_next = image_path_list[i + 1]
                    image1 = cv.imread(image_path, flags=0)  # (h,w)
                    image2 = cv.imread(image_path_next, flags=0)
                    # print(image1.shape, image2.shape)
                    flows = cv.calcOpticalFlowFarneback(image1,
                                                        image2,
                                                        None,
                                                        pyr_scale=0.5,

                                                        levels=3,
                                                        winsize=4,
                                                        iterations=3,
                                                        poly_n=7,
                                                        poly_sigma=1.1,

                                                        flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN)
                    flows_img = flow_to_image(flows)
                    # cv.imshow('flows_img', flows_img)
                    # cv.waitKey()
                    # y_pred = (np.clip(y_pred, 0, 1) * 255).astype('uint8')  # (h,w)
                    if flows_img is not None:
                        save_output(image_dir, image_path_list[i], flows_img, dsize, save_dir)
    print('已完成全部，共耗时:%s' % (time.time() - t1))

iden_ty = ['026', '027', '028', '029', '030', '031', '032', '033', '034', '035']


image_dir = r'3_Cut'
save_dir = r'5_op'
dsize = None
flow_to_image_from_dir(image_dir, save_dir, dsize=None)

