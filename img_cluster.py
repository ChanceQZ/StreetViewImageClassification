# -*- coding: utf-8 -*-

"""
@File: img_cluster.py
@Author: Chance (Qian Zhen)
@Description: Cluster the sound barrier street image to infer classes of them.
@Date: 2020/12/7
"""

import os
import cv2
import numpy as np
from utils import center_crop_img, normalize_img
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

IMG_SIZE = 200


def get_best_cluster(data, min_cluster, max_cluster):
    n, max_score, labels = 0, 0.0, None
    for i in range(min_cluster, max_cluster + 1):
        kmeans = KMeans(n_clusters=i).fit(data)
        score = metrics.silhouette_score(data, kmeans.labels_, metric='euclidean')
        if score > max_score:
            max_score = score
            n = i
            labels = kmeans.labels_
    return n, np.array(labels)


def show_images(imgs, num_rows=3, num_cols=6, scale=4):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes


def get_data(path, center_cropping=True, normalization=True):
    img_path_list = [os.path.join(path, file) for file in os.listdir(path)]
    img_data = []
    for img_path in img_path_list:
        if center_cropping:
            img = center_crop_img(cv2.imread(img_path), IMG_SIZE)
        else:
            img = cv2.resize(cv2.imread(img_path), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        if img is None:
            continue
        if normalization:
            img = normalize_img(img)
        img_data.append(img.reshape(-1))
    return np.array(img_data)


if __name__ == '__main__':
    sb_crop = 'Data/SoundBarrier_old/SB_Crop'
    img_data = get_data(sb_crop, center_cropping=False, normalization=False)
    print(img_data.shape)

    pca = PCA(n_components=300)
    pca_img_data = pca.fit_transform(img_data)
    print(pca_img_data.shape)

    n, labels = get_best_cluster(pca_img_data, 5, 8)
    print('the best cluster number: %d' % n)
    print(labels)

    imgs = []
    plot_col_num = 24
    for i in np.unique(labels):
        idxs = np.where(labels == i)[0]
        for j in range(plot_col_num):
            imgs.append(img_data[idxs[j]].reshape(IMG_SIZE, IMG_SIZE, 3))
    show_images(imgs, n, plot_col_num)
    plt.show()
