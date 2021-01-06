# -*- coding: utf-8 -*-

"""
@File: utils.py
@Author: Chance
@Description: This is utility package contained some useful functions or classes.
@Date: 2020/12/7
"""
import os
import glob
import cv2
import torch
import time
import random
from PIL import Image
from multiprocessing import Pool, Manager
import xml.etree.ElementTree as ET
from shutil import copyfile
import matplotlib.pyplot as plt
from typing import Generator
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
from torch.nn import Module
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def get_coords_from_label(label_file):
    """
    Get coordinates of bounding box from label. Only support (.xml) file generated from Labelimg.
    :param label_file: (.xml) file generated from Labelimg
    :return: Coordinate tuple.
    """
    tree = ET.parse(open(label_file))
    root = tree.getroot()
    coords = []
    for obj in root.iter("object"):
        xmlbox = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (int(xmlbox.find("xmin").text),
                                  int(xmlbox.find("ymin").text),
                                  int(xmlbox.find("xmax").text),
                                  int(xmlbox.find("ymax").text))
        coords.append((xmin, ymin, xmax, ymax))
    return coords


def crop_img(img_path, lbl_path, output_path):
    """
    Crop image by labeling file, which contains the top-left and low-right coordinates.
    :param img_path: The path of origin image files.
    :param lbl_path: The path of label files.
                        (*.txt, contained the top-left and low-right coordinates, sep=" ")
    :param output_path: The path of images cropped.
    :return: None
    """
    img_file_list = os.listdir(img_path)
    lbl_file_list = os.listdir(lbl_path)

    img_endswith = img_file_list[0].split(".")[-1]
    cnt = 1
    for lbl_file in lbl_file_list:
        coords = get_coords_from_label(os.path.join(lbl_path, lbl_file))
        for coord in coords:
            xmin, ymin, xmax, ymax = coord

            img = cv2.imread(os.path.join(img_path, lbl_file.replace("xml", img_endswith)))
            cropped = img[ymin:ymax, xmin:xmax]

            output_file = os.path.join(output_path, "{}.{}".format(cnt, img_endswith))
            cv2.imwrite(output_file, cropped)
            cnt += 1


def normalize_img(img):
    """
    Normalize image based on z-score.
    :param img:  Numpy ndarray type.
    :return:  Numpy ndarray type.
    """
    channel_mean = img.mean(axis=0, keepdims=True).mean(axis=1, keepdims=True)
    channel_std = img.std(axis=0, keepdims=True).mean(axis=1, keepdims=True)
    return (img - channel_mean) / channel_std


def center_crop_img(img, size=100):
    """
    Crop image on the center.
    :param img: Numpy ndarray type.
    :param size: int
    :return: Numpy ndarray type.
    """
    if img.shape[0] < size or img.shape[1] < size:
        return
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    half_size = size // 2
    try:
        return img[center_y - half_size: center_y + half_size, center_x - half_size: center_x + half_size]
    except:
        return img[:size, :size]


def evaluate_accuracy(data_iter, net, device=None):
    """
    Evaluate test_total dataset accuracy.
    :param data_iter: data generator, containing X and y with batch size
    :param net: model
    :param device: if the device is not assigned, will use the device of net
    :return: the test_total dataset accuracy
    """
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()  # it"s necessary to trigger to evaluation mode
            y_hat = net(X.to(device)).argmax(dim=1)
            acc_sum += (y_hat == y.to(device)).float().sum().cpu().item()
            net.train()  # trigger to training mode
            n += y.shape[0]
    return acc_sum / n


class InvalidArguments(Exception):
    pass


def check_device(device):
    if (device == "gpu" or device == "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def calculate_classification_score(y_true, y_pred, score):
    if score.lower() == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif score.lower() == "precision":
        return precision_score(y_true, y_pred)
    elif score.lower() == "recall":
        return recall_score(y_true, y_pred)
    elif score.lower() == "f1_score":
        return f1_score(y_true, y_pred)


def plot_curve(arr1, arr2, label1, label2, xlabel, ylabel, **kwargs):
    plt.plot(arr1, label=label1)
    plt.plot(arr2, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(kwargs.get("title"))
    plt.legend()
    plt.show()


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    normalize
])


def default_loader(path):
    img_pil = Image.open(path)
    img_tensor = preprocess(img_pil)
    return img_tensor


class PredictDataset(Dataset):
    def __init__(self, file_paths, loader=default_loader):
        # 定义好 image 的路径
        self.file_paths = file_paths
        self.loader = loader

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        return self.loader(file_path)

    def __len__(self):
        return len(self.file_paths)


def multi_processing_copyfile(
        src_file_list: list,
        dst_path: str,
        process_num: int = 4
) -> None:
    start = time.time()
    pool = Pool(process_num)
    q = Manager().Queue()

    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    for src in src_file_list:
        dst = os.path.join(dst_path, os.path.basename(src))
        pool.apply_async(copyfile, args=(src, dst))
        q.put(src)

    pool.close()
    pool.join()

    print("Time total cost is %.3f" % (time.time() - start))


if __name__ == "__main__":
    img_path = "../Data/SoundBarrier_old/img"
    lbl_path = "../Data/SoundBarrier_old/label_xml"
    output_path = "../Data/SoundBarrier_old/SB_Crop"

    crop_img(img_path, lbl_path, output_path)
