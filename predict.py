# -*- coding: utf-8 -*-

"""
@File: baseline_test.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 12/21/20
"""

import os
import glob
from transfering_model import TransferingModel
from torchvision import transforms
from torch import nn, optim
from PIL import Image
from ensemble_model import EnsembleClassificationModel
import json
from torchvision.datasets import ImageFolder
from utils import PredictDataset, multi_processing_copyfile
from torch.utils.data import DataLoader

if __name__ == "__main__":

    with open("ensamble_config.json") as f:
        weights = json.load(f)
    model_dict = {name: TransferingModel(name.split("_")[0], weight).model for name, weight in weights.items()}
    ensemble_model = EnsembleClassificationModel(model_dict)

    file_list = glob.glob("E:/SZ_bridge_StreetViews/*.png")
    predict_dataset = PredictDataset(file_list)
    data_loader = DataLoader(predict_dataset, batch_size=256)
    predict_list = []
    for data in data_loader:
        predict_list.extend(ensemble_model.predict(data))

    noise_barrier_list = [file for flag, file in zip(predict_list, file_list) if flag == 1]
    dst_path = "E:/Suzhou_noise_barrier_predict"
    multi_processing_copyfile(noise_barrier_list, dst_path)