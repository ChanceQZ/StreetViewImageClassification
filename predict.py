# -*- coding: utf-8 -*-

"""
@File: baseline_test.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 12/21/20
"""

import torch
import os
import glob
import time
import json
from tqdm import tqdm
from transfering_model import TransferingModel
from ensemble_model import EnsembleClassificationModel
from utils import PredictDataset, multi_processing_copyfile
from torch.utils.data import DataLoader


def predict(file_list):
    predict_dataset = PredictDataset(file_list)
    data_loader = DataLoader(predict_dataset, batch_size=256, num_workers=12, pin_memory=True)
    predict_list = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            predict_list.extend(ensemble_model.predict(data))
    return predict_list


if __name__ == "__main__":
    ensamble_config = "C:\Level4Project\model\ensemble_config.json"

    print("Creating ensamble model")

    with open(ensamble_config) as f:
        weights = json.load(f)
    model_dict = {name: TransferingModel(name.split("_")[0], weight).model for name, weight in weights.items()}
    ensemble_model = EnsembleClassificationModel(model_dict)

    with open(r"C:/Level4Project/total_coordinates.csv") as f:
        img_path_list = [row.strip().split(",")[-1] for row in f.readlines()]

    pos_dst_folder = r"C:\Level4Project\total_img_predict\positive"
    neg_dst_folder = r"C:\Level4Project\total_img_predict\negative"

    if not os.path.exists(pos_dst_folder):
        os.makedirs(pos_dst_folder)

    if not os.path.exists(neg_dst_folder):
        os.makedirs(neg_dst_folder)

    predict_list = predict(img_path_list)
    pos_noise_barrier_list = [file for flag, file in zip(predict_list, img_path_list) if flag == 1]
    multi_processing_copyfile(pos_noise_barrier_list, pos_dst_folder)

    neg_noise_barrier_list = [file for flag, file in zip(predict_list, img_path_list) if flag == 0]
    multi_processing_copyfile(neg_noise_barrier_list, neg_dst_folder)

    torch.cuda.empty_cache()
