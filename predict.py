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
    data_loader = DataLoader(predict_dataset, batch_size=350, num_workers=12, pin_memory=True)
    predict_list = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            predict_list.extend(ensemble_model.predict(data))
    return predict_list


if __name__ == "__main__":
    ensamble_config = "ensamble_config.json"
    src_folder = "C:/multi_temporal"
    dst_folder = "C:/multi_temporal/predict"

    print("Creating ensamble model")

    with open(ensamble_config) as f:
        weights = json.load(f)
    model_dict = {name: TransferingModel(name.split("_")[0], weight).model for name, weight in weights.items()}
    ensemble_model = EnsembleClassificationModel(model_dict)
    cities = ["img"]
    for city in cities:
        start = time.time()
        print("Identifying {} street view images".format(city))
        file_list = glob.glob(os.path.join(src_folder, city, "*.png"))
        dst_path = os.path.join(dst_folder, city)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        predict_list = predict(file_list)
        noise_barrier_list = [file for flag, file in zip(predict_list, file_list) if flag == 1]
        multi_processing_copyfile(noise_barrier_list, dst_path)
        torch.cuda.empty_cache()

        print("Time cost of {} is {}".format(city, time.time() - start))
