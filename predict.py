# -*- coding: utf-8 -*-

"""
@File: baseline_test.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 12/21/20
"""

import os
import glob
import time
import json
from transfering_model import TransferingModel
from ensemble_model import EnsembleClassificationModel
from utils import PredictDataset, multi_processing_copyfile
from torch.utils.data import DataLoader

def predict(file_list):
    predict_dataset = PredictDataset(file_list)
    data_loader = DataLoader(predict_dataset, batch_size=256)
    predict_list = []
    for data in data_loader:
        predict_list.extend(ensemble_model.predict(data))
    return predict_list

if __name__ == "__main__":
    ensamble_config = "ensamble_config.json"
    src_folder = "C:/jjpq_81Cities/街景图片_全国"
    dst_folder = "C:/noise_barrier_predict"

    print("Creating ensamble model")


    with open(ensamble_config) as f:
        weights = json.load(f)
    model_dict = {name: TransferingModel(name.split("_")[0], weight).model for name, weight in weights.items()}
    ensemble_model = EnsembleClassificationModel(model_dict)

    for city in ["1线", "2线", "3线", "4线", "5线", "港澳台"]:
        start = time.time()
        print("Identifying street view images")
        file_list = glob.glob(os.path.join(src_folder, city, "*.png"))
        predict_list = predict(file_list)

        print("Copying noise barrier samples")
        noise_barrier_list = [file for flag, file in zip(predict_list, file_list) if flag == 1]
        multi_processing_copyfile(noise_barrier_list, dst_folder)

        print("Time cost of {} is {}".format(city, time.time() - start))
