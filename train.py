# -*- coding: utf-8 -*-

"""
@File: train.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 2021/01/06
"""

import glob
import os
from transfering_model import TransferingModel
from street_view_dataset import StreetViewDataset
from torch.utils.data import DataLoader

def create_dataset(
        path,
        pos="positive",
        neg="negative"
):
    positive_path_list = glob.glob(os.path.join(path, pos, "*.png"))
    negative_path_list = glob.glob(os.path.join(path, neg, "*.png"))
    data_path_list = positive_path_list + negative_path_list
    label_list = [1] * len(positive_path_list) + [0] * len(negative_path_list)
    return StreetViewDataset(
        data_path_list,
        label_list
    )

if __name__ == "__main__":
    train_data_path = "C:/Level4Project/data/train"
    valid_data_path = "C:/Level4Project/data/valid"

    train_dataset = create_dataset(train_data_path)
    valid_dataset = create_dataset(valid_data_path)

    train_iter = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_iter = DataLoader(valid_dataset, batch_size=32)

    model_names = ["resnet101", "resnet152", "densenet161", "densenet201"]
    optimizer = "sgd"
    for lr in [0.01, 0.001]:
        for model_name in model_names:
            print("============%s is training============" % model_name)
            model_save_path = os.path.join("C:/Level4Project/model", "lr_{}".format(lr), model_name)
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            # load pretrained model
            # model_weight_path = glob.glob(os.path.join(model_save_path, "epoch40_*"))[0]
            model = TransferingModel(model_name)
            model.fit(
                lr=lr,
                optimizer=optimizer,
                num_epochs=60,
                checkpoint_epochs=None,
                train_iter=train_iter,
                valid_iter=valid_iter,
                model_save_path=model_save_path,
                is_plot=False
            )