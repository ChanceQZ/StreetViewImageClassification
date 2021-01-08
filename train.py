# -*- coding: utf-8 -*-

"""
@File: train.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 2021/1/6
"""

import glob
import os
from transfering_model import TransferingModel

if __name__ == "__main__":
    train_data_path = "data/train"
    validation_data_path = "data/validation"
    model_names = ["resnet101", "resnet152", "densenet161", "densenet201"]
    optimizer = "sgd"
    for lr in [0.0001]:
        for model_name in model_names:
            print("============%s is training============" % model_name)
            model_save_path = os.path.join("E:/Model_weights/model_weights", "lr_{}".format(lr), model_name)
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            model_weight_path = glob.glob(os.path.join(model_save_path, "epoch40_*"))[0]
            model = TransferingModel(model_name, model_weight_path)
            model.fit(
                lr=lr,
                batch_size=16,
                optimizer=optimizer,
                num_epochs=70,
                checkpoint_epochs=30,
                train_data_path=train_data_path,
                validation_data_path=validation_data_path,
                model_save_path=model_save_path,
                is_plot=True
            )