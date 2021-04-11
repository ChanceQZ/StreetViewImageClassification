# -*- coding: utf-8 -*-

"""
@File: plot_learning_curve.py
@Author: Chance (Qian Zhen)
@Description: Plot learning rate based on weights log.
@Date: 2020/12/17
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_curve

weight_path = "C:/Level4Project/SuZhouTest/model_weights/cosineannealing/lr_0.01/densenet161"
weight_list = os.listdir(weight_path)
weight_list.sort(key=lambda x: int(x.split("_")[0][5:]))
train_acc_list, test_acc_list = [], []
train_loss_list, test_loss_list = [], []
for weight in weight_list:
    train_loss, test_loss, train_acc, test_acc = re.findall(r"\d+\.?\d*", weight)[1:]
    train_loss_list.append(float(train_loss))
    test_loss_list.append(float(test_loss))
    train_acc_list.append(float(train_acc))
    test_acc_list.append(float(test_acc))

print("min train loss: {} in eopch: {}".format(min(train_loss_list), np.argmin(train_loss_list)))
print("min test loss: {} in eopch: {}".format(min(test_loss_list), np.argmin(test_loss_list)))
print("max train acc: {} in eopch: {}".format(max(train_acc_list), np.argmax(train_acc_list)))
print("max test acc: {} in eopch: {}".format(max(test_acc_list), np.argmax(test_acc_list)))

plot_curve(
    train_loss_list,
    test_loss_list,
    "Training loss",
    "Test loss",
    "Epochs",
    "Loss",
    title=weight_path.split("/")[-2] + "_" + weight_path.split("/")[-1]
)


# plot_curve(
#     train_acc_list,
#     test_acc_list,
#     "Training accuracy",
#     "Test accuracy",
#     "Epochs",
#     "Accuracy",
#     title=weight_path.split("/")[-2] + "_" + weight_path.split("/")[-1]
# )
