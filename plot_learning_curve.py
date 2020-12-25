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

weight_path = 'model_weights/adam_lr_0.001/resnet101'
weight_list = os.listdir(weight_path)
weight_list.sort(key=lambda x: int(x.split('_')[0][5:]))
train_acc_list, test_acc_list = [], []
for weight in weight_list:
    train_acc, test_acc = re.findall(r"\d+\.?\d*", weight)[-2:]
    train_acc_list.append(float(train_acc))
    test_acc_list.append(float(test_acc))
print('max train acc: {} in eopch: {}'.format(max(train_acc_list), np.argmax(train_acc_list)))
print('max test acc: {} in eopch: {}'.format(max(test_acc_list), np.argmax(test_acc_list)))

plot_curve(
    train_acc_list,
    test_acc_list,
    "Training accuracy",
    "Test accuracy",
    "Epochs",
    "Accuracy",
    title=weight_path.split('/')[-2] + '_' + weight_path.split('/')[-1]
)
plt.plot(train_acc_list, label='Training accuracy')
plt.plot(test_acc_list, label='Test accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(weight_path.split('/')[-2] + '_' + weight_path.split('/')[-1], fontsize=15)
plt.legend()
plt.show()
