# -*- coding: utf-8 -*-

"""
@File: model_predict.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 2020/12/15
"""

import os
import numpy as np
import glob
from transfering_model import TransferingModel
from ensemble_model import EnsembleClassificationModel
from torchvision import transforms
import json
from torch.utils.data import DataLoader
from street_view_dataset import create_dataset


data_dir = 'data'
train_data_path = os.path.join(data_dir, 'train')
validation_data_path = os.path.join(data_dir, 'validation')
test_data_path = os.path.join(data_dir, 'test_total')
class_to_idx = {'negative': 0, 'positive': 1}

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_augs = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    normalize
])


def confidence_interval(score, mean, std):
    print("%s: %.2f (+/- %.2f)" % (score, mean, std))


def batch_evaluation(model, test_data_path_list):
    temp = []
    for test_data_path in test_data_path_list:
        test_ds = create_dataset(test_data_path)
        test_iter = DataLoader(test_ds, 128)
        result_dict = model.evaluation(test_iter, "all", False)
        print(result_dict)
        temp.append(list(result_dict.values()))
    temp = np.array(temp)
    for i, score in enumerate(result_dict.keys()):
        confidence_interval(score, temp[:, i].mean(), temp[:, i].std())


if __name__ == '__main__':
    import time
    # for config in glob.glob(r"C:\Level4Project\SuZhouTest\model_weights\ensemble_config\*.json"):
    #     print(os.path.basename(config))
    config = r"C:\Level4Project\model\ensemble_config.json"
    with open(config) as f:
        weights = json.load(f)
    test_data_path_list = glob.glob(r'C:\StreetViewImageClassification\data\test_total\*')
    #
    model_dict = {name: TransferingModel(name.split("_")[0], weight).model for name, weight in weights.items()}
    start = time.time()
    ensemble_model = EnsembleClassificationModel(model_dict)
    # base_path = "./model_weights"
    # model = TransferingModel("resnet101", os.path.join(base_path, "epoch26_loss0.0009_trainacc0.992_testacc0.990.pth"))
    batch_evaluation(ensemble_model, test_data_path_list)
    print(time.time() - start)