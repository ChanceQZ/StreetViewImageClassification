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
from torchvision.datasets import ImageFolder

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
        test_image_folder = ImageFolder(test_data_path, transform=test_augs)
        test_iter = DataLoader(test_image_folder, 128)
        result_dict = model.evaluation(test_iter, "all", None, True)
        # print(result_dict)
        temp.append(list(result_dict.values()))
    temp = np.array(temp)
    for i, score in enumerate(result_dict.keys()):
        confidence_interval(score, temp[:, i].mean(), temp[:, i].std())


if __name__ == '__main__':
    import time
    # for config in glob.glob(r"C:\Level4Project\SuZhouTest\model_weights\ensemble_config\*.json"):
    #     print(os.path.basename(config))
    config = r"C:\Level4Project\SuZhouTest\model_weights\ensemble_config\多模型集成_0.01.json"
    with open(config) as f:
        weights = json.load(f)
    test_data_path_list = glob.glob('data/test_total/*')
    #
    model_dict = {name: TransferingModel(name.split("_")[0], weight).model for name, weight in weights.items()}
    start = time.time()
    ensemble_model = EnsembleClassificationModel(model_dict)
    # base_path = r"C:\Level4Project\SuZhouTest\model_weights\\cosineannealing\lr_0.01"
    # model = TransferingModel("resnet152", os.path.join(base_path, "resnet152\epoch58_loss0.0149_trainacc0.995_validacc0.996.pth"))
    batch_evaluation(ensemble_model, test_data_path_list)
    print(time.time() - start)