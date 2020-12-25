# -*- coding: utf-8 -*-

"""
@File: ensemble_model.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 2020/12/16
"""
import torch
from torch import nn, Tensor
from torchvision import transforms
from baseline import Model
from utils import check_device


class EnsembleClassificationModel(Model):
    def __init__(self, model_dict, device: str = "gpu"):
        self.model_dict = model_dict
        self.device = check_device(device)

    def fit(self, *arg, **kwargs):
        print("This version only support to ensemble trained model")

    def predict(self, X: Tensor = None) -> Tensor:
        ensemble_pred = []
        for i, (name, model) in enumerate(self.model_dict.items()):
            model.eval()
            with torch.no_grad():
                model = model.to(self.device)
                X = X.to(self.device)
                y_pred = model(X)

            temp = torch.zeros_like(y_pred)
            temp[range(len(temp)), y_pred.argmax(dim=1)] = 1
            ensemble_pred.append(temp)

        return torch.stack(ensemble_pred, 0).cpu().sum(dim=0).argmax(dim=1).tolist()