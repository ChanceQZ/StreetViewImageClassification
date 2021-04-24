# -*- coding: utf-8 -*-

"""
@File: ensemble_model.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 2020/12/16
"""
import torch
from torch import nn, Tensor
from scipy import stats
from torchvision import transforms
from base_model import BaseModel
from utils import check_device, calculate_classification_score


class EnsembleClassificationModel(BaseModel):
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

    @torch.no_grad()
    def evaluation(
            self, eval_iter,
            score: str,
            TTA_mode=False
    ):
        y_true_list, y_pred_list = [], []
        # TTA condition
        if TTA_mode:
            # each data group has n groups data which are augmented by TTA
            for data_group in eval_iter:
                y_pred_voting_list = []
                # each data item has X and y (data and label)
                for data_item in data_group:
                    X = data_item[0]

                    y_pred = self.predict(X)  # map probability into category
                    y_pred_voting_list.append(y_pred)
                y_true = data_group[0][1]
                y_true_list.extend(y_true.tolist())
                # calculate mode based on prediction of different groups
                y_pred_list.extend(list(stats.mode(y_pred_voting_list)[0][0]))

        else:
            for X, y in eval_iter:
                y_true_list.extend(y.tolist())
                y_pred_list.extend(self.predict(X))
        score = calculate_classification_score(y_true_list, y_pred_list, score)

        return score
