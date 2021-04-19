# -*- coding: utf-8 -*-

"""
@File: transfering_model.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 2020/12/20
"""

import torch
from torch import nn, optim
from base_model import BaseModel
from torchvision import models, transforms
from typing import Generator
from utils import InvalidArguments
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from street_view_dataset import StreetViewDataset


class TransferingModel(BaseModel):
    def __init__(
            self,
            model_name: str,
            model_weight_path: str = None,
            output_num: int = 2,
            device: str = "gpu"
    ) -> None:
        is_pretrained = False if model_weight_path else True

        if model_name.lower() == "resnet101":
            model = models.resnet101(pretrained=is_pretrained)
        elif model_name.lower() == "resnet152":
            model = models.resnet152(pretrained=is_pretrained)
        elif model_name.lower() == "densenet161":
            model = models.densenet161(pretrained=is_pretrained)
        elif model_name.lower() == "densenet201":
            model = models.densenet201(pretrained=is_pretrained)
        elif model_name.lower() == "inception_v3":
            model = models.inception_v3(pretrained=is_pretrained)

        # Different backbone networks have different nicknames of the same layer.
        if hasattr(model, "fc"):  # Resnet and Inception
            model.fc = nn.Linear(model.fc.in_features, output_num)
        elif hasattr(model, "classifier"):  # Densenet
            model.classifier = nn.Linear(model.classifier.in_features, output_num)

        super().__init__(model, device)

        if model_weight_path:
            self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device))

    def fit(
            self,
            lr: float = 0.01,
            optimizer: str = "sgd",
            scheduler: str = None,
            num_epochs: int = 30,
            checkpoint_epochs: int = None,
            train_iter: Generator = None,
            valid_iter: Generator = None,
            model_save_path: str = None,
            is_plot: bool = True,
    ) -> None:

        if hasattr(self.model, "fc"):
            output_params = list(map(id, self.model.fc.parameters()))
            lastlyr_params = self.model.fc.parameters()
        elif hasattr(self.model, "classifier"):
            output_params = list(map(id, self.model.classifier.parameters()))
            lastlyr_params = self.model.classifier.parameters()

        feature_params = filter(lambda p: id(p) not in output_params, self.model.parameters())

        if optimizer.lower() == "sgd":
            optimizer = optim.SGD([{"params": feature_params},
                                   {"params": lastlyr_params, "lr": lr * 10}],
                                  lr=lr, weight_decay=0.001)
        elif optimizer.lower() == "adam":
            optimizer = optim.Adam([{"params": feature_params},
                                    {"params": lastlyr_params, "lr": lr * 10}],
                                   lr=lr, weight_decay=0.001)



        loss_criterion = torch.nn.CrossEntropyLoss()

        super().fit(
            train_iter,
            valid_iter,
            lr,
            loss_criterion,
            optimizer,
            scheduler,
            num_epochs,
            checkpoint_epochs,
            model_save_path,
            is_plot
        )
