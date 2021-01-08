# -*- coding: utf-8 -*-

"""
@File: transfering_modelv2.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 2020/12/20
"""

import torch
from torch import nn, optim
from baseline import Model
from torchvision import models, transforms
from typing import Generator
from utils import InvalidArguments
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class TransferingModel(Model):
    def __init__(
            self,
            model_name: str,
            model_weight_path: str = None,
            output_num: int = 2,
            train_augs: transforms.Compose = None,
            test_augs: transforms.Compose = None,
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

        if hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, output_num)
        elif hasattr(model, "classifier"):
            model.classifier = nn.Linear(model.classifier.in_features, output_num)

        super().__init__(model, device)

        if model_weight_path:
            self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device))

        # Initialize data augmenter.
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if "inception" in model_name:
            self.model.aux_logits = False
            crop_size = 299
            test_transform_size = 331
        else:
            crop_size = 224
            test_transform_size = 256

        if train_augs:
            self.train_augs = train_augs
        else:
            self.train_augs = transforms.Compose([
                transforms.RandomResizedCrop(size=crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        if test_augs:
            self.test_augs = test_augs
        else:
            self.test_augs = transforms.Compose([
                transforms.Resize(size=test_transform_size),
                transforms.CenterCrop(size=crop_size),
                transforms.ToTensor(),
                normalize
            ])

    def fit(
            self,
            lr: float = 0.01,
            batch_size: int = 16,
            optimizer: str = "sgd",
            num_epochs: int = 30,
            checkpoint_epochs: int = None,
            train_data_path: str = None,
            validation_data_path: str = None,
            train_iter: Generator = None,
            validation_iter: Generator = None,
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

        if (train_data_path is None and validation_data_path is None) \
                and (train_iter is None and validation_iter is None):
            raise InvalidArguments("Training/validation data path or generator must be provided!")

        if train_data_path and validation_data_path:
            # Create image folder
            train_image_folder = ImageFolder(train_data_path, transform=self.train_augs)
            validation_image_folder = ImageFolder(validation_data_path, transform=self.test_augs)

            # Create dataset generator
            train_iter = DataLoader(train_image_folder, batch_size, shuffle=True)
            validation_iter = DataLoader(validation_image_folder, batch_size)

        loss_criterion = torch.nn.CrossEntropyLoss()

        super().fit(
            train_iter,
            validation_iter,
            lr,
            loss_criterion,
            optimizer,
            num_epochs,
            checkpoint_epochs,
            model_save_path,
            is_plot
        )
