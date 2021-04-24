# -*- coding: utf-8 -*-

"""
@File: base_model.py
@Author: Chance (Qian Zhen)
@Description: Baseline of neural network model based on PyTorch.
@Date: 2020/12/19
"""

import time
import torch
from scipy import stats
import numpy as np
import tqdm
from torch import nn, optim
from typing import Generator, Union
from torchvision import transforms
from utils import check_device, plot_curve, calculate_classification_score, InvalidArguments
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class BaseModel:
    def __init__(self, model: nn.Module, device: str = "gpu") -> None:
        self.model = model
        self.device = check_device(device)
        self._loss_criterion = None
        self._optimizer = None
        self._scheduler = None

    def fit(
            self,
            train_iter: Generator,
            validation_iter: Generator,
            lr: float = 0.01,
            loss_criterion: Union[str, nn.Module] = "cross_entropy",
            optimizer: Union[str, optim.Optimizer] = "sgd",
            scheduler: Union[str, optim.Optimizer] = None,
            num_epochs: int = 30,
            checkpoint_epochs: int = None,
            model_save_path: str = None,
            is_plot: bool = False
    ) -> None:
        print("training on %s" % self.device)
        self.model = self.model.to(self.device)

        if train_iter is None or validation_iter is None:
            raise InvalidArguments("Must input training data and validation data!")

        if isinstance(loss_criterion, str):
            if loss_criterion.lower() == "cross_entropy":
                self._loss_criterion = torch.nn.CrossEntropyLoss()
        else:
            self._loss_criterion = loss_criterion

        if isinstance(optimizer, str):
            if optimizer.lower() == "sgd":
                self._optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.001)
        else:
            self._optimizer = optimizer

        if isinstance(scheduler, str):
            if scheduler.lower() == "cawr":
                self._scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            else:
                self._scheduler = scheduler

        # Learning rate scheduler
        train_acc_list, valid_acc_list = [], []

        # If checkpoint_epochs is not None, it means that model will continue to train from last weights.
        epoch_iter = range(num_epochs, num_epochs + checkpoint_epochs) if checkpoint_epochs else range(num_epochs)
        for epoch in epoch_iter:
            sample_count, batch_count, train_loss_sum, train_acc_sum, start = 0, 0, 0.0, 0.0, time.time()
            self.model.train()

            for idx, (X, y) in tqdm.tqdm(enumerate(train_iter)):
                X = X.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(X)
                loss = self._loss_criterion(y_pred, y)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if scheduler is not None:
                    self._scheduler.step(epoch + idx / len(train_iter))

                train_loss_sum += loss.cpu().item()
                train_acc_sum += (y_pred.argmax(dim=1).cpu() == y.cpu()).sum()

                # batch_count += 1
                sample_count += y.shape[0]
            mean_valid_acc, mean_valid_loss = self.evaluation(validation_iter, "Accuracy")

            mean_train_loss = train_loss_sum / sample_count
            mean_train_acc = train_acc_sum / sample_count

            train_acc_list.append(mean_train_loss)
            valid_acc_list.append(mean_valid_acc)
            print("epoch %d, train_loss %.4f, valid_loss %.4f, train acc %.3f, valid_total acc %.3f, time %.1f sec"
                  % (epoch + 1, mean_train_loss, mean_valid_loss, mean_train_acc, mean_valid_acc, time.time() - start))

            if model_save_path is not None:
                torch.save(
                    self.model.state_dict(),
                    "%s/epoch%d_trainloss%.3f_validloss%.3f_trainacc%.3f_validacc%.3f.pth"
                    % (model_save_path,
                       epoch + 1,
                       mean_train_loss,
                       mean_valid_loss,
                       mean_train_acc,
                       mean_valid_acc)
                )

        if is_plot is True:
            plot_curve(
                train_acc_list,
                valid_acc_list,
                "Training accuracy",
                "Validation accuracy",
                "Epochs",
                "Accuracy"
            )

    @torch.no_grad()
    def evaluation(
            self, eval_iter: Generator,
            score: str,
            TTA_mode=False
    ):
        """
        evaluation
        :param eval_iter:
        :param score:
        :param TTA_mode: if TTA mode is turned on, loss critirion
        :return: benchmarks(acc, recall, precision and f1) [TTA mode], or benchmarks and loss
        """
        self.model.eval()

        y_true_list, y_pred_list = [], []
        eval_loss_sum = 0
        sample_count = 0
        # TTA condition
        if TTA_mode:
            # each data group has n groups data which are augmented by TTA
            for data_group in eval_iter:
                y_pred_voting_list = []
                # each data item has X and y (data and label)
                for data_item in data_group:
                    X = data_item[0]

                    y_pred = self.model(X.to(self.device)).argmax(dim=1).cpu().tolist()  # map probability into category
                    y_pred_voting_list.append(y_pred)
                y_true = data_group[0][1]
                y_true_list.extend(y_true.tolist())
                # calculate mode based on prediction of different groups
                y_pred_list.extend(list(stats.mode(y_pred_voting_list)[0][0]))

            score = calculate_classification_score(y_true_list, y_pred_list, score)
            return score

        else:
            for X, y in eval_iter:
                y_pred_prob = self.model(X.to(self.device))

                y_true_list.extend(y.tolist())
                y_pred_list.extend(y_pred_prob.argmax(dim=1).cpu().tolist())
                eval_loss_sum += self._loss_criterion(y_pred_prob, y.to(self.device))
                sample_count += y.shape[0]

            mean_eval_loss = eval_loss_sum / sample_count
            score = calculate_classification_score(y_true_list, y_pred_list, score)

            return score, mean_eval_loss
