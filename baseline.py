# -*- coding: utf-8 -*-

"""
@File: model.py
@Author: Chance (Qian Zhen)
@Description: Baseline of neural network model based on PyTorch.
@Date: 2020/12/19
"""

import time
import torch
from torch import nn, optim
from typing import Generator, Union
from utils import check_device, plot_curve, calculate_classification_score, InvalidArguments

class Model:
    def __init__(self, model: nn.Module, device: str = "gpu") -> None:
        self.model = model
        self.device = check_device(device)

    def fit(
            self,
            train_iter: Generator,
            validation_iter: Generator,
            lr: float = 0.01,
            loss_criterion: Union[str, nn.Module] = "cross_entropy",
            optimizer: Union[str, optim.Optimizer] = "sgd",
            num_epochs: int = 30,
            model_save_path: str = None,
            is_plot: bool = False
    ) -> None:
        print("training on %s" % self.device)
        self.model = self.model.to(self.device)

        if train_iter is None or validation_iter is None:
            raise InvalidArguments("Must input training data and validation data!")

        if isinstance(loss_criterion, str):
            if loss_criterion.lower() == "cross_entropy":
                loss_criterion = torch.nn.CrossEntropyLoss()

        if isinstance(optimizer, str):
            if optimizer.lower() == "sgd":
                optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.001)

        train_acc_list, valid_acc_list = [], []

        for epoch in range(num_epochs):
            n, batch_count, train_loss_sum, train_acc_sum, start = 0, 0, 0.0, 0.0, time.time()
            self.model.train()

            for X, y in train_iter:
                X = X.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(X)
                loss = loss_criterion(y_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.cpu().item()
                train_acc_sum += (y_pred.argmax(dim=1).cpu() == y.cpu()).sum()

                batch_count += 1
                n += y.shape[0]
            valid_acc = self.evaluation(validation_iter, "Accuracy")

            train_acc_list.append(train_acc_sum / n)
            valid_acc_list.append(valid_acc)
            print("epoch %d, loss %.4f, train acc %.3f, valid_total acc %.3f, time %.1f sec"
                  % (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n, valid_acc, time.time() - start))

            if model_save_path:
                torch.save(
                    self.model.state_dict(),
                    "%s/epoch%d_loss%.4f_trainacc%.3f_validacc%.3f.pth"
                    % (model_save_path,
                       epoch + 1,
                       train_loss_sum / batch_count,
                       train_acc_sum / n,
                       valid_acc)
                )

        if is_plot:
            plot_curve(
                train_acc_list,
                valid_acc_list,
                "Training accuracy",
                "Validation accuracy",
                "Epochs",
                "Accuracy"
            )

    def predict(self, X: torch.Tensor) -> list:
        self.model = self.model.to(self.device)
        X = X.to(self.device)
        self.model.eval()

        with torch.no_grad():
            return self.model(X).argmax(dim=1).cpu().tolist()

    def evaluation(self, test_iter: Generator, score: str) -> dict:
        y_true, y_pred = [], []
        for X, y in test_iter:
            y_true.extend(y.tolist())
            y_pred.extend(self.predict(X))
        return calculate_classification_score(y_true, y_pred, score)
