# -*- coding: utf-8 -*-

"""
@File: sampling.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 3/23/2021
"""
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit as SSS


class DataSplit:
    def __init__(
            self, X, y,
            training_sampling_size=0.6,
            validation_sampling_size=0.2,
            test_sampling_size=0.2,
            sampling_times_size=(10, 1000)
    ):
        """
        Dividing data into training sampling set and union test sampling set which
        combined with validation sampling set and test sampling set.

        Most importantly! Traning, validation and test dataset are all sampled from the above sampling sets.
        :param X: Data
        :param y: Label
        :param training_sampling_size: Size of training sampling dataset, 0<i<1
        :param validation_sampling_size: Size of validation sampling dataset, 0<i<1
        :param test_sampling_size: Size of test sampling dataset, 0<i<1
        :param sampling_times_size: Sampling times and size of validation and test set
        """
        self.sampling_set_split(validation_sampling_size, test_sampling_size)

        self.validation_sampling_list = self.repeat_sampling(
            X[self._validation_sampling_idx],
            y[self._validation_sampling_idx],
            self.split,
            *sampling_times_size
        )

        self.test_sampling_list = self.repeat_sampling(
            X[self._test_sampling_idx],
            y[self._test_sampling_idx],
            self.split,
            *sampling_times_size
        )

    @staticmethod
    def repeat_sampling(X, y, split_method, repeat_times, sampling_size):
        """
        Independently and repeatly sampling in each time. Notably, data may be repeat in different times,
        but sampling without replacement in each time.

        :return: Sampling result (not index of data).
        """
        sampling_list = []
        for i in range(repeat_times):
            _, idx = split_method(X, y, sampling_size, i)
            sampling_list.append(X[idx])
        return sampling_list

    @staticmethod
    def split(X, y, split_size, random_state=1212):
        """
        Dividing data into 2 parts, that is training set and test set in logical.
        :param X: Data
        :param y: Label
        :return: training set index and test set index
        """
        sss = SSS(
            n_splits=1,
            test_size=split_size,
            random_state=random_state
        )
        return next(sss.split(X, y))

    def sampling_set_split(self, validation_sampling_size, test_sampling_size):
        self._training_sampling_idx, self._union_test_sampling_idx = self.split(
            X, y,
            validation_sampling_size + test_sampling_size
        )

        self._validation_sampling_idx, self._test_sampling_idx = self.split(
            X[self._union_test_sampling_idx],
            y[self._union_test_sampling_idx],
            test_sampling_size / (validation_sampling_size + test_sampling_size)
        )


if __name__ == "__main__":
    X = np.random.randint(0, 10, (10000, 4))
    y = np.random.randint(0, 2, 10000)
    ds = DataSplit(X, y)
    print(len(ds.validation_sampling_list))
    print(len(ds.test_sampling_list))
