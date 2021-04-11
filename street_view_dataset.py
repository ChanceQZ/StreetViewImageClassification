# -*- coding: utf-8 -*-

"""
@File: street_view_dataset.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 2021/04/11
"""

import cv2
import albumentations as A
from torchvision import transforms
from torch.utils.data import Dataset

CROP_SIZE = 400
IMAGE_SIZE = 224
ROTATE_ANGLE = 30
transform = A.Compose([
    A.RandomCrop(CROP_SIZE, CROP_SIZE),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=ROTATE_ANGLE, p=0.5),
    A.GaussNoise(p=0.5)
])


class StreetViewDataset(Dataset):
    def __init__(
            self,
            data_path_list,
            label_list,
            transform,
            test_mode=False,
            TTA=False,
            TTA_count=5
    ):
        self.data_path_list = data_path_list
        self.label_list = label_list
        self.transform = transform
        self.test_mode = test_mode
        self.TTA = TTA
        self.TTA_count = TTA_count

        self.as_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


    def __getitem__(self, idx):
        img = cv2.imread(self.data_path_list[idx])
        label = self.label_list[idx]

        if not self.test_mode:
            augments = self.transform(image=img)
            return self.as_tensor(augments['image']), label
        else:
            if not self.TTA:
                return self.as_tensor(img), label
            else:
                augments_list = []
                for _ in range(self.TTA_count):
                    augments = self.transform(image=img)
                    augments_list.append([self.as_tensor(augments['image']), label])
                return augments_list

    def __len__(self):
        return len(self.data_path_list)


if __name__ == "__main__":
    import glob
    from torch.utils.data import DataLoader

    positive_path_list = glob.glob("./data/train/positive/*.png")
    negative_path_list = glob.glob("./data/train/negative/*.png")
    data_path_list = positive_path_list + negative_path_list
    label_list = [1] * len(positive_path_list) + [0] * len(negative_path_list)

    train_ds = StreetViewDataset(data_path_list, label_list, transform, test_mode=True, TTA=True)
    data_iter = DataLoader(train_ds, batch_size=16, shuffle=True)
    iters = 0
    for data_group in data_iter:
        print(len(data_group))
        # each data group has n groups data which are augmented by TTA
        # for data_item in data_group:
        #     # each data item has X and y (data and label)
        #     X, y = data_item
        #     print(X.shape)
        #     print(y.shape)
        #     break
        iters += 1
        break
    # print(iters)
