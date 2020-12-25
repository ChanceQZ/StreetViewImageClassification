# -*- coding: utf-8 -*-

"""
@File: sample.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 2020/12/8
"""
import os
import glob
import torch
import random
from PIL import Image
from shutil import copyfile
from torch import nn
from torchvision import transforms
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
augs = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    normalize
])

net = models.resnet101(pretrained=True)
net.fc = nn.Linear(2048, 2)
net.load_state_dict(torch.load('ResNet101_weights/epoch17_loss0.0017_trainacc0.988_testacc0.990.pth'))
net = net.to(device)

net.eval()
origin_dir = 'K:/jjpq_81Cities/街景图片_全国'
citys = ['1线']
target_dir = 'Data/SoundBarrierPredict'

for city in citys:
    print('#########%s start###########' % city)
    img_files = glob.glob(os.path.join(origin_dir, city) + '/*/*.png')
    n = len(img_files)
    random.shuffle(img_files)
    for img_file in img_files[:n // 2]:
        if not img_file.endswith('png'):
            continue
        print(img_file)
        try:
            image = Image.open(img_file)
        except Exception as e:
            print(e)
            print('Open Error! Try again!')
            continue
        else:
            with open(os.path.join(target_dir, city + '.txt'), 'a') as f:
                f.write(img_file)

                image = augs(image)
                image = image.to(device)
                y_hat = net(image.view((1,) + image.shape))
                if y_hat.argmax(dim=1).item() == 1:
                    copyfile(img_file, os.path.join(target_dir, city, img_file.split('\\')[-1]))
                    f.write(',1线')
                else:
                    f.write(',0')
                f.write('\n')