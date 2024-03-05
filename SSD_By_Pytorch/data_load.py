import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from d2l import torch as d2l
import cv2 as cv
import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from model import TinySSD
import xml.etree.ElementTree as ET
import  PIL
import cv2
'''
#读取数据，返回图像和标签
def read_data(is_train=True):
    """读取检测数据集中的图像和标签"""

    csv_data = pd.read_csv('../data_banana/label.csv')
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    print('数据形状：'+str(csv_data))

    j=1
    for img_name, target in csv_data.iterrows():
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（索引为0）
        images.append(torchvision.io.read_image('../data_banana/images/'+str(img_name)))
        targets.append(list(target))
        if j==1:
            print(target.shape)
            print(type(target))
            print(target[0])
            print(target[1])
            j = j+1

    return images, torch.tensor(targets).unsqueeze(1) / 256
'''


# voc_labels为VOC数据集中20类目标的类别名称
voc_labels = ('mask', 'face', )

# 创建label_map字典，用于存储类别和类别索引之间的映射关系。比如：{1：'aeroplane'， 2：'bicycle'，......}
label_map = {k: v  for v, k in enumerate(voc_labels)}
# VOC数据集默认不含有20类目标中的其中一类的图片的类别为background，类别索引设置为0
#label_map['background'] = 0

# 将映射关系倒过来，{类别名称：类别索引}
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping


# 解析xml文件，最终返回这张图片中所有目标的标注框及其类别信息，以及这个目标是否是一个difficult目标
def read_data(is_train=True):
    # 解析xml

    annotation_path = '../data_mask/MaskDataset/VOC/train/Annotations/train_'
    images = list()
    labels = list()
    for i in range(1,1201):
        path = annotation_path+str(i)+'.xml'
        tree = ET.parse(path)
        root = tree.getroot()

        img_name = root.find('filename').text
        #print(img_name)
        height = int(root.find('size').find('height').text)
        width = int(root.find('size').find('width').text)
        #img = cv2.imread('../data_mask/MaskDataset/VOC/train/JPEGImages/' + str(img_name))

        #cv2.imshow('pic', img)

        #img = cv.resize(img, (512,512), fx=2, fy=2,interpolation=PIL.Image.BICUBIC)
        img = torchvision.io.read_image('../data_mask/MaskDataset/VOC/train/JPEGImages/' + str(img_name))/ 255.0
        resize = transforms.Resize([512, 512],interpolation=PIL.Image.BICUBIC)
        img = resize(img)
        transform1 = transforms.Compose([

            transforms.Normalize(mean=(0.48, 0.45, 0.44), std=(0.27, 0.26, 0.265))
        ]
        )
        img = transform1(img)


        # 遍历xml文件中所有的object，前面说了，有多少个object就有多少个目标
        for object in root.iter('object'):
            images.append(img)


            label = object.find('name').text.lower().strip()
            if label not in label_map:
                continue

            bbox = object.find('bndbox')

            xmin = int(bbox.find('xmin').text) * (512 / width)/512
            ymin = int(bbox.find('ymin').text) * (512 / height)/512
            xmax = int(bbox.find('xmax').text) * (512 / width)/512
            ymax = int(bbox.find('ymax').text) * (512 / height)/512


            # 存储
            label = pd.Series([float(label_map[label]), xmin, ymin, xmax, ymax])
            labels.append(list(label))

    annotation_path = '../data_mask/MaskDataset/VOC/test/Annotations/val_'

    for i in range(1, 401):
        path = annotation_path + str(i) + '.xml'
        tree = ET.parse(path)
        root = tree.getroot()

        img_name = root.find('filename').text
        # print(img_name)
        height = int(root.find('size').find('height').text)
        width = int(root.find('size').find('width').text)
        # img = cv2.imread('../data_mask/MaskDataset/VOC/train/JPEGImages/' + str(img_name))

        # cv2.imshow('pic', img)

        # img = cv.resize(img, (512,512), fx=2, fy=2,interpolation=PIL.Image.BICUBIC)
        img = torchvision.io.read_image('../data_mask/MaskDataset/VOC/test/JPEGImages/' + str(img_name)) / 255.0
        resize = transforms.Resize([512, 512], interpolation=PIL.Image.BICUBIC)
        img = resize(img)
        transform1 = transforms.Compose([

            transforms.Normalize(mean=(0.48, 0.45, 0.44), std=(0.27, 0.26, 0.265))
        ]
        )
        img = transform1(img)

        # 遍历xml文件中所有的object，前面说了，有多少个object就有多少个目标
        for object in root.iter('object'):
            images.append(img)

            label = object.find('name').text.lower().strip()
            if label not in label_map:
                continue

            bbox = object.find('bndbox')

            xmin = int(bbox.find('xmin').text) * (512 / width) / 512
            ymin = int(bbox.find('ymin').text) * (512 / height) / 512
            xmax = int(bbox.find('xmax').text) * (512 / width) / 512
            ymax = int(bbox.find('ymax').text) * (512 / height) / 512

            # 存储
            label = pd.Series([float(label_map[label]), xmin, ymin, xmax, ymax])
            labels.append(list(label))

    annotation_path = '../data_mask/MaskDataset/VOC/train/Annotations/train_auged_'

    for i in range(1, 1201):
        path = annotation_path + str(i) + '.xml'
        tree = ET.parse(path)
        root = tree.getroot()

        img_name = root.find('filename').text+'.jpg'
        #print(img_name)
        height = int(root.find('size').find('height').text)
        width = int(root.find('size').find('width').text)
        # img = cv2.imread('../data_mask/MaskDataset/VOC/train/JPEGImages/' + str(img_name))

        # cv2.imshow('pic', img)

        # img = cv.resize(img, (512,512), fx=2, fy=2,interpolation=PIL.Image.BICUBIC)
        img = torchvision.io.read_image('../data_mask/MaskDataset/VOC/train/JPEGImages/' + str(img_name))/ 255.0
        resize = transforms.Resize([512, 512], interpolation=PIL.Image.BICUBIC)
        img = resize(img)
        transform1 = transforms.Compose([
            
            transforms.Normalize(mean=(0.48, 0.45, 0.44), std=(0.27, 0.26, 0.265))
        ]
        )
        img = transform1(img)

        # 遍历xml文件中所有的object，前面说了，有多少个object就有多少个目标
        for object in root.iter('object'):

            images.append(img)

            label = object.find('name').text.lower().strip()
            if label not in label_map:
                continue

            bbox = object.find('bndbox')

            xmin = int(bbox.find('xmin').text) * (512 / width)/512
            ymin = int(bbox.find('ymin').text) * (512 / height)/512
            xmax = int(bbox.find('xmax').text) * (512 / width)/512
            ymax = int(bbox.find('ymax').text) * (512 / height)/512

            # 存储
            label = pd.Series([float(label_map[label]), xmin, ymin, xmax, ymax])
            labels.append(list(label))


    # 返回包含图片标注信息的字典
    return images, torch.tensor(labels).unsqueeze(1)
'''
VOCdevkit/VOC2007
├── Annotations
├── ImageSets
├── JPEGImages
├── SegmentationClass
└── SegmentationObject
'''

#对自己的数据集，需要重定义Dataset
class Dataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)

#对数据进行batch的划分 迭代器
def load_data(batch_size):
    """加载香蕉检测数据集"""
    train_iter = torch.utils.data.DataLoader(Dataset(is_train=True),
                                             batch_size, shuffle=True)
    #val_iter = torch.utils.data.DataLoader(Dataset(is_train=False),
                                           #batch_size)

    return train_iter, None