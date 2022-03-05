#-*- coding:utf-8 _*-
import random
import math
import torch

from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms

class Resize(object):#调整大小
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))
            #Image.crop(left, up, right, below)left：与左边界的距离，up：上边界，
        img = img.resize(self.size, self.interpolation)

        return img

class RandomRotate(object):#随机旋转
    def __init__(self, degree, p=0.5):
        self.degree = degree
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
        return img

class RandomGaussianBlur(object):#随机高斯模糊


    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
# 创建高斯模糊滤镜，radius-模糊半径，通过改变半径的值，得到不同强度的高斯模糊图像，越大越糊
        return img

def get_train_transform(mean, std, size):
    train_transform = transforms.Compose([
        Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.RandomCrop(size),#裁剪

        transforms.RandomHorizontalFlip(),#依概率水平翻转，默认值为0.5
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),#标准化，先减均值，再除以标准差
    ])
    return train_transform

def get_test_transform(mean, std, size):
    return transforms.Compose([
        Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.CenterCrop(size),#在图片的中间区域进行裁剪
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_transforms(input_size=224, test_size=224, backbone=None):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if backbone is not None and backbone in ['pnasnet5large', 'nasnetamobile']:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    transformations = {}
    transformations['val_train'] = get_train_transform(mean, std, input_size)
    transformations['val_test'] = get_test_transform(mean, std, test_size)
    return transformations

