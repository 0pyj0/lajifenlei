import torch
import torchvision
#-*- coding:utf-8 _*-

import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
from cv2 import transform
from torch import classes

import dataset
import numpy as np

from Image import Image
from args import args
from build_net import make_model
from transform import get_transforms
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, get_optimizer, save_checkpoint
#model = torchvision.models.vgg16()
from utils.utils import device

model = torchvision.models.resnet101()
state_dict = torch.load("E:/data0/search/qlmx/clover/garbage/res_16_288_last1/model_cur.pth")
# print(state_dict)
model.load_state_dict(state_dict, False)
model.eval()
def prediect(img_path):
    net=torch.load('modelcur.pth')
    net=net.to(device)
    torch.no_grad()
    img=Image.open(img_path)
    img=transform(img).unsqueeze(0)
    img_ = img.to(device)
    outputs = net(img_)
    _, predicted = torch.max(outputs, 1)
    # print(predicted)
    print('this picture maybe :',classes[predicted[0]])
if __name__ == '__main__':
    prediect('D:\垃圾目录\厨余垃圾\菜梗菜叶/baidu000000.jpg')

#x = torch.rand(1, 3, 256, 256)
#ts = torch.jit.trace(model, x)
#torch.save(state_dict, "modelcur.pth", _use_new_zipfile_serialization=False)
#print(ts)
#ts.save('modelcur.py')
