# coding=utf-8
from resnet import resnet50
from model_edgeconv import Model
from torchvision import models
import torch
import numpy as np
import torch.nn as nn

m = Model(class_num=395)
x = torch.rand((16, 3, 288, 144))
x1 = torch.rand((16, 3, 288, 144))
m(x, x1)
