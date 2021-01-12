"""
baseline model
"""

import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50
import torch.nn.functional as F
import numpy



class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class Model(nn.Module):
    def __init__(self, class_num):
        super(Model, self).__init__()

        dim = 2048
        self.base = resnet50(pretrained=True)
        self.bn = nn.BatchNorm1d(dim)
        self.classifier = nn.Linear(dim, class_num, bias=False)
        self.bn.apply(weights_init_kaiming)
        self.bn.bias.requires_grad_(False)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2, training=True):
        if training:
            x = torch.cat((x1, x2), 0)
        else:
            x = x1
        x = self.base(x)

        feat = self.avgpool(x)
        feat = feat.view(feat.size(0), -1)
        feat_bn = self.bn(feat)
        out = self.classifier(feat_bn)

        if training:
            return feat, out
        else:
            return F.normalize(feat_bn), F.normalize(feat)
