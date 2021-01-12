"""
model with relation
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

        self.emb1 = nn.Sequential(
            nn.Conv2d(in_channels=45, out_channels=5, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(5),
            nn.ReLU()
        )
        self.emb2 = nn.Sequential(
            nn.Conv2d(in_channels=45, out_channels=5, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(5),
            nn.ReLU()
        )
        self.em3 = nn.Sequential(
            nn.Conv2d(in_channels=45, out_channels=5, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(5),
            nn.ReLU()
        )
        self.channel = nn.Sequential(
                nn.Conv2d(in_channels=2048, out_channels=258,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(258),
                nn.ReLU()
            )
        self.W_channel = nn.Sequential(
            nn.Conv2d(in_channels=259, out_channels=32,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1)
        )

    def forward(self, x1, x2, training=True):
        if training:
            x = torch.cat((x1, x2), 0)
        else:
            x = x1
        x = self.base(x)

        b, c, h, w = x.size()
        xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)
        x_em1 = self.emb1(xc).squeeze(-1).sum(dim=1)
        x_em2 = self.emb2(xc).squeeze(-1).sum(dim=1)

        x_s = x_em1[0].expand(1, c) - x_em2[0].expand(1, c).t()
        x_s = x_s.unsqueeze(0)
        for i in range(1, b):
            x_t = x_em1[i].expand(1, c) - x_em2[i].expand(1, c).t()
            x_t = x_t.unsqueeze(0)
            x_s = torch.cat((x_s, x_t), 0)

        x_sj = self.channel(x_s.unsqueeze(-1))

        x_t = self.em3(xc)
        x_t = torch.mean(x_t, dim=1, keepdim=True)

        x_t = torch.cat((x_sj, x_t), 1)
        W_yc = self.W_channel(x_t).transpose(1, 2)
        x_o = F.sigmoid(W_yc) * x

        feat = self.avgpool(x_o + x)
        feat = feat.view(feat.size(0), -1)
        feat_bn = self.bn(feat)
        out = self.classifier(feat_bn)

        if training:
            return feat, out
        else:
            return F.normalize(feat_bn)
