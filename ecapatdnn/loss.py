from dataclasses import dataclass
import math

import torch
import torch.nn.functional as fn
from torch import nn

@dataclass
class AAAMSoftmax_config:
    embed_dim:int
    n_class: int
    m:int
    s:int

class ClassificationLoss(nn.Module):
    def __init__(self, config:AAAMSoftmax_config) -> None:
        super().__init__()
        self.fc =nn.Linear(config.embed_dim, config.n_class, bias=False)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, label):
        x = self.fc(x)
        loss = self.loss(x, label)
        return loss

class AAMSoftmax(nn.Module):
    def __init__(self, config:AAAMSoftmax_config) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.n_class, config.embed_dim))
        nn.init.xavier_normal_(self.weight)
        self.ce = nn.CrossEntropyLoss()

        self.m = config.m
        self.s = config.s
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m


    def forward(self, x, label):
        cosine = fn.linear(fn.normalize(x), fn.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        loss = self.ce(output, label)
        return loss