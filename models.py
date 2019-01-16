import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyll.base import TorchModel, calc_out_shape


class SingleCell(TorchModel):
    def __init__(self, fc_units=2048, dropout=0.5, num_classes=209, input_shape=None):
        super(SingleCell, self).__init__()
        assert input_shape
        in_c = input_shape[0]
        in_h = input_shape[1]
        in_w = input_shape[2]
        
        self.block1 = nn.Sequential(
            # input downsampling
            nn.Conv2d(in_c, 32, kernel_size=3, stride=2, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True)
        )
        # global average pooling
        h, w = calc_out_shape(in_h, in_w, self.block1)        
        self.gap1 = nn.AvgPool2d(kernel_size=(h, w))
        h, w = calc_out_shape(h, w, self.block2)        
        self.gap2 = nn.AvgPool2d(kernel_size=(h, w))
        h, w = calc_out_shape(h, w, self.block3)        
        self.gap3 = nn.AvgPool2d(kernel_size=(h, w))
        h, w = calc_out_shape(h, w, self.block4)        
        self.gap4 = nn.AvgPool2d(kernel_size=(h, w))
        
        # classifier        
        gap_shape = 32 + 64 + 128 + 256
        self.classifier = nn.Sequential(
            nn.Linear(gap_shape, fc_units),
            nn.SELU(inplace=True),
            nn.AlphaDropout(p=dropout),
            nn.Linear(fc_units, fc_units),
            nn.SELU(inplace=True),
            nn.AlphaDropout(p=dropout),
            nn.Linear(fc_units, num_classes)
        )
        
        # init
        self.init_parameters()
    
    def init_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_in = np.prod(module.weight.size()[1:])
                nn.init.normal_(module.weight, 0, np.sqrt(1 / fan_in))
                nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Linear):
                fan_in = module.weight.size()[0]
                nn.init.normal_(module.weight, 0, np.sqrt(1 / fan_in))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        batchsize = x.size(0)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        gaps = []        
        gaps.append(self.gap1(x1).view(batchsize, -1))        
        gaps.append(self.gap2(x2).view(batchsize, -1))        
        gaps.append(self.gap3(x3).view(batchsize, -1))        
        gaps.append(self.gap4(x4).view(batchsize, -1))
        x = torch.cat(gaps, dim=1)
        x = self.classifier(x)
        return x
    
    def loss(self, prediction, target):
        y = target / 2 + 0.5
        p = prediction
        eps = 1e-7
        mask = (y != 0.5).float().detach()        
        bce = p.clamp(min=0) - p * y + torch.log(1.0 + torch.exp(-p.abs()))
        bce[mask == 0] = 0
        loss = bce.sum() / (mask.sum() + eps)
        return loss


class GAPNet(TorchModel):
    def __init__(self, fc_units=2048, dropout=0.5, num_classes=209, input_shape=None):
        super(GAPNet, self).__init__()
        assert input_shape
        in_c = input_shape[0]
        in_h = input_shape[1]
        in_w = input_shape[2]
        # fc_units = model_params.get_value("fc_units", 1024)
        # drop_prob = model_params.get_value("dropout", 0.5)
        
        self.block1 = nn.Sequential(
            # input downsampling
            nn.Conv2d(in_c, 32, kernel_size=3, stride=2, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.SELU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.SELU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.SELU(inplace=True)
        )
        # global average pooling
        h, w = calc_out_shape(in_h, in_w, self.block1)        
        self.gap1 = nn.AvgPool2d(kernel_size=(h, w))
        h, w = calc_out_shape(h, w, self.block2)        
        self.gap2 = nn.AvgPool2d(kernel_size=(h, w))
        h, w = calc_out_shape(h, w, self.block3)        
        self.gap3 = nn.AvgPool2d(kernel_size=(h, w))
        h, w = calc_out_shape(h, w, self.block4)        
        self.gap4 = nn.AvgPool2d(kernel_size=(h, w))
        
        # classifier        
        gap_shape = 32 + 64 + 128 + 256
        self.classifier = nn.Sequential(
            nn.Linear(gap_shape, fc_units),
            nn.SELU(inplace=True),
            nn.AlphaDropout(p=dropout),
            nn.Linear(fc_units, fc_units),
            nn.SELU(inplace=True),
            nn.AlphaDropout(p=dropout),
            nn.Linear(fc_units, num_classes)
        )
        
        # init
        self.init_parameters()
    
    def init_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_in = np.prod(module.weight.size()[1:])
                nn.init.normal_(module.weight, 0, np.sqrt(1 / fan_in))
                nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Linear):
                fan_in = module.weight.size()[0]
                nn.init.normal_(module.weight, 0, np.sqrt(1 / fan_in))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        batchsize = x.size(0)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        gaps = []        
        gaps.append(self.gap1(x1).view(batchsize, -1))        
        gaps.append(self.gap2(x2).view(batchsize, -1))        
        gaps.append(self.gap3(x3).view(batchsize, -1))        
        gaps.append(self.gap4(x4).view(batchsize, -1))
        x = torch.cat(gaps, dim=1)
        x = self.classifier(x)
        return x
    
    def loss(self, prediction, target):
        y = target / 2 + 0.5
        p = prediction
        eps = 1e-7
        mask = (y != 0.5).float().detach()        
        bce = p.clamp(min=0) - p * y + torch.log(1.0 + torch.exp(-p.abs()))
        bce[mask == 0] = 0
        loss = bce.sum() / (mask.sum() + eps)
        return loss


class MILnet(TorchModel):
    def __init__(self, model_params=None, num_classes=478, input_shape=None):
        super(MILnet, self).__init__()
        assert input_shape
        in_c = input_shape[0]
        self.a = model_params.get_value("a", 10)
        self.num_classes = num_classes
        # --
        self.features = nn.Sequential(
            nn.AvgPool2d(3, stride=2),
            nn.Conv2d(in_c, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.2),
            nn.Conv2d(128, 1000, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(1000, num_classes, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.thresholds = nn.Parameter(torch.Tensor(num_classes))
        nn.init.constant_(self.thresholds, 0.1)
        self.classifier = nn.Linear(num_classes, num_classes)
        # init
        self.init_parameters()
    
    def init_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def loss(self, output, target):
        p, pooling = torch.split(output, self.num_classes, dim=1)
        y = target / 2 + 0.5
        mask = (y != 0.5).float().detach()
        eps = 1e-7        
        bce = p.clamp(min=0) - p * y + torch.log(1.0 + torch.exp(-p.abs()))
        bce[mask == 0] = 0
        loss = bce.sum() / (mask.sum() + eps)
        # mil-loss
        mil = pooling.clamp(min=0) - pooling * y + torch.log(1.0 + torch.exp(-pooling.abs()))
        mil[mask == 0] = 0
        mil_loss = mil.sum() / (mask.sum() + eps)        
        return loss + mil_loss
        
    def forward(self, input):
        x = self.features(input)    
        bi = self.thresholds
        a = self.a
        mean = torch.mean(torch.mean(x, dim=2), dim=2)
        pooling = (torch.sigmoid(a * (mean - bi)) - torch.sigmoid(-a * bi)) / (
                torch.sigmoid(a * (1 - bi)) - torch.sigmoid(-a * bi))
        x = self.classifier(pooling)
        return torch.cat([x, pooling], 1)


class KlammbauerNetRelu(TorchModel):
    def __init__(self, model_params=None, num_classes=209, input_shape=None):
        super(KlammbauerNetRelu, self).__init__()
        assert input_shape
        in_d = input_shape[0]
        fc_units = model_params.get_value("fc_units", 2048)
        drop_prob = model_params.get_value("dropout", 0.5)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(in_d, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
            nn.Linear(fc_units, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
            nn.Linear(fc_units, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
            nn.Linear(fc_units, num_classes)
        )
        
        # init
        self.init_parameters()
    
    def init_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.classifier(x)
    
    def loss(self, prediction, target):
        y = target / 2 + 0.5
        p = prediction
        eps = 1e-7
        mask = (y != 0.5).float().detach()
        bce = p.clamp(min=0) - p * y + torch.log(1.0 + torch.exp(-p.abs()))
        bce[mask == 0] = 0
        loss = bce.sum() / (mask.sum() + eps)        
        return loss


class MCNN(TorchModel):
    def __init__(self, num_classes=209, input_shape=None):
        super(MCNN, self).__init__()
        assert input_shape
        in_c = input_shape[0]
        in_h = input_shape[1]
        in_w = input_shape[2]
        
        for i, (n, c) in enumerate(zip(range(0, 7), [16, 16, 16, 32, 32, 32, 64])):
            s = int(math.pow(2, n))  # downscaling factor before conv
            p = int(math.pow(2, 6 - n))  # downscaling factor after conv
            setattr(self, "scale{}".format(i), nn.Sequential(
                nn.MaxPool2d(kernel_size=s, stride=s),
                nn.Conv2d(in_c, out_channels=c, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, out_channels=c, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, out_channels=c, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=p, stride=p)
            ))
        
        self.combine = nn.Sequential(
            nn.Conv2d(in_channels=208, out_channels=1024, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # calc shape after multiscale encoder
        h, w = calc_out_shape(in_h, in_w, self.scale0)
        h, w = calc_out_shape(h, w, self.combine)
        # FC layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features=h * w * 1024, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=num_classes)
        )
        
        # init
        self.init_parameters()
    
    def init_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        scales = []
        for i in range(0, 7):
            scales.append(getattr(self, "scale{}".format(i))(x))
        x = torch.cat(scales, dim=1)
        x = self.combine(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def loss(self, prediction, target):
        y = target / 2 + 0.5
        p = prediction
        eps = 1e-7
        mask = (y != 0.5).float().detach()
        bce = p.clamp(min=0) - p * (y * mask) + torch.log(1.0 + torch.exp(-p.abs()))
        bce[mask == 0] = 0
        loss = bce.sum() / (mask.sum() + eps)
        return loss
