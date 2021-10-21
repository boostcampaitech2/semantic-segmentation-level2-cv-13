import torch.nn as nn
import torch.optim as optim
from torchvision import models

import argparse
from importlib import import_module

class FCN_resnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.fcn = models.segmentation.fcn_resnet50(pretrained=True)
        self.fcn.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    
    def forward(self, x):
        return self.fcn(x)

class DeepLabV3_resnet101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.deeplab.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, x):
        return self.deeplab(x)