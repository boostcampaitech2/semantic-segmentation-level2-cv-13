import torch.nn as nn
import torch.optim as optim
from torchvision import models
import yaml
from hrnet.seg_hrnet_orc import get_seg_model
import torch.nn.functional as F


class FCN_resnet50(nn.Module):
    def __init__(self, num_classes = 11):
        super().__init__()
        self.num_classes = num_classes
        self.fcn = models.segmentation.fcn_resnet50(pretrained=True)
        self.fcn.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    
    def forward(self, x):
        return self.fcn(x)


class HRNetOCR(nn.Module):

    def __init__(self, num_classes = 11, target_size = 512, mode = "train"):

        super().__init__()
        self.num_classes = num_classes
        self.w, self.h = target_size, target_size
        self.mode = mode
        with open("./hrnet/hrnet_config/seg_hrnet_ocr_w48_512x512.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        self.model = get_seg_model(cfg)
    
    def forward(self, x):
        
        if self.training:
            x = self.model(x)
            x = [F.interpolate(input = x_, size = (self.w, self.h), mode = "bilinear", align_corners = True) for x_ in x]
            return {'out' : x}

        else:
            x = self.model(x)
            x = F.interpolate(input = x[1], size = (self.w, self.h), mode = "bilinear", align_corners = True)
        return {'out' : x}