import torch.nn as nn
from torchvision import models
import yaml
from hrnet.seg_hrnet_ocr import get_seg_model
from unet_custom.unet_custom import get_unet_custom
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from dpt.models import DPTSegmentationModel

class FCN_resnet50(nn.Module):
    model_name = "FCN_resnet50"
    def __init__(self, num_classes = 11, target_size=512):
        super().__init__()
        self.num_classes = num_classes
        self.fcn = models.segmentation.fcn_resnet50(pretrained=True)
        self.fcn.classifier[4] = nn.Conv2d(target_size, num_classes, kernel_size=1)
    
    def forward(self, x):
        return self.fcn(x)

class DeepLabV3_resnet101(nn.Module):
    model_name = "DeepLabV3_resnet101"
    def __init__(self, num_classes=11, target_size=256):
        super().__init__()
        self.num_classes = num_classes
        self.deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.deeplab.classifier[4] = nn.Conv2d(target_size, num_classes, kernel_size=1)
    
    def forward(self, x):
        return self.deeplab(x)

class HRNetOCR(nn.Module):
    model_name = "HRNetOCR"
    def __init__(self, num_classes = 11, target_size = 512):

        super().__init__()
        self.num_classes = num_classes
        self.w, self.h = target_size, target_size

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

class Unet(nn.Module):
    model_name = "Unet"

    def __init__(self, encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=3, classes=11):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
        self.model = smp.Unet(
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            in_channels = self.in_channels,
            classes = self.classes
        )
    
    def forward(self, x):
        return {'out': self.model(x)}

class DeepLabV3Plus(nn.Module):
    model_name = "DeepLabV3Plus"

    def __init__(self, encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=3, classes=11):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
        self.model = smp.DeepLabV3Plus(
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            in_channels = self.in_channels,
            classes = self.classes
        )
    
    def forward(self, x):
        return {'out': self.model(x)}

class MAnet(nn.Module):
    model_name = "MAnet"

    def __init__(self, encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=3, classes=11):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
        self.model = smp.MAnet(
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            in_channels = self.in_channels,
            classes = self.classes
        )
    
    def forward(self, x):
        return {'out': self.model(x)}

class Linknet(nn.Module):
    model_name = "Linknet"

    def __init__(self, encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=3, classes=11):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
        self.model = smp.Linknet(
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            in_channels = self.in_channels,
            classes = self.classes
        )
    
    def forward(self, x):
        return {'out': self.model(x)}

class PSPNet(nn.Module):
    model_name = "PSPNet"

    def __init__(self, encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=3, classes=11):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
        self.model = smp.PSPNet(
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            in_channels = self.in_channels,
            classes = self.classes
        )
    
    def forward(self, x):
        return {'out': self.model(x)}

class PAN(nn.Module):
    model_name = "PAN"

    def __init__(self, encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=3, classes=11):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
        self.model = smp.PAN(
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            in_channels = self.in_channels,
            classes = self.classes
        )
    
    def forward(self, x):
        return {'out': self.model(x)}


class Custom_Unet(nn.Module):
    '''
    config
    "model": {"name" : "Custom_Unet",
            "args" : {"stride" : 1, << ASPP dialation rate 
                    "num_classes" :11}},
    '''
    model_name = 'Custom_Unet'
    def __init__(self, stride = 1, num_classes=11):
        super().__init__()
        self.model = get_unet_custom(stride, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return {'out': x}


class CustomDPTHybrid(nn.Module):
    model_name = "CustomDPTHybrid"
    def __init__(self, num_classes = 11, path = "./dpt/dpt_pretrained/dpt_hybrid-ade20k-53898607.pt"):
        super().__init__()
        self.model = DPTSegmentationModel(150, path = path, backbone="vitb_rn50_384")
        self.model.auxlayer[4] = nn.Conv2d(256, num_classes, kernel_size = 1)
        self.model.scratch.output_conv[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, x):
        if self.training:
            x = self.model(x)
            x = [x[0], self.model.auxlayer(x[1])]
        else:
            x = self.model(x)[0]
        return {'out' : x}


class CustomDPTLarge(nn.Module):
    model_name = "CustomDPTLarge"
    def __init__(self, num_classes = 11, path = "./dpt/dpt_pretrained/dpt_large-ade20k-b12dca68.pt"):
        super().__init__()
        self.model = DPTSegmentationModel(150, path = path, backbone="vitl16_384")
        self.model.auxlayer[4] = nn.Conv2d(256, num_classes, kernel_size = 1)
        self.model.scratch.output_conv[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, x):
        if self.training:
            x = self.model(x)
            x = [x[0], self.model.auxlayer(x[1])]
        else:
            x = self.model(x)[0]
        return {'out' : x}