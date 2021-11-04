from dpt.models import DPTSegmentationModel
from fpn.model import FPN_Model
from hrnet.seg_hrnet_ocr import get_seg_model
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F
from unet_custom.unet_custom import get_unet_custom
import yaml



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
            x = F.interpolate(input = x[0], size = (self.w, self.h), mode = "bilinear", align_corners = True)
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

class DeformableConv2d(nn.Module):
    """
    Deformable Convolution v2 2d Layer

    code reference: https://github.com/developer0hye/PyTorch-Deformable-Convolution-v2/blob/main/dcn.py
    """
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False,
                 use_v2 = False):
        
        super(DeformableConv2d, self).__init__()

        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.use_v2 = use_v2

        self.offset_conv = nn.Conv2d(in_channels,
                                    2*kernel_size*kernel_size,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=self.padding,
                                    bias=True)
        
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        if self.use_v2:
            self.modulator_conv = nn.Conv2d(in_channels,
                                            1*kernel_size*kernel_size,
                                            kernel_size = kernel_size,
                                            stride=stride,
                                            padding=self.padding,
                                            bias=True)

            nn.init.constant_(self.modulator_conv.weight, 0.)
            nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels = out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):

        offset = self.offset_conv(x)
        
        if self.use_v2:
            modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        else:
            modulator = None

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride)
            
        return x


class DeformableDeepLabV3(nn.Module):
    """
    Deformable Convolutions applied to backbone of DL3
    """
    model_name = 'DeformableDeepLabV3'
    def __init__(self,
                 num_classes = 11,
                 target_size = 256,
                 use_v2 = False,
                 use_aux_loss = False):
        super(DeformableDeepLabV3, self).__init__()
        self.deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True, aux_loss = use_aux_loss)
        self._deformable_conv(self.deeplab, use_v2)
        self.deeplab.classifier[4] = nn.Conv2d(target_size, num_classes, kernel_size=1)
        
    def _deformable_conv(self, model, use_v2):

        layers = [name for name, _ in model.backbone.named_children() if 'layer' in name]
        if not use_v2:
            layers = [layers[-1]]
            
        for layer in layers:
            layer_ = getattr(model.backbone, layer)
            for block in layer_[-3:]:
                in_channels = getattr(block.conv2, "in_channels")
                out_channels = getattr(block.conv2, 'out_channels')
                block.conv2 = DeformableConv2d(in_channels, out_channels, use_v2 = use_v2)

    def forward(self, x):
        return self.deeplab(x)


class ResNestDeepLabV3(nn.Module):
    model_name = "ResNestDeepLabV3"

    def __init__(self, encoder_name, encoder_weight='imagenet', in_channels = 3, num_classes = 11):
        super(ResNestDeepLabV3, self).__init__()
        self.encoder_name = encoder_name
        self.encoder_weight = encoder_weight
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.model = smp.DeepLabV3(
                                    encoder_name=self.encoder_name,
                                    encoder_weights = self.encoder_weight,
                                    in_channels = self.in_channels,
                                    classes = self.num_classes
                                    )
        
    def forward(self, x):
        return {'out': self.model(x)}

class FPN(nn.Module):
    model_name = "FPN"

    def __init__(self, encoder_name, encoder_weight="imagenet", decoder_merge_policy = "cat", in_channels = 3, num_classes = 11):
        super(FPN, self).__init__()
        self.encoder_name = encoder_name
        self.encoder_weight = encoder_weight
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.decoder_merge_policy = decoder_merge_policy
        self.model = smp.FPN(
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weight,
            classes = self.num_classes,
            in_channels = self.in_channels,
            decoder_merge_policy = self.decoder_merge_policy
        )

    def forward(self, x):
        return {'out': self.model(x)}


class CustomFPN(nn.Module):
    model_name = "FPN"

    def __init__(self, encoder_name, decoder_augmented_pyramid_channels, encoder_weight="imagenet", decoder_merge_policy="cat", in_channels=3, num_classes=11):
        super(CustomFPN, self).__init__()
        self.encoder_name=encoder_name
        self.encoder_weight=encoder_weight
        self.in_channels=in_channels
        self.num_classes=num_classes
        self.decoder_augmented_pyramid_channels=decoder_augmented_pyramid_channels
        self.decoder_merge_policy=decoder_merge_policy
        self.model=FPN_Model(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weight,
            decoder_augmented_pyramid_channels=self.decoder_augmented_pyramid_channels,
            classes=self.num_classes,
            in_channels=self.in_channels
        )

    def forward(self, x):
        return {'out': self.model(x)}