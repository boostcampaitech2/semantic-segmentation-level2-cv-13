# Custom fpn decoder with augmented bottom-up structure

# - Citation
# @misc{Yakubovskiy:2019,
#   Author = {Pavel Yakubovskiy},
#   Title = {Segmentation Models Pytorch},
#   Year = {2020},
#   Publisher = {GitHub},
#   Journal = {GitHub repository},
#   Howpublished = {\url{https://github.com/qubvel/segmentation_models.pytorch}}
# }

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample=upsample
        self.block=nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding = 1, bias = False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        
        return x

class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.block(x)


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels, mode="top-down", upsample = "interpolate"):
        """
        Custom FPN Block
        - Args
            pyramid_channels: a channel of feature pyramids (i.e. 256)
            skip_channels: a channel of the corresponding feature map with a feature pyramid
            mode: direction of path, "top-down" for standard fpn or "bottom-up" for augmented bottom-up structure (default: top-down)
            upsample: how to upsample the feature map. must be one of "upconv" or "interpolate" (default: interpolate)
        """
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1) # 1x1 conv
        self.down_conv = nn.Conv2d(pyramid_channels, pyramid_channels, kernel_size=3, stride=2, padding=1) # output size를 맞춰주기 위해 padding
        self.upsample = upsample
        if self.upsample == "upconv":
            self.up_conv = nn.ConvTranspose2d(pyramid_channels, pyramid_channels, kernel_size=3, stride=2, padding=1, output_padding=1) # output_padding을 줘야 2배 upsample이 가능.
        self.mode = mode
    
    def forward(self, x, skip=None):
        if self.mode == "top-down":
            if self.upsample == "interpolate":
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            elif self.upsample == "upconv":
                x = self.up_conv(x) # upsampling with factor 2.
            else:
                raise ValueError("upsample must be one of [interpolate, upconv]")
        elif self.mode == "bottom-up":
            x = self.down_conv(x)
        else:
            raise ValueError("Unexpected mode {self.mode}. Mode must be of 'bottom-up' or 'top-down'")
        
        skip = self.skip_conv(skip)
        x = x + skip
        
        return x


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy)
            )        


class FPNDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        encoder_depth=5,
        pyramid_channels=256,
        augmented_pyramid_channels=256,
        segmentation_channels=128,
        dropout=0.2,
        merge_policy="cat",
        upsample="interpolate"
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels*4
        
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[:encoder_depth + 1]

        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.n2 = nn.Conv2d(pyramid_channels, augmented_pyramid_channels, kernel_size=1)
        self.n3 = FPNBlock(augmented_pyramid_channels, pyramid_channels, mode="bottom-up", upsample=upsample)
        self.n4 = FPNBlock(augmented_pyramid_channels, pyramid_channels, mode="bottom-up", upsample=upsample)
        self.n5 = FPNBlock(augmented_pyramid_channels, pyramid_channels, mode="bottom-up", upsample=upsample)

        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(augmented_pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
            for n_upsamples in [3,2,1,0]
        ])

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, *features):
        c2, c3, c4, c5 = features[-4:]
            
        # top-down path
        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        # bottom-up augmented path
        n2 = self.n2(p2) # largest feature map
        n3 = self.n3(n2, p3) 
        n4 = self.n4(n3, p4)
        n5 = self.n5(n4, p5) # smallest feature map

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [n5,n4,n3,n2])]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)

        return x