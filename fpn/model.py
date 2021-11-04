# Custom fpn with augmented bottom-up structure

# - Citation
# @misc{Yakubovskiy:2019,
#   Author = {Pavel Yakubovskiy},
#   Title = {Segmentation Models Pytorch},
#   Year = {2020},
#   Publisher = {GitHub},
#   Journal = {GitHub repository},
#   Howpublished = {\url{https://github.com/qubvel/segmentation_models.pytorch}}
# }

from .decoder import FPNDecoder
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead, ClassificationHead
from typing import Optional, Union

class FPN_Model(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "efficientnet-b3",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_pyramid_channels: int = 256,
        decoder_augmented_pyramid_channels: int = 192,
        decoder_segmentation_channels: int = 128,
        decoder_merge_policy: str = "cat",
        decoder_dropout: float = 0.2,
        in_channels: int = 3,
        classes: int = 11,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            augmented_pyramid_channels = decoder_augmented_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "fpn-{}".format(encoder_name)
        self.initialize()
