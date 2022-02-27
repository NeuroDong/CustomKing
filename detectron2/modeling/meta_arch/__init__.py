# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from .build import META_ARCH_REGISTRY, build_model  # isort:skip

from .panoptic_fpn import PanopticFPN

# import all the meta_arch, so they will be registered
from .rcnn import GeneralizedRCNN, ProposalNetwork
from .dense_detector import DenseDetector
from .retinanet import RetinaNet
from .fcos import FCOS
from .semantic_seg import SEM_SEG_HEADS_REGISTRY, SemanticSegmentor, build_sem_seg_head
from .SE_Resnext import se_resnext_50,se_resnext_101,se_resnext_152
from .Transformer_cls import Transformer_cls
from .SE_Rexnext_Transformer import Se_resnext_tranformer
from .ViT import VIT
from .Swin_Transformer import SwinTransformer
from .SENeXt_Transformer import SENeXt_Transformer
from .SENeXt_Encoder import SENeXt_Encoder
from .SENeXt_Decoder import SENeXt_Decoder
from .SE_Rexnext_Decoder import Se_resnext_Decoder
from .SE_Rexnext_Encoder import Se_resnext_Encoder
from .Resnext import ResNeXt101
from .Resnext_Transformer import Resnext_tranformer
from .Resnext_Decoder import Resnext_decoder
from .Resnext_Encoder import Resnext_encoder
from .Resnet_Transformer import Resnet_tranformer
from .Resnet_Decoder import Resnet_decoder
from .Resnet_Encoder import Resnet_encoder


__all__ = list(globals().keys())
