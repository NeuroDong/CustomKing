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


from .Multivariable_classification.Transformer_cls import Transformer_cls

from .Multimodal_data_fusion.SE_Rexnext_Transformer import Se_resnext_tranformer
from .Multimodal_data_fusion.SENeXt_Transformer import SENeXt_Transformer
from .Multimodal_data_fusion.SENeXt_Encoder import SENeXt_Encoder
from .Multimodal_data_fusion.SENeXt_Decoder import SENeXt_Decoder
from .Multimodal_data_fusion.SE_Rexnext_Decoder import Se_resnext_Decoder
from .Multimodal_data_fusion.SE_Rexnext_Encoder import Se_resnext_Encoder
from .Multimodal_data_fusion.Resnext_Transformer import Resnext_tranformer
from .Multimodal_data_fusion.Resnext_Decoder import Resnext_decoder
from .Multimodal_data_fusion.Resnext_Encoder import Resnext_encoder
from .Multimodal_data_fusion.Resnet_Transformer import Resnet_tranformer
from .Multimodal_data_fusion.Resnet_Decoder import Resnet_decoder
from .Multimodal_data_fusion.Resnet_Encoder import Resnet_encoder

from .Image_classification.Resnext import ResNeXt101
from .Image_classification.ViT import VIT
from .Image_classification.Swin_Transformer import SwinTransformer
from .Image_classification.SE_Resnext import se_resnext_50,se_resnext_101,se_resnext_152
from .Image_classification.SK_CoAtNets import Sk_coatnet_0,Sk_coatnet_1,Sk_coatnet_2,Sk_coatnet_3,Sk_coatnet_4
from .Image_classification.CLFE_Multi_head import CLFE_Multi_head

__all__ = list(globals().keys())
