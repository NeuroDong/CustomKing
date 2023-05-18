# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from turtle import forward
from .build import META_ARCH_REGISTRY, build_model  # isort:skip

#Classfication model
from .Image_classification.Resnext import Resnet20,Resnet110,Resnet18,Resnet34,Resnet50,Resnet101,Resnet152,ResNeXt29_8x64d,ResNeXt29_16x64d,ResNeXt50,ResNeXt101,Wide_resnet50_2,Wide_resnet101_2

__all__ = list(globals().keys())
