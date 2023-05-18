# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from customKing.data import DatasetCatalog

from .Image_classification.Cifar10 import register_Cifar10

def register_all_Cifar10(root):
    names = ["Cifar10_train","Cifar10_valid","Cifar10_train_and_valid","Cifar10_test",
            "Cifar10_train_and_valid_and_test"]
    for name in names:
        register_Cifar10(name,root)

if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.path.expanduser(os.getenv("CUSTOM_KING_DATASETS", "datasets"))
    register_all_Cifar10(_root)
    
