"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from minigpt4.common.registry import registry
from minigpt4.datasets.datasets.pathvqa_dataset import PathVQADataset, PathVQAEvalDataset



@registry.register_builder("pathvqa")
class PathVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = PathVQADataset
    eval_dataset_cls = PathVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/medvqa_dataset/pathvqa.yaml",
    }
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass


