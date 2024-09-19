"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from minigpt4.common.registry import registry
from minigpt4.datasets.datasets.rad_dataset import RadDataset, RadEvalDataset



@registry.register_builder("rad")
class RadBuilder(BaseDatasetBuilder):
    train_dataset_cls = RadDataset
    eval_dataset_cls = RadEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/medvqa_dataset/rad.yaml",
    }
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass


