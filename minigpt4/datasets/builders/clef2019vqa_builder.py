"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from minigpt4.common.registry import registry
from minigpt4.datasets.datasets.clef2019vqa_dataset import Clef2019VQADataset,Clef2019VQAEvalDataset



@registry.register_builder("clef2019vqa")
class Clef2019vqaBuilder(BaseDatasetBuilder):
    train_dataset_cls = Clef2019VQADataset
    eval_dataset_cls = Clef2019VQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/medvqa_dataset/clef2019vqa.yaml",
    }
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass


