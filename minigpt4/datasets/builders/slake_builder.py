"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from minigpt4.common.registry import registry
from minigpt4.datasets.datasets.slake_dataset import SlakeVQADataset, SlakeVQAEvalDataset



@registry.register_builder("slake")
class SlakeVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = SlakeVQADataset
    eval_dataset_cls = SlakeVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/slake/slake.yaml",
    }
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    # def build(self):
    #     self.build_processors()
    #
    #     build_info = self.config.build_info

