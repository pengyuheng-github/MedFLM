"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from minigpt4.common.registry import registry
from minigpt4.datasets.datasets.pmc_dataset import PmcVQADataset,PmcVQAEvalDataset



@registry.register_builder("pmc")
class PmcVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = PmcVQADataset
    eval_dataset_cls = PmcVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/pmc/pmc.yaml",
    }
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass


