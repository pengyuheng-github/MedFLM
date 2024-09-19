"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import torch
from PIL import Image

from minigpt4.datasets.datasets.base_dataset import BaseDataset

class SlakeVQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        self.vis_root = vis_root
        self.annotation = []
        self.annotation_path = ann_paths[0]


        anns = json.load(open(ann_paths[0], "r"))
        self.answer_list = list(map(text_processor,json.load(open(ann_paths[1], "r"))))
        self.answer_list.remove('')
        for ann in anns:
            item = {}
            item["question"] = ann["question"]
            item["answer"] = ann["answer"]
            item["img_name"] = ann["img_name"]
            item["img_id"] = ann["img_id"]
            item["qid"]= ann["qid"]
            self.annotation.append(item)

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["img_name"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        answer = self.text_processor(ann["answer"])

        return {
            "image": image,
            "question": question,
            "answer": answer,
            "qid": ann["qid"],
        }


class SlakeVQAEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        self.vis_root = vis_root
        self.annotation = []
        self.annotation_path = ann_paths[0]

        anns = json.load(open(ann_paths[0], "r"))
        self.answer_list = list(map(text_processor,json.load(open(ann_paths[1], "r"))))
        self.answer_list.remove('')
        for ann in anns:
            item = {}
            item["question"] = ann["question"]
            item["answer"] = ann["answer"]
            item["img_name"] = ann["img_name"]
            item["img_id"] = ann["img_id"]
            item["qid"] = ann["qid"]
            item["answer_list"] = self.answer_list

            self.annotation.append(item)

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["img_name"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        answer = self.text_processor(ann["answer"])

        return {
            "image": image,
            "question": question,
            "answer": answer,
            "qid": ann["qid"],
            "answer_list":ann["answer_list"]
        }
