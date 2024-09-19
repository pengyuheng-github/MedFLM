"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import os

import minigpt4.common.dist_utils as dist_utils
from minigpt4.common.registry import registry
from minigpt4.common.vqa_tools.vqa import VQA
from minigpt4.common.vqa_tools.vqa_eval import VQAEval
from minigpt4.tasks.base_task import BaseTask


@registry.register_task("med_vqa")
class MedVQATask(BaseTask):
    def __init__(self):
        super().__init__()

        self.answer_list = None

        self.ques_files = dict()
        self.anno_files = dict()


    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get question file, annotation file and anwser list in COCO format
        for dataset in datasets.values():
            for split in dataset:
                self.ques_files[split] = dataset[split].annotation_path
                self.anno_files[split] = dataset[split].annotation_path
            try:
                self.answer_list = dataset[split].answer_list
            except AttributeError:
                # if answer_list is not provided, then set it to None
                pass

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets

    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples
        )
        pred_qa_pairs = []

        question_id = samples["qid"]
        for answer, ques_id in zip(answers, question_id):
            ques_id = int(ques_id.item())
            pred_qa_pairs.append({"qid": ques_id, "answer": answer})

        return pred_qa_pairs
    def before_evaluation(self, model, dataset, **kwargs):
        pass
    def after_evaluation(self, val_result, split_name, epoch,**kwargs):
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_epoch:{epoch}_vqa_result",
            remove_duplicate="qid",
        )

        metrics = self._report_metrics(result_file=result_file, split=split_name)

        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Use official VQA evaluation script to report metrics.
        """
        metrics = {}

        if split in self.ques_files and split in self.anno_files:
            vqa = VQA(self.anno_files[split], self.ques_files[split])
            vqa_result = vqa.loadRes(
                resFile=result_file, quesFile=self.ques_files[split]
            )
            # create vqaEval object by taking vqa and vqaRes
            # n is precision of accuracy (number of places after decimal), default is 2
            vqa_scorer = VQAEval(vqa, vqa_result, n=2)
            logging.info("Start VQA evaluation.")
            vqa_scorer.evaluate()

            # print accuracies
            overall_acc = vqa_scorer.accuracy["overall"]
            yesno_acc = vqa_scorer.accuracy["yesno"]
            bleu1_avg = vqa_scorer.accuracy["bleu1_avg"]
            f1_avg = vqa_scorer.accuracy["f1_avg"]
            metrics["split"] = split
            metrics["agg_metrics"] = overall_acc
            metrics["yes/no"] = yesno_acc
            metrics["bleu1_avg"] = bleu1_avg
            metrics["f1_avg"] = f1_avg


            logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
            logging.info("yes/no Accuracy is: %.02f\n" % yesno_acc)
            logging.info("bleu1_avg is: %.02f\n" % bleu1_avg)
            logging.info("f1_avg is: %.02f\n" % f1_avg)
            logging.info("Per Answer Type Accuracy is the following:")

            for ans_type in vqa_scorer.accuracy["perAnswerType"]:
                logging.info(
                    "%s : %.02f"
                    % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
                )
                metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

            with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
            ) as f:
                f.write(json.dumps(metrics) + "\n")

        return metrics

