from minigpt4.common.vqa_tools.vqa import VQA
from minigpt4.common.vqa_tools.vqa_eval import VQAEval

vqa = VQA("/home/pyh/minigpt4_cache/pmc/pmc_train.json","/home/pyh/minigpt4_cache/pmc/pmc_train.json" )

#print(vqa.dataset)
#{'img_id': 106, 'img_name': 'xmlab106/source.jpg', 'question': 'What diseases are included in the picture?', 'answer': 'Lung Cancer', 'qid': 9860}
# print(vqa.questions)
#{'img_id': 0, 'img_name': 'xmlab0/source.jpg', 'question': 'Where is the liver?', 'answer': 'Right', 'qid': 9844}
#print(vqa.qa)
#{9835: {'img_id': 0, 'img_name': 'xmlab0/source.jpg', 'question': 'What modality is used to take this image?', 'answer': 'MRI', 'qid': 9835} }
# print(vqa.qqa)
#{9835: {'img_id': 0, 'img_name': 'xmlab0/source.jpg', 'question': 'What modality is used to take this image?', 'answer': 'MRI', 'qid': 9835} }
#print(vqa.imgToQA)
#{0: {'img_id': 0, 'img_name': 'xmlab0/source.jpg', 'question': 'What modality is used to take this image?', 'answer': 'MRI', 'qid': 9835} }
#
# vqa_result = vqa.loadRes("/home/pyh/MedGPT/minigpt4/output/minigpt4_stage2_finetune_forSlake/20231018120/result/val_epoch:10_vqa_result.json","/home/pyh/minigpt4_cache/slake/Slake1.0/val_en.json")
# vqa_scorer = VQAEval(vqa, vqa_result, n=2)
# vqa_scorer.evaluate()
# overall_acc = vqa_scorer.accuracy["overall"]
# print(overall_acc)