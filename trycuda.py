import torch
flag = torch.cuda.is_available()
if flag:
    print("CUDA可使用")
else:
    print("CUDA不可用")

ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("驱动为：",device)
print("GPU型号： ",torch.cuda.get_device_name(0))

print(torch.__version__)

print(torch.version.cuda)
print(torch.backends.cudnn.version())

# python demo.py --cfg-path=eval_configs/minigpt4_eval.yaml
#python train.py --cfg-path=train_configs/minigpt4_stage2_finetune.yaml
#python train.py --cfg-path=train_configs/minigpt4_stage2_finetune_forSlake.yaml

# python evaluate.py --cfg-path eval_configs/vqav2_zeroshot_flant5xl_eval.yaml


"""
Traceback (most recent call last):
  File "/home/pyh/MiniGPT4/MiniGPT-4-main/evaluate.py", line 92, in <module>
    main()
  File "/home/pyh/MiniGPT4/MiniGPT-4-main/evaluate.py", line 82, in main
    datasets = task.build_datasets(cfg)
  File "/home/pyh/MiniGPT4/MiniGPT-4-main/minigpt4/tasks/vqa.py", line 72, in build_datasets
    datasets = super().build_datasets(cfg)
  File "/home/pyh/MiniGPT4/MiniGPT-4-main/minigpt4/tasks/base_task.py", line 57, in build_datasets
    dataset = builder.build_datasets()
  File "/home/pyh/MiniGPT4/MiniGPT-4-main/minigpt4/datasets/builders/base_dataset_builder.py", line 57, in build_datasets
    datasets = self.build()  # dataset['train'/'val'/'test']
  File "/home/pyh/MiniGPT4/MiniGPT-4-main/minigpt4/datasets/builders/base_dataset_builder.py", line 172, in build
    self.build_processors()
  File "/home/pyh/MiniGPT4/MiniGPT-4-main/minigpt4/datasets/builders/base_dataset_builder.py", line 70, in build_processors
    self.vis_processors["eval"] = self._build_proc_from_cfg(vis_eval_cfg)
  File "/home/pyh/MiniGPT4/MiniGPT-4-main/minigpt4/datasets/builders/base_dataset_builder.py", line 82, in _build_proc_from_cfg
    registry.get_processor_class(cfg.name).from_config(cfg)
AttributeError: 'NoneType' object has no attribute 'from_config'



"""
