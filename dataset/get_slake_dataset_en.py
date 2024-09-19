import json
with open('E:/LAVIS/datasets/Slake1.0/train.json',encoding='UTF-8') as user_file:
    file_contents = json.load(user_file)
slake_train_en_list = []
for item in file_contents:
    if(item['q_lang'] == 'en'):
        slake_train_en_list.append(item)
with open("E:/LAVIS/datasets/Slake1.0/train_en.json","w") as f:
    json.dump(slake_train_en_list,f)
    print("加载入文件完成...")

with open('E:/LAVIS/datasets/Slake1.0/test.json',encoding='UTF-8') as user_file:
    file_contents = json.load(user_file)
slake_train_en_list = []
for item in file_contents:
    if(item['q_lang'] == 'en'):
        slake_train_en_list.append(item)
with open("E:/LAVIS/datasets/Slake1.0/test_en.json","w") as f:
    json.dump(slake_train_en_list,f)
    print("加载入文件完成...")

with open('E:/LAVIS/datasets/Slake1.0/val.json',encoding='UTF-8') as user_file:
    file_contents = json.load(user_file)
slake_train_en_list = []
for item in file_contents:
    if(item['q_lang'] == 'en'):
        slake_train_en_list.append(item)
with open("E:/LAVIS/datasets/Slake1.0/val_en.json","w") as f:
    json.dump(slake_train_en_list,f)
    print("加载入文件完成...")

