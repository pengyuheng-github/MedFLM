import json
ans_list = []
with open('E:/LAVIS/datasets/Slake1.0/train_en.json',encoding='UTF-8') as user_file:
    file_contents = json.load(user_file)
for item in file_contents:
    ans_list.append(item['answer'])

with open('E:/LAVIS/datasets/Slake1.0/test_en.json',encoding='UTF-8') as user_file:
    file_contents = json.load(user_file)
for item in file_contents:
    ans_list.append(item['answer'])

with open('E:/LAVIS/datasets/Slake1.0/val_en.json',encoding='UTF-8') as user_file:
    file_contents = json.load(user_file)
for item in file_contents:
    ans_list.append(item['answer'])

print(list(set(ans_list)))
str_ans_list = "["
for item in list(set(ans_list)):
    str_ans_list+="\""
    str_ans_list+=str(item)
    str_ans_list+="\","

str_ans_list = str_ans_list[:-1]+"]"
print(str_ans_list)

f=open("answer_list_en.json","w")
f.write(str_ans_list)
f.close()