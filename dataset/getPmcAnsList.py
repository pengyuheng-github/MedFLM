import json
ans_list = []
with open('pmc_train.json',encoding='UTF-8') as user_file:
    file_contents = json.load(user_file)
for item in file_contents:
    ans_list.append(json.loads(item)['answer'])

with open('pmc_test.json',encoding='UTF-8') as user_file:
    file_contents = json.load(user_file)
for item in file_contents:
    ans_list.append(json.loads(item)['answer'])

print(list(set(ans_list)))
str_ans_list = "["
for item in list(set(ans_list)):
    str_ans_list+="\""
    str_ans_list+=str(item).replace("\"","").replace("\'","")
    str_ans_list+="\","

str_ans_list = str_ans_list[:-1]+"]"
print(str_ans_list.__len__())

f=open("pmc_answer_list.json","w")
f.write(str_ans_list)
f.close()