import json

def getAnsList():
    ans_list = []
    splits = ["trainset", "testset"]
    for split in splits:
        with open('/home/pyh/minigpt4_cache/rad/{}.json'.format(split)) as file:
            file_contents = json.load(file)
            for item in file_contents:
                ans_list.append(item['answer'])
    str_ans_list = "["
    for item in list(set(ans_list)):
        str_ans_list += "\""
        str_ans_list += str(item).replace("\t"," ")
        str_ans_list += "\","

    str_ans_list = str_ans_list[:-1] + "]"
    print(str_ans_list)

    f = open("/home/pyh/minigpt4_cache/rad/rad_answer_list.json", "w")
    f.write(str_ans_list)
    f.close()
# getAnsList()
def addImgId():
    list1 = []
    list2 = []
    splits = ["trainset", "testset"]
    for split in splits:
        with open('/home/pyh/minigpt4_cache/rad/{}.json'.format(split)) as file:
            file_contents = json.load(file)
            for item in file_contents:
                item['img_id'] = item['qid']
                if split == "trainset":
                    list1.append(item)
                else:
                    list2.append(item)
    with open('/home/pyh/minigpt4_cache/rad/{}.json'.format("trainset"),"w") as file:
        json.dump(list1,file)
    with open('/home/pyh/minigpt4_cache/rad/{}.json'.format("testset"),"w") as file:
        json.dump(list2,file)
addImgId()
