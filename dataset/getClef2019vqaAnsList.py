import json

def getAnsList():
    ans_list = []
    splits = ["All_QA_Pairs_train", "All_QA_Pairs_val"]
    for split in splits:
        with open('/home/pyh/minigpt4_cache/clef2019vqa/{}.json'.format(split)) as file:
            file_contents = json.load(file)
            for item in file_contents:
                ans_list.append(json.loads(item)['answer'])
    str_ans_list = "["
    for item in list(set(ans_list)):
        str_ans_list += "\""
        str_ans_list += str(item).replace("\t"," ").replace('\'',' ').replace('\"',' ')
        str_ans_list += "\","

    str_ans_list = str_ans_list[:-1] + "]"
    print(str_ans_list)

    f = open("/home/pyh/minigpt4_cache/clef2019vqa/clef2019vqa_answer_list.json", "w")
    f.write(str_ans_list)
    f.close()
def addimgID():

    splits = ["All_QA_Pairs_train", "All_QA_Pairs_val"]
    for split in splits:
        alist = []
        with open('/home/pyh/minigpt4_cache/clef2019vqa/{}.json'.format(split)) as file:
            #"{\"qid\": 1, \"image_name\": \"synpic54733.jpg\", \"question\": \"what ?\", \"answer\": \"c \"}"
            file_contents = json.load(file)
            for file_content in file_contents:
                file_content = json.loads(file_content)
                file_content['img_id'] = file_content['qid']
                alist.append(file_content)
        with open('/home/pyh/minigpt4_cache/clef2019vqa/{}_.json'.format(split),'w') as file:
            json.dump(alist, file)
addimgID()