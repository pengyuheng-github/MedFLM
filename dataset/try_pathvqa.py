import pandas as pd
import json
def pkl2json():
    i = 0
    splits = ["train","test","val"]
    for split in splits:
        data =  pd.read_pickle('/home/pyh/minigpt4_cache/pvqa/qas/{}/{}_qa.pkl'.format(split,split))
        print("%0d captions loaded from json " % len(data))

        pvqa = []

        for item in data:
            item['img_id'] = i
            item['qid'] = i
            item['image'] = split+"/"+item['image']+".jpg"
            i += 1
            pvqa.append(item)
        with open("/home/pyh/minigpt4_cache/pvqa/pathvqa_{}.json".format(split),"w") as f:
            json.dump(pvqa,f)
            print("加载入文件完成{}...".format(split))

def getAnsList():
    ans_list = []
    splits = ["train", "test", "val"]
    for split in splits:
        with open('/home/pyh/minigpt4_cache/pvqa/pathvqa_{}.json'.format(split), encoding='UTF-8') as file:
            file_contents = json.load(file)
            for item in file_contents:
                ans_list.append(item['answer'])
    str_ans_list = "["
    for item in list(set(ans_list)):
        str_ans_list += "\""
        str_ans_list += str(item)
        str_ans_list += "\","

    str_ans_list = str_ans_list[:-1] + "]"
    print(str_ans_list)

    f = open("/home/pyh/minigpt4_cache/pvqa/pathvqa_answer_list.json", "w")
    f.write(str_ans_list)
    f.close()
data =  pd.read_pickle('/home/pyh/minigpt4_cache/pvqa/qas/{}/{}_qa.pkl'.format("test","test"))
print(data[0])


