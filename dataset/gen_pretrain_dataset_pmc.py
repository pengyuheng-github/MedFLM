import pandas as pd
import json
# index	Figure_path	Caption	Question	Choice A	Choice B	Choice C	Choice D	Answer	split
def check_en_str(string):
    import re
    pattern = re.compile('^[A-Za-z0-9.,:;!?()_*"\' ]+$')
    if pattern.fullmatch(string):
        return True
    else:
        return False
df = pd.read_csv('/home/pyh/dataset_med/PMC_VQA/train.csv',encoding='utf-8')

test_list=[]
for index, row in df.iterrows():
    image,question,ans = str(row['Figure_path']).strip(),str(row['Question']).strip(),str(row['Answer']).strip()
    if(check_en_str(image) and check_en_str(question) and check_en_str(ans)):
        cor = {}
        cor['image'] = image
        cor['question'] = question
        cor['answer'] = ans
        cor['img_id'] = index
        cor['qid'] = index
        # test_list.append(cor)
        test_list.append(json.dumps(cor))

with open("/home/pyh/dataset_med/PMC_VQA/pmc_train.json","w") as f:
    json.dump(test_list,f)
    print("写入文件完成...",len(test_list))

# with open("/home/pyh/dataset_med/PMC_VQA/pmc_train_2.json") as f:
#     pmc_json = json.load(f)
#     print(len(pmc_json))
#     print(json.loads(pmc_json[0])['caption'])

