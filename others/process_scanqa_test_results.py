import json
import tqdm
import re

def clean_answer(data):
    data = data.lower()
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub('\.[ ]{2,}', '. ', data)
    data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)
    data = re.sub(r'\bletf\b' ,'left', data)
    data = re.sub(r'\blet\b' ,'left', data)
    data = re.sub(r'\btehre\b' ,'there', data)
    data = re.sub(r'\brigth\b' ,'right', data)
    data = re.sub(r'\brght\b' ,'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b' ,'TV', data)
    data = re.sub(r'\bchai\b' ,'chair', data)
    data = re.sub(r'\bwasing\b' ,'washing', data)
    data = re.sub(r'\bwaslked\b' ,'walked', data)
    data = re.sub(r'\boclock\b' ,'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data


preds = json.load(open('/mnt/petrelfs/huanghaifeng/share_hw/Chat-3D-v2/outputs/20240802_152505_lr5e-6_ep3_scanrefer#scan2cap#obj_align#scanqa#sqa3d#multi3dref#nr3d_caption__scanqa#scanqa_test__eval/preds_epoch-1_step0_scanqa_test.json', 'r'))

for split in ['w', 'wo']:
    # with open(f"annotations/scanqa/ScanQA_v1.0_test_w_obj.json", "r") as f:
    #     annos = json.load(f)
    with open(f"annotations/scanqa/ScanQA_v1.0_test_{split}_obj.json", "r") as f:
        annos = json.load(f)
        

    final_preds = []
    used_ids = set()

    for anno in tqdm.tqdm(annos):
        flag = 0
        for pred in preds:
            if pred['scene_id'] != anno['scene_id']:
                continue
            if pred['prompt'].split(' Answer the question')[0] != anno['question']:
                continue
            pred_ans = clean_answer(pred['pred'].lower()[:-1])
            if flag == 1:
                print(final_preds[-1], pred)
                continue
            final_preds.append({
                "scene_id": pred['scene_id'],
                "question_id": anno['question_id'],
                "answer_top10": [pred_ans] * 10,
                "bbox": []
            })
            used_ids.add(anno['question_id'])
            flag = 1

    print(len(used_ids))
    print(len(final_preds))
    print(len(annos))

    with open(f'scanqa_test_{split}_obj_preds.json', 'w') as f:
        json.dump(final_preds, f, indent=4)