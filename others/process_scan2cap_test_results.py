import json
import nltk
import sys
sys.path.append('.')
from utils.box_utils import construct_bbox_corners
import torch
import csv
from tqdm import tqdm


def process_text(text):
    sentences = nltk.sent_tokenize(text)
    capitalized_sentences = [sentence[0].lower()+sentence[1:] for sentence in sentences]
    tmp_text = ' '.join(capitalized_sentences)
    tokens = nltk.tokenize.word_tokenize(tmp_text)
    result = " ".join(tokens)
    result = "sos " + result + " eos"
    return result

gt_annos = json.load(open('/mnt/petrelfs/huanghaifeng/share_hw/Chat-3D-v2/annotations/scannet_val_attributes.json'))
tot_len = 0
for anno in gt_annos.values():
    tot_len += len(anno['obj_ids'])
print(float(tot_len)/len(gt_annos))

preds = json.load(open('/mnt/petrelfs/huanghaifeng/share_hw/Chat-3D-v2/outputs/20240829_220503_lr5e-6_ep3_scanrefer#obj_align#nr3d_caption#scan2cap#scanqa#sqa3d#multi3dref__scan2cap_test__chatscene/preds_epoch-1_step0_scan2cap_test.json'))
instance_attribute_file = f"annotations/scannet_mask3d_test_attributes.pt"
instance_attrs = torch.load(instance_attribute_file, map_location='cpu')


type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
    'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
    'refrigerator':12, 'shower curtain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'others':17}
id2class = {}
with open("/mnt/petrelfs/huanghaifeng/share_hw/Chat-3D-v2/annotations/scannet/scannetv2-labels.combined.tsv", "r") as f:
    csvreader = csv.reader(f, delimiter="\t")
    csvreader.__next__()
    for line in csvreader:
        nyuclass = line[7]
        tmp_id =int(line[0])
        if nyuclass in type2class:
            id2class[tmp_id] = type2class[nyuclass]
        else:
            id2class[tmp_id] = type2class['others']
outputs = {}

for pred in tqdm(preds):
    scene_id = pred['scene_id']
    pred_file = f"/mnt/petrelfs/huanghaifeng/share_hw/Chat-3D-v2/tmp_files/Mask3DInst/test/{scene_id}.txt"
    pred_infos = [x.strip() for x in open(pred_file).readlines()]
    if scene_id not in outputs:
        outputs[scene_id] = []
    instance_locs = instance_attrs[scene_id]["locs"]
    pred_id = pred['pred_id']
    pred_locs = instance_locs[pred_id].tolist()
    pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:]).tolist()
    caption = pred['pred']
    caption = process_text(caption)
    pred_info = pred_infos[pred_id].split(" ")
    objectness = float(pred_info[-1])
    if objectness < 0.75:
        continue
    sem_prob = [0.] * 18
    class_id = int(pred_info[1])
    try:
        sem_prob[id2class[class_id]] = 1.
    except:
        breakpoint()
    outputs[scene_id].append({
        "caption": caption,
        "box": pred_corners,
        "sem_prob": sem_prob,
        "obj_prob": [0., objectness]
    })

tot_len = 0
for v in outputs.values():
    tot_len += len(v)
print(float(tot_len) / len(outputs))

with open('tmp_files/scan2cap_test_results.json', 'w') as f:
    json.dump(outputs, f, indent=4)