import json
import re
import torch
from tqdm import tqdm
import sys

sys.path.append('.')
from utils.box_utils import construct_bbox_corners

preds = json.load(open('/mnt/petrelfs/huanghaifeng/share_hw/Chat-3D-v2/outputs/20240829_191645_lr5e-6_ep3_scanrefer#obj_align#nr3d_caption#scan2cap#scanqa#sqa3d#multi3dref__scanrefer_test__chatscene/preds_epoch-1_step0_scanrefer_test.json'))

instance_attrs = torch.load("/mnt/petrelfs/huanghaifeng/share_hw/Chat-3D-v2/annotations/scannet_mask3d_test_attributes.pt", map_location="cpu")

print(len(preds))
outputs = []

id_format = "<OBJ\\d{3}>"

for pred in tqdm(preds):
    scene_id = pred['scene_id']
    object_id = pred['gt_id']
    ann_id = pred['type_info']
    pred_str = pred['pred']
    instance_locs = instance_attrs[scene_id]['locs']
    instance_num = instance_locs.shape[0]
    pred_id = 0
    for match in re.finditer(id_format, pred_str):
        idx = match.start()
        cur_id = int(pred_str[idx+4:idx+7])
        if cur_id < instance_num:
            pred_id = cur_id
            break
    pred_locs = instance_locs[pred_id].tolist()
    pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:]).tolist()
    outputs.append({
        "scene_id": scene_id,
        "object_id": object_id,
        "ann_id": ann_id,
        "bbox": pred_corners
    })

with open('tmp_files/scanrefer_test_results.json', 'w') as f:
    json.dump(outputs, f, indent=4)