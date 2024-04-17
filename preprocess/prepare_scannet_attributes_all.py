import torch
import json
import os
import glob
import numpy as np

encoder = "mask3d"

split = "val"
max_inst_num = 100
scan_dir = f"/mnt/petrelfs/share_data/huanghaifeng/data/processed/scannet/{encoder}_ins_data_v3/pcd_all"
output_dir = "annotations"
split_path = f"annotations/scannet/scannetv2_{split}.txt"

scan_ids = [line.strip() for line in open(split_path).readlines()]

scan_ids = sorted(scan_ids)
# print(scan_ids)
from tqdm import tqdm

scans = {}
for scan_id in tqdm(scan_ids):
    pcd_path = os.path.join(scan_dir, f"{scan_id}.pth")
    if not os.path.exists(pcd_path):
        print('skip', scan_id)
        continue
    points, colors, instance_class_labels, instance_segids = torch.load(pcd_path)
    inst_locs = []
    num_insts = len(instance_class_labels)
    for i in range(min(num_insts, max_inst_num)):
        inst_mask = instance_segids[i]
        pc = points[inst_mask]
        if len(pc) < 10:
            print(scan_id, i, 'empty bbox')
            inst_locs.append(np.zeros(6, ).astype(np.float32))
            continue
        center = pc.mean(0)
        size = pc.max(0) - pc.min(0)
        inst_locs.append(np.concatenate([center, size], 0))
    inst_locs = torch.tensor(np.stack(inst_locs, 0), dtype=torch.float32)
    scans[scan_id] = {
        'objects': instance_class_labels,  # (n_obj, )
        'locs': inst_locs,  # (n_obj, 6) center xyz, whl
    }

# with open(os.path.join(output_dir, f"scannet_pointgroup_{split}_attributes.json"), "w") as f:
#     json.dump(scans, f)
torch.save(scans, os.path.join(output_dir, f"scannet_{encoder}_{split}_attributes{max_inst_num}_v3.pt"))