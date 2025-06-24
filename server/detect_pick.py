from collections import defaultdict
import os
import cv2
import numpy as np
import mmcv
from pathlib import Path
# from mmengine import ProgressBar
from pyskl.apis import inference_recognizer, init_recognizer

config = mmcv.Config.fromfile('configs/posec3d/slowonly_r50_hmdb51_k400p/s1_joint.py')
config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']

model = init_recognizer(config, 'model/best_top1_acc_epoch_11.pth', 'cpu')
label_map = [x.strip() for x in open('tools/data/label_map/hmdb51.txt').readlines()]

def analyze_pick_action(keypoints_buffer, h, w):
    pick_results = []

    for tid, kps_seq in keypoints_buffer.items():
        frame_indices, keypoint_seq = zip(*kps_seq)
        keypoint_seg = np.array(keypoint_seq)[..., :2] 
        score_seg = np.array(keypoint_seq)[..., 2:] 

        fake_anno_seg = dict(
            frame_dir='',
            label=-1,
            img_shape=(h, w),
            original_shape=(h, w),
            start_index=0,
            modality='Pose',
            total_frames=keypoint_seg.shape[0],
            keypoint=keypoint_seg[np.newaxis, ...],
            keypoint_score=score_seg[np.newaxis, ...],
        )

        top5, _ = inference_recognizer(model, fake_anno_seg)
        top5_labels = [label_map[cls_id] for cls_id, _ in top5]

        if 'pick' in top5_labels:
            pick_results.append(tid)

    return pick_results