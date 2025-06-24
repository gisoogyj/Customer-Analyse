# Copyright (c) OpenMMLab. All rights reserved.
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['PYTORCH_JIT'] = '0'
import argparse
import cv2
import sys
# import mmcv
import csv
# import os.path as osp
# import numpy as np
# import shutil
import torch
import warnings
import torchvision
import torch.nn as nn
import threading
import requests
# import onnxruntime
from insightface.app import FaceAnalysis
from ultralytics import YOLO
from collections import defaultdict, deque
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tracker.mc_bot_sort import BoTSORT
from all import analyze_stay_behavior, analyze_pick_behavior

from count import analyze_crowd
from pose import detection_inference
# from detect_pick import analyze_pick_action_segments
from person_demographic import person_demo, get_person_results
from detect_item import detect_hand_objects

try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
except (ImportError, ModuleNotFoundError):
    def init_pose_model(*args, **kwargs):
        pass

    def inference_top_down_pose_model(*args, **kwargs):
        pass

    def vis_pose_result(*args, **kwargs):
        pass

    warnings.warn(
        'Failed to import `init_pose_model`, `inference_top_down_pose_model`, `vis_pose_result` from '
        '`mmpose.apis`. Make sure you can successfully import these if you want to use related features. '
    )

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1

picked_id_buffer = []
def send_keypoints_async(payload):
    def send():
        try:
            res = requests.post("http://<服务器IP>:5000/keypoints", json=payload, timeout=10)
            res_json = res.json()
            picked_ids = res_json.get("picked_ids", [])
            print(f"[线程] 收到 picked_ids: {picked_ids}")

            # ✅ 不在这里分析，只是放入缓冲区
            picked_id_buffer.extend(picked_ids)

        except Exception as e:
            print("⚠️ 请求失败：", e)

    threading.Thread(target=send).start()


def write_one_result_to_csv(track_id, gender, age, race, behavior, objects, output_path='stream_analysis_results.csv'):
    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([track_id, gender, age, race, behavior, ', '.join(objects)])

def main():
    args = parse_args()

    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
    tracker = BoTSORT(args)
    yolo_model = YOLO(args.weights)
    device = torch.device('cpu')
    # Load FairFace model
    fairface_model = torchvision.models.resnet34(pretrained=True)
    fairface_model.fc = nn.Linear(fairface_model.fc.in_features, 18)
    fairface_model.load_state_dict(torch.load('Pretrained/res34_fair_align_multi_7_20190809.pt', map_location=device))
    fairface_model = fairface_model.to(device).eval()

    # Load InsightFace
    face_app = FaceAnalysis(name='buffalo_l')
    face_app.prepare(ctx_id=0)
    # face_app = FaceAnalysis(name='scrfd', allowed_modules=['detection'])
    # face_app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    h, w = None, None

    keypoints_buffer = defaultdict(lambda: deque(maxlen=75))
    image_buffer = defaultdict(lambda: deque(maxlen=10))
    pose_results_window = {}

    stay_counter = defaultdict(int)
    stay_ids = set()
    analyzed_stay_ids = set()
    analyzed_pick_ids = set()
    analyzed_id_attributes = dict()
    stay_threshold = 50
    # output_path = 'stream_analysis_results.csv'

    # with open(output_path, 'w', newline='', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['TrackID', 'Gender', 'Age', 'Race', 'Behavior', 'DetectedObjects'])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx == 0:
            h, w = frame.shape[:2]
        
        print("[DEBUG] frame_idx:", frame_idx)

        bboxes = detection_inference(yolo_model, frame)
        tracks = tracker.update(bboxes, frame)

        pose_results = []
        det_inputs = []
        track_ids = []
        poses = []
        boxes_with_ids = []

        for t in tracks:
            tid = t.track_id
            bbox = [int(x) for x in t.tlbr]
            det_inputs.append({'bbox': bbox})
            track_ids.append(tid)
            boxes_with_ids.append((tid, bbox))

        if det_inputs:
            for single_input in det_inputs:
                pose = inference_top_down_pose_model(pose_model, frame, [single_input], format='xyxy')[0]
                poses.extend(pose)
        #   poses = inference_top_down_pose_model(pose_model, frame, det_inputs, format='xyxy')[0]
        else:
            poses = []

        if frame_idx not in pose_results_window:
            pose_results_window[frame_idx] = []

        for pose, tid, bbox_info in zip(poses, track_ids, det_inputs):
            pose['track_id'] = tid
            pose['bbox'] = bbox_info['bbox'] + [1.0]
            pose_results.append(pose)
            keypoints = pose['keypoints']
            keypoints_buffer[tid].append((frame_idx, keypoints))
            pose_results_window[frame_idx].append(pose)

        if frame_idx % int(fps * 5) == 0 and frame_idx != 0:
            analyze_crowd(
                frame_idx, fps, frame, boxes_with_ids,
                fairface_model, face_app,
                analyzed_id_attributes
            )

        if frame_idx % 5 == 0:
            image_buffer['full'].append({
                'frame_idx': frame_idx,
                'image': frame.copy()
            })

        for pose in pose_results:
            tid = pose['track_id']
            stay_counter[tid] += 1
            if stay_counter[tid] >= stay_threshold and tid not in stay_ids:
                stay_ids.add(tid)
        
        analyze_stay_behavior(image_buffer, analyzed_stay_ids, keypoints_buffer, stay_ids,
                          args, fairface_model, face_app, analyzed_id_attributes,
                          person_demo, get_person_results, detect_hand_objects,
                          pose_results_window, center_size=50)
        print("stayid:", analyzed_stay_ids)
        print("analysed:", analyzed_id_attributes)

        if frame_idx % int(fps * 2) == 0 and frame_idx != 0:
            # Step 1: 截取最近2秒的keypoints
            start_frame = frame_idx - int(fps * 2)
            end_frame = frame_idx
            tem_keypoints_buffer = {}

            for tid, deque_data in keypoints_buffer.items():
                filtered = [
                    [fid, kps] for (fid, kps) in deque_data
                    if start_frame <= fid < end_frame
                ]
                if filtered:
                    tem_keypoints_buffer[tid] = filtered

            # 截取image_buffer
            tem_image_buffer = {
                fid: image_buffer[fid]
                for fid in range(start_frame, end_frame)
                if fid in image_buffer
            }

            # Step 2: POST发送到服务器
            payload = {
                "device": "raspberry_pi_01",
                # "frame_range": [start_frame, end_frame],
                # "fps": fps,
                'shape_h': h,
                'shape_w': w,
                "keypoints_buffer": tem_keypoints_buffer
            }

            send_keypoints_async(payload) 
        
        if picked_id_buffer:
            det_id = picked_id_buffer.copy()
            picked_id_buffer.clear()  # 清空缓冲区，避免重复处理

            analyze_pick_behavior(det_id, tem_image_buffer, analyzed_pick_ids, tem_keypoints_buffer, args,
                          fairface_model, face_app, h, w, analyzed_id_attributes,
                          person_demo, get_person_results, detect_hand_objects,
                          pose_results_window, hand_box_size=50)
            print("pick:", analyzed_pick_ids)
            print("analysed:", analyzed_id_attributes)

        frame_idx += 1


def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo')
    parser.add_argument('video', help='video file/url')
    parser.add_argument(
        '--pose-config',
        default='Pretrained/litehrnet_18_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default='Pretrained/litehrnet_18_coco_256x192.pth',
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--device', type=str, default='cpu', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    
    parser.add_argument('--weights', type=str, default='yolov8s-seg.pt', help='YOLOv8-seg model path')
    parser.add_argument('--project', default='runs/detect', help='Project folder') 


    # tracking args
    parser.add_argument('--name', default='exp', help='Name of the current experiment')
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")
    parser.add_argument('--ablation', default=False, action='store_true', help='Ablation study mode (default False)')


    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"Pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')

    args = parser.parse_args()
    return args




if __name__ == '__main__':
    main()
