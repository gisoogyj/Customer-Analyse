# from pathlib import Path
# from collections import defaultdict
from ultralytics import YOLO

def detect_hand_objects(hand_crop_buffer, model_path='yolov8s.pt', output_dir=None):
    objects_dict = {}
    model = YOLO(model_path)

    for track_id, frame_list in hand_crop_buffer.items():
        detected_objects = set()
        for entry in frame_list:
            img = entry['image']
            if img is None:
                continue

            results = model.predict(img, verbose=False)
            if not results:
                continue

            result = results[0]
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            class_names = result.names

            for cls_id in cls_ids:
                obj_name = class_names[cls_id]
                detected_objects.add(obj_name)
                # if obj_name != 'person':
                #     detected_objects.add(obj_name)
        objects_dict[track_id] = list(detected_objects)

    return objects_dict

# def detect_hand_objects(hand_crop_buffer, model_path='yolov8s.pt', output_dir=None):
#     objects_dict = {}
#     model = YOLO(model_path)

#     for track_id, frame_list in hand_crop_buffer.items():
#         detected_objects = []
#         for entry in frame_list:
#             img = entry['image']
#             if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
#                 print(f"[WARNING] Empty or invalid image for tid={track_id}, skipping.")
#                 continue
#             try:
#                 results = model.predict(img, verbose=False)
#                 if results:
#                     classes = results[0].boxes.cls.cpu().numpy()
#                     names = results[0].names
#                     for cls_id in classes:
#                         detected_objects.append(names[int(cls_id)])
#             except Exception as e:
#                 print(f"[ERROR] YOLO inference failed for tid={track_id}: {e}")
#                 continue

#         # 去重
#         objects_dict[track_id] = list(set(detected_objects))

#     return objects_dict