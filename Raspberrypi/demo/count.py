from collections import defaultdict
import os
import csv
import cv2
import torch
import numpy as np
from torchvision import transforms
from fewshot_gender import predict_gender_fewshot

prototypes = np.load("Pretrained/proto_new.npy", allow_pickle=True).item()
gender_map = ["M", "F"]
age_map = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
race_map = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']

fairface_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def compute_containment(inner, outer):
    xi1, yi1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    xi2, yi2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter_area / inner_area if inner_area > 0 else 0

def analyze_crowd(
    frame_idx, fps, frame, boxes_with_ids,
    fairface_model, face_app,
    track_id_attributes
):
    male_count = 0
    female_count = 0
    age_stats = defaultdict(int)

    for tid, box in boxes_with_ids:
        print("tid:", tid)
        # 已分析过，则直接使用缓存结果
        if tid in track_id_attributes:
            gender_str, age_str = track_id_attributes[tid]
            print("analysed")
        else:
            # 尚未分析，需执行人脸识别 + 属性分析
            print("not analysed")
            x1, y1, x2, y2 = map(int, box)
            print("person-x-y", x1, y1, x2, y2)
            person_crop = frame[y1:y2, x1:x2]
            face_results = face_app.get(person_crop)
            if face_results:
                print("face exist")
            if not face_results:
                print("face not xist")
                continue

            face = face_results[0]
            try:
                fx1, fy1, fx2, fy2 = map(int, face.bbox)
                print("face-x-y", x1, y1, x2, y2)
                face_crop = person_crop[fy1:fy2, fx1:fx2]
                face_crop = cv2.resize(face_crop, (224, 224))
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_tensor = fairface_transform(face_crop).unsqueeze(0).to(fairface_model.device)
                if face_crop:
                    print("face crop valid")
                with torch.no_grad():
                    output = fairface_model(face_tensor).cpu().numpy().squeeze()

                race = int(np.argmax(output[:7]))
                gender = int(np.argmax(output[7:9]))
                age = int(np.argmax(output[9:18]))

                gender_str = gender_map[gender] if gender in [0, 1] else 'U'
                age_str = age_map[age] if 0 <= age < len(age_map) else 'U'
                race_str = race_map[race] if 0 <= race < len(race_map) else 'U'

                track_id_attributes[tid] = (gender_str, age_str, race_str)
                print("analyse", track_id_attributes[tid])
            except Exception as e:
                # print(f"[!] Face crop error: {e}")
                continue

        # 汇总统计
        if gender_str.startswith('M'):
            male_count += 1
        elif gender_str.startswith('F'):
            female_count += 1
        if age_str != 'U':
            age_stats[age_str] += 1

    # 写入 CSV（每 5 秒一行）
    sec_start = (frame_idx // fps) * 5
    sec_range = f"{sec_start}-{sec_start + 5}s"

    with open('crowd_summary.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if frame_idx == 0:
            writer.writerow(['TimeRange', 'Total', 'Male', 'Female'] + age_map)
        row = [sec_range, len(boxes_with_ids), male_count, female_count]
        row += [age_stats.get(age, 0) for age in age_map]
        writer.writerow(row)

    print(f"[INFO] Time {sec_range}: {len(boxes_with_ids)} total, M:{male_count}, F:{female_count}")


# def analyze_crowd(
#     frame_idx, fps, frame, boxes, fairface_model, face_app
# ):
#     person_boxes = []
#     for box in boxes:
#         x1, y1, x2, y2, _, _ = box
#         person_boxes.append([x1, y1, x2, y2])

#     male_count = 0
#     female_count = 0
#     age_stats = defaultdict(int)
#     person_id = 1
#     print(f"[DEBUG] {len(person_boxes)} persons after filtering")

#     for box in person_boxes:
#         x1, y1, x2, y2 = map(int, box)
#         print(x1, y1, x2, y2)
#         person_crop = frame[y1:y2, x1:x2]
#         os.makedirs('debug_crops', exist_ok=True)
#         cv2.imwrite(f'debug_crops/frame_{frame_idx}_id_{person_id}.jpg', person_crop)

#         # face_results = face_app.get(person_crop)
#         face_results = face_app.get(person_crop)
#         if face_results:
#             face = face_results[0]
#             try:
#                 fx1, fy1, fx2, fy2 = map(int, face.bbox)
#                 face_crop = person_crop[fy1:fy2, fx1:fx2]
#                 face_crop = cv2.resize(face_crop, (224, 224))
#                 face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
#                 face_tensor = fairface_transform(face_crop).unsqueeze(0).to('cpu')

#                 with torch.no_grad():
#                     output = fairface_model(face_tensor).cpu().numpy().squeeze()

#                 race = int(np.argmax(output[:7]))
#                 gender = int(np.argmax(output[7:9]))
#                 age = int(np.argmax(output[9:18]))

#                 gender_str = gender_map[gender] if gender in [0, 1] else 'U'
#                 if gender_str == 'M':
#                     fewshot_result = predict_gender_fewshot(face.embedding, prototypes, id=person_id)
#                     if fewshot_result == 'neutral_female':
#                         gender_str = 'F-n'

#                 age_str = age_map[age] if 0 <= age < len(age_map) else 'U'

#                 if gender_str.startswith('M'):
#                     male_count += 1
#                 elif gender_str.startswith('F'):
#                     female_count += 1

#                 if age_str != 'U':
#                     age_stats[age_str] += 1

#                 person_id += 1
#             except Exception as e:
#                 print(f"[!] Face crop error: {e}")

#     sec_start = (frame_idx // int(fps * 5)) * 5
#     # sec_range = f"{sec_start}-{sec_start + 5}s"

#     with open('crowd_summary.csv', 'a', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         if frame_idx == 0:
#             writer.writerow(['TimeRange', 'Total', 'Male', 'Female'] + age_map)
#         row = [sec_start, len(person_boxes), male_count, female_count]
#         row += [age_stats.get(age, 0) for age in age_map]
#         writer.writerow(row)

#     print(f"[INFO] Time {sec_start}: {len(person_boxes)} total, M:{male_count}, F:{female_count}")
