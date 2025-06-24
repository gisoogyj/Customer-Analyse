# -*- coding: utf-8 -*-
import os
import sys
import cv2
import torch
import numpy as np
from torchvision import transforms
# from insightface.app import FaceAnalysis
from collections import Counter, defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

def person_demo(frames_id, args, fairface_model, face_app):
    # save_dir = Path(args.project) / args.name
    # save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cpu')

    results = []

    # 閬嶅巻姣忎釜 track_id 鍜屽搴�? frame_idx -> bbox
    for track_id, frame_list in frames_id.items():
         for frame_idx, img in frame_list:
            # face_results = face_app.get(img)
            face_results = face_app.get(img)
            if not face_results:
                continue
    
            if len(face_results) > 1:
                print(f"[!]: frame {frame_idx}, track {track_id}")
            face = face_results[0]

            try:
                fx1, fy1, fx2, fy2 = map(int, face.bbox)
                face_crop = img[fy1:fy2, fx1:fx2]
                face_crop = cv2.resize(face_crop, (224, 224))
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_tensor = fairface_transform(face_crop).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = fairface_model(face_tensor).cpu().numpy().squeeze()

                race = int(np.argmax(output[:7]))
                gender = int(np.argmax(output[7:9]))
                age = int(np.argmax(output[9:18]))

                gender_str = gender_map[gender] if gender in [0, 1] else 'U'
                if gender_str == 'M':
                    fewshot_result = predict_gender_fewshot(face.embedding, prototypes, id=track_id)
                    if fewshot_result == 'neutral_female':
                        gender_str = 'F-n'

                age_str = age_map[age] if 0 <= age < len(age_map) else 'U'
                race_str = race_map[race] if 0 <= race < len(race_map) else 'U'

                results.append([track_id, frame_idx, gender_str, age_str, race_str])
            except Exception as e:
                print(f"[!] 澶勭悊澶辫触 track {track_id}, frame {frame_idx}锛岄敊璇�?: {e}")

    print("[DEBUG] person_demo results:", results)

    return results



def majority_vote(attributes):
    """鍘婚櫎鏃犳晥(U,U,U)锛屽啀鍙栦紬鏁�?"""
    filtered = [attr for attr in attributes if attr != ('U','U','U')]
    if not filtered:
        return 'U', 'U', 'U'
    gender_mode = Counter([x[0] for x in filtered]).most_common(1)[0][0]
    age_mode = Counter([x[1] for x in filtered]).most_common(1)[0][0]
    race_mode = Counter([x[2] for x in filtered]).most_common(1)[0][0]
    return gender_mode, age_mode, race_mode

def get_person_results(results, behavior_type='unknown'):
    group_dict = defaultdict(list)
    for r in results:
        if len(r) == 5:
            track_id, frame_idx, gender, age, race = r
            group_dict[track_id].append((gender, age, race))

    # 瀵规瘡缁勮绠椾紬鏁�?
    final_results = []
    for track_id, attr_list in group_dict.items():
        gender, age, race = majority_vote(attr_list)
        final_results.append([track_id, gender, age, race, behavior_type])

    print("[DEBUG] person_demo results:", final_results)

    return final_results
