# import os
from collections import defaultdict
import cv2
import requests
def send_analysis_result_to_server(track_id, gender, age, race, behavior, obj_list):
    data = {
        "track_id": track_id,
        "gender": gender,
        "age": age,
        "race": race,
        "behavior": behavior,  # e.g. "stay"
        "objects": obj_list
    }

    try:
        r = requests.post("http://<服务器IP>:5000/save_result", json=data, timeout=5)
        print("✅ 服务器保存成功" if r.status_code == 200 else "❌ 保存失败")
    except Exception as e:
        print("⚠️ 网络错误：", e)

def send_image_to_server(image_array, save_name, behavior='stay'):
    url = "http://<服务器IP>:5000/upload_image"
    _, img_encoded = cv2.imencode('.jpg', image_array)
    files = {
        'image': (save_name, img_encoded.tobytes(), 'image/jpeg')
    }
    data = {
        'save_name': save_name,
        'behavior': behavior  # 👈 新增字段
    }

    try:
        res = requests.post(url, files=files, data=data, timeout=5)
        if res.status_code == 200 and res.json().get('status') == 'ok':
            print(f"✅ [{behavior}] 图片已上传并保存为：{save_name}")
        else:
            print("❌ 上传失败：", res.text)
    except Exception as e:
        print("⚠️ 上传出错：", e)


def analyze_stay_behavior(image_buffer, analyzed_stay_ids, keypoints_buffer, stay_ids,
                          args, fairface_model, face_app, analyzed_id_attributes,
                          person_demo, get_person_results, detect_hand_objects,
                          pose_results_window, center_size=50):
    center_crop_buffer = defaultdict(list)
    full_crop_buffer = defaultdict(list)
    valid_tids = []

    for tid in stay_ids:
        if tid in analyzed_stay_ids or tid not in keypoints_buffer:
            continue

        matched_entry_count = 0
        for entries in image_buffer.values():
            for entry in entries:
                frame_idx = entry['frame_idx']
                img = entry['image']
                h, w = img.shape[:2]

                found_pose = None
                for pose in pose_results_window.get(frame_idx, []):
                    if pose.get('track_id') == tid:
                        found_pose = pose
                        break

                if found_pose is None or len(found_pose['keypoints']) < 11:
                    continue

                keypoints = found_pose['keypoints']
                x1, y1, x2, y2 = map(int, found_pose['bbox'][:4])
                matched_entry_count += 1

                person_crop = img[y1:y2, x1:x2]
                if person_crop is not None and person_crop.size > 0:
                    full_crop_buffer[tid].append({'frame_idx': frame_idx, 'image': person_crop})
                    # save_path_person = os.path.join('outputs/stay_person', f"tid{tid}_f{frame_idx}.jpg")
                    # os.makedirs(os.path.dirname(save_path_person), exist_ok=True)
                    # cv2.imwrite(save_path_person, person_crop)
                    

                lx, ly = keypoints[9][:2]
                rx, ry = keypoints[10][:2]
                cx, cy = int((lx + rx) / 2), int((ly + ry) / 2)
                hx1 = max(cx - center_size, 0)
                hy1 = max(cy - center_size, 0)
                hx2 = min(cx + center_size, w)
                hy2 = min(cy + center_size, h)

                center_crop = img[hy1:hy2, hx1:hx2]
                if center_crop is not None and center_crop.size > 0:
                    center_crop_buffer[tid].append({'frame_idx': frame_idx, 'image': center_crop})
                    # save_path_center = os.path.join('outputs/stay_center', f"tid{tid}_f{frame_idx}.jpg")
                    # os.makedirs(os.path.dirname(save_path_center), exist_ok=True)
                    # cv2.imwrite(save_path_center, center_crop)
                    send_image_to_server(center_crop, f"tid{tid}_f{frame_idx}.jpg", behavior="stay")


        if matched_entry_count >= 1:
            valid_tids.append(tid)

    if full_crop_buffer:
        frames_stay_dict = defaultdict(list)
        for tid, entries in full_crop_buffer.items():
            for entry in entries:
                frames_stay_dict[tid].append((entry['frame_idx'], entry['image']))

        results_stay_id = person_demo(frames_stay_dict, args, fairface_model, face_app)
        stay_result = get_person_results(results_stay_id, behavior_type='stay')
        object_dict = detect_hand_objects(center_crop_buffer, model_path='yolov8n.pt', output_dir=None)
        print("[DEBUG] Detected objects:", object_dict)

        if stay_result:
            for tid in valid_tids:
                analyzed_stay_ids.add(tid)

        for track_id, gender, age, race, behavior in stay_result:
            if gender != 'U' and age != 'U':
                analyzed_id_attributes[track_id] = (gender, age)
            obj_list = object_dict.get(track_id, [])
            # write_one_result_to_csv(track_id, gender, age, race, behavior, obj_list, output_path)
            send_analysis_result_to_server(track_id, gender, age, race, behavior, obj_list)
            print(f"[DEBUG] Stay--Writing to db: {track_id}, {gender}, {age}, {race}, {behavior}, {obj_list}")

        center_crop_buffer.clear()
        full_crop_buffer.clear()


def analyze_pick_behavior(det_id, image_buffer, analyzed_pick_ids, keypoints_buffer, args,
                          fairface_model, face_app, h, w, analyzed_id_attributes,
                          person_demo, get_person_results, detect_hand_objects,
                          pose_results_window, hand_box_size=50):
    pick_item = []
    hand_crop_buffer = defaultdict(list)
    full_crop_buffer = defaultdict(list)
    valid_tids = []


    for tid in det_id:
        if tid in analyzed_pick_ids or tid not in keypoints_buffer:
            continue

        match_count = 0

        for entries in image_buffer.values():
            for entry in entries:
                frame_idx = entry['frame_idx']
                frame_img = entry['image']
                h, w = frame_img.shape[:2]

                found_pose = None
                for pose in pose_results_window.get(frame_idx, []):
                    if pose.get('track_id') == tid:
                        # print(tid, "pose涓瓨璇d")
                        found_pose = pose
                        break

                if found_pose is None or len(found_pose['keypoints']) < 11:
                    continue

                keypoints = found_pose['keypoints']
                x1, y1, x2, y2 = map(int, found_pose['bbox'][:4])
                match_count += 1

                person_img = frame_img[y1:y2, x1:x2]
                if person_img is not None and person_img.size > 0:
                    full_crop_buffer[tid].append({'frame_idx': frame_idx, 'image': person_img})
                    # save_path_center = os.path.join('outputs/pick_person', f"tid{tid}_f{frame_idx}.jpg")
                    # os.makedirs(os.path.dirname(save_path_center), exist_ok=True)
                    # cv2.imwrite(save_path_center, person_img)

                lx, ly = keypoints[9][:2]
                rx, ry = keypoints[10][:2]
                cx, cy = int((lx + rx) / 2), int((ly + ry) / 2)
                hx1 = max(cx - hand_box_size, 0)
                hy1 = max(cy - hand_box_size, 0)
                hx2 = min(cx + hand_box_size, w)
                hy2 = min(cy + hand_box_size, h)

                hand_crop = frame_img[hy1:hy2, hx1:hx2]
                if hand_crop is not None and hand_crop.size > 0:
                    hand_crop_buffer[tid].append({'frame_idx': frame_idx, 'image': hand_crop})
                    # save_path_center = os.path.join('outputs/pick_hand', f"tid{tid}_f{frame_idx}.jpg")
                    # os.makedirs(os.path.dirname(save_path_center), exist_ok=True)
                    # cv2.imwrite(save_path_center, hand_crop)
                    send_image_to_server(hand_crop, f"tid{tid}_f{frame_idx}.jpg", behavior="pick")


        if match_count >= 6:
            valid_tids.append(tid)

    if full_crop_buffer:
        frames_dict = defaultdict(list)
        for tid, entries in full_crop_buffer.items():
            for entry in entries:
                frames_dict[tid].append((entry['frame_idx'], entry['image']))

        results_pick_id = person_demo(frames_dict, args, fairface_model, face_app)
        pick_result = get_person_results(results_pick_id, behavior_type='pick')
        objects_dict = detect_hand_objects(hand_crop_buffer, model_path='yolov8n.pt', output_dir=None)
        print("[DEBUG] Detected objects:", objects_dict)
        pick_item.append(objects_dict)

        if pick_result:
            for tid in valid_tids:
                analyzed_pick_ids.add(tid)

        for track_id, gender, age, race, behavior in pick_result:
            if gender != 'U' and age != 'U':
                analyzed_id_attributes[track_id] = (gender, age)
            obj_list = objects_dict.get(track_id, [])
            # print("[DEBUG] Detected objects:", obj_list)
            # write_one_result_to_csv(track_id, gender, age, race, behavior, obj_list, output_path)
            send_analysis_result_to_server(track_id, gender, age, race, behavior, obj_list)
            print(f"[DEBUG] Pick -- Writing to db: {track_id}, {gender}, {age}, {race}, {behavior}, {obj_list}")

        hand_crop_buffer.clear()
        full_crop_buffer.clear()
