import numpy as np

def detection_inference(model, frame):
    # print('Performing Human Detection for each frame')
    r = model.predict(frame, verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    # masks = r.masks.data.cpu().numpy() if r.masks is not None else []
    cls_ids = r.boxes.cls.cpu().numpy()
    
    # person_masks = []
    detections = []
    for box, conf, cls_id in zip(boxes, confs, cls_ids):
        if int(cls_id) == 0:
            x1, y1, x2, y2 = box
            detections.append([x1, y1, x2, y2, conf, int(cls_id)])
            # person_masks.append(mask)

    if len(detections) == 0:
        return np.empty((0, 6))
    return np.array(detections)
