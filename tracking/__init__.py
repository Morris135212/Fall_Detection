from collections import namedtuple

import cv2
import torch
from dataset.cnn3d import DEFAULT_TRANSFORMS
from utils.metrics import bbox_iou
from utils.plots import save_one_box, compute_color_for_labels
from collections import deque
from PIL import Image


Detection = namedtuple("Detection", ["bbox", "conf"])
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Track:
    def __init__(self, idx, bbox, conf, img):
        self.idx = idx
        self.age = 0
        self.box_history = deque(maxlen=10)
        self.__update__(bbox, conf, img)
        self.current = False
        self.fall_history = []

    def __update__(self, bbox, conf, img):
        self.bbox = bbox
        self.conf = conf
        crops = save_one_box(bbox, img, save=False, BGR=False)
        self.box_history.append(DEFAULT_TRANSFORMS(Image.fromarray(crops)))

    def __draw__(self, img, offset=(0, 0), color=None):
        x1, y1, x2, y2 = [int(i) for i in self.bbox]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        if not color:
            color = compute_color_for_labels(self.idx)
        label = '{}{:d}'.format("", self.idx)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        return img

    def __inference__(self, model, duration=2, target_idx=1):
        if len(self.box_history) < 10:
            return False
        imgs = torch.stack(list(self.box_history), dim=1).unsqueeze(0).to(device)
        out = model(imgs)
        pred_fall_idx = torch.argmax(out, dim=1)[0]
        self.fall_history.append(pred_fall_idx.item())
        flag = False
        if len(self.fall_history) >= duration:
            tmp_fall_history = self.fall_history[-duration:]
            flag = all(t == target_idx for t in tmp_fall_history)
        return flag

    def __str__(self):
        return f"{self.idx} {self.bbox}"


class Tracks:
    def __init__(self, min_iou=0.5, min_conf=0.3, max_age=30):
        self.min_iou = min_iou
        self.min_conf = min_conf
        self.max_age = max_age
        self.latest_id = 0
        self.tracks = []
        self.detections = []

    def update(self, bboxes_xywh, confidences, img):
        """
        The input bounding boxes must be in the form [xmin, ymin, xmax, ymax]
        """

        def match(idx):
            for j, track in enumerate(self.tracks):
                if bbox_iou(detection.bbox, track.bbox, x1y1x2y2=True) > self.min_iou and not vis[j]:
                    vis[j] = 1
                    if matches[j] == -1 or match(matches[j]):
                        matches[j] = idx
                        return True
            return False

        if not self.tracks:
            for i, (box, conf) in enumerate(zip(bboxes_xywh, confidences)):
                self.tracks.append(Track(i, box, conf, img))
                self.latest_id += i
        else:
            self.detections = []
            for i, (box, conf) in enumerate(zip(bboxes_xywh, confidences)):
                self.detections.append(Detection(box, conf))

            matches = [-1 for _ in range(len(self.tracks))]  # Tracks' corresponding detections
            for i, detection in enumerate(self.detections):
                vis = [0 for _ in range(len(self.tracks))]  # Whehter tracks has been visited
                match(i)
            for i, track in enumerate(self.tracks):
                if matches[i] == -1:
                    track.age += 1
                    track.current = False
                else:
                    track.__update__(self.detections[matches[i]].bbox, self.detections[matches[i]].conf, img)
                    track.current = True
                    track.age = 0

            # Delete all tracks with age greater than desired age
            for i, track in enumerate(self.tracks):
                if track.age > self.max_age:
                    del self.tracks[i]

            # Add ummatched detections into tracks
            matches = set(matches)
            for i, detection in enumerate(self.detections):
                if i not in matches and detection.conf > self.min_conf:
                    self.tracks.append(Track(self.latest_id + 1, detection.bbox, detection.conf, img))
                    self.latest_id += 1
