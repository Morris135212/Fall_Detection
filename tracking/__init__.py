from collections import namedtuple
from utils.metrics import bbox_iou

Detection = namedtuple("Detection", ["bbox", "conf"])


class Track:
    def __init__(self, idx, bbox, conf):
        self.idx = idx
        self.bbox = bbox
        self.conf = conf
        self.age = 0
        self.box_history = [bbox]
        self.current = False

    def __update__(self, bbox, conf):
        self.bbox = bbox
        self.conf = conf
        self.box_history.append(bbox)

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

    def update(self, bboxes_xywh, confidences):
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
                self.tracks.append(Track(i, box, conf))
                self.latest_id = i
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
                    track.__update__(self.detections[matches[i]].bbox, self.detections[matches[i]].conf)
                    track.current = True

            # Delete all tracks with age greater than desired age
            for i, track in enumerate(self.tracks):
                if track.age > self.max_age:
                    del self.tracks[i]

            # Add ummatched detections into tracks
            matches = set(matches)
            for i, detection in enumerate(self.detections):
                if i not in matches and detection.conf > self.min_conf:
                    self.tracks.append(Track(self.latest_id + 1, detection.bbox, detection.conf))
                    self.latest_id += 1
