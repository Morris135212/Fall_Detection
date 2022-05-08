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
    """
    This class will track object in the video
    """

    def __init__(self, idx, bbox, conf, img):
        """
        :param idx:int index of current track
        :param bbox:torch.tensor bounding box of current moment, this variable will be updated if new status been updated
        :param conf:float confidence of current moment, this variable will be updated if new status been updated
        :param img:np.array image
        """
        self.idx = idx
        self.age = 0
        self.box_history = deque(maxlen=10)  # data flow that will be fed into 3dcnn
        self.__update__(bbox, conf, img)
        self.current = False
        self.fall_history = []

    def __update__(self, bbox, conf, img):
        """
        For each matched tracks at moment t, the latest bounding box, confidence and current image will be passed and updated
        :param bbox:torch.tensor bounding box of current moment
        :param conf:float confidence for current object
        :param img: np.array images
        """
        self.bbox = bbox
        self.conf = conf
        crops = save_one_box(bbox, img, save=False, BGR=False)  # Crops current object from the img
        # Transform it into form where can be fed into 3dcnn
        self.box_history.append(DEFAULT_TRANSFORMS(Image.fromarray(crops)))

    def __draw__(self, img, offset=(0, 0), color=None):
        """
        Draw bounding box to the img
        """
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
        """
        Do inference on 3dcnn, detecting whether current track have fall action
        :param model:torch.nn.Module 3dcnn model
        :param duration:int the number of consecutive moment that if the output is target_idx will be viewed as
        the happens of the action
        :param target_idx:int the index where we want to identified. The desired output that we want from the model
        """
        # In order to be fed into 3dcnn, we must have 10 frames of images
        if len(self.box_history) < 10:
            return False
        # Stack 10 images
        imgs = torch.stack(list(self.box_history), dim=1).unsqueeze(0).to(device)  # size: (10, 3, 112, 112)
        # output from model
        out = model(imgs)  # size: (1, # of cls)

        pred_fall_idx = torch.argmax(out, dim=1)[0]
        self.fall_history.append(pred_fall_idx.item())
        flag = False
        if len(self.fall_history) >= duration:
            tmp_fall_history = self.fall_history[-duration:]
            # Judge whehter in the past duration period the output of the model all equals to target idx
            # if yes, then the flag = True: the action actually happened
            # if no, then the flag = False: the action not happened
            flag = all(t == target_idx for t in tmp_fall_history)
        return flag

    def __str__(self):
        return f"{self.idx} {self.bbox}"


class Tracks:
    """
    This class include the information of all track at real time at the video stream
    This class also respond for matching current detection with previous tracks
    """

    def __init__(self, min_iou=0.5, min_conf=0.3, max_age=30):
        """
        :param min_iou:float minimum iou, if two boxes have iou > min_iou will be viewed as connected,
        later with Hungarian algorithm we can find the match.
        :param min_conf:float a new detection must have confidence > min_conf
        :param max_age: if a track have age > max_age, than this track will be deleted
        """
        self.min_iou = min_iou
        self.min_conf = min_conf
        self.max_age = max_age
        self.latest_id = 0  # A new track will be assigned with this latest_id
        self.tracks = []  # all tracks are included in this list
        self.detections = []  # detection at current moment will be included in this list

    def update(self, bboxes_xyxy, confidences, img):
        """
        Match current detections with the tracks
        There are three cases at last:
        1. tracks are matched - update the new bounding boxes and confidence to the corresponding track
        2. tracks doesn't match - updated the tracks with track.age += 1
        3. detections doesn't match - add new track to tracks list
        :param bboxes_xyxy:torch.tensor The input bounding boxes must be in the form [xmin, ymin, xmax, ymax]
        """

        def match(idx):
            """
            Hungarian algorithm
            """
            for j, track in enumerate(self.tracks):
                if bbox_iou(detection.bbox, track.bbox, x1y1x2y2=True) > self.min_iou and not vis[j]:
                    vis[j] = 1
                    if matches[j] == -1 or match(matches[j]):
                        matches[j] = idx
                        return True
            return False

        if not self.tracks:  # If tracks are empty, then add new tracks
            for i, (box, conf) in enumerate(zip(bboxes_xyxy, confidences)):
                self.tracks.append(Track(self.latest_id, box, conf, img))
                self.latest_id += i
        else:
            self.detections = []  # initialize current detection
            for i, (box, conf) in enumerate(zip(bboxes_xyxy, confidences)):
                self.detections.append(Detection(box, conf))  # Append necessary information to detection

            # Run Hungarian algorithm
            matches = [-1 for _ in range(len(self.tracks))]  # Tracks' corresponding detections
            for i, detection in enumerate(self.detections):
                vis = [0 for _ in range(len(self.tracks))]  # Whehter tracks has been visited
                match(i)
            # Find the match between tracks and current detection
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
