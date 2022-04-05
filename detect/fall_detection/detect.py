import os
import sys
from pathlib import Path
import IPython
import cv2
import torch
from IPython.core.display import display
from PIL import Image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from dataset.yolo.datasets import LoadImages
from utils.general import (increment_path, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors, save_one_box


@torch.no_grad()
def run(yolo_model,  # model.pt path(s)
        device=torch.device("cpu"),
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        view_img=False,  # show results
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):
    # Load model
    stride, names, pt, jit, onnx, engine = yolo_model.stride, yolo_model.names, yolo_model.pt, yolo_model.jit, yolo_model.onnx, yolo_model.engine

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = yolo_model(im, augment=augment, visualize=False)
        """
        TODO
        Object tracking
        """
        boxes = []
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            imc = im0.copy()  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    box = save_one_box(xyxy, imc, save=False, BGR=False)
                    boxes.append(box)
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            if view_img:
                frame_array = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame_array)
                IPython.display.clear_output(wait=True)  # clear the previous frame
                display(frame)
    return im0, boxes
