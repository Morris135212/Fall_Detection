import os
import sys
from pathlib import Path
import IPython
import cv2
import numpy as np
import torch
from utils.inference_utils import show_array

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import (increment_path, non_max_suppression, scale_coords, check_img_size)
from tracking import Tracks

tracks = Tracks(min_conf=0.5)
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
HISTORY = []
FLAG = False


@torch.no_grad()
def live_inference(yolo_model,
                   cnn3d,
                   duration=2,
                   source=0,  # file/dir/URL/glob, 0 for webcam
                   out_name="tmp",
                   imgsz=(416, 416),  # inference size (height, width)
                   detect_interval=1,
                   detect=True,
                   fall=False,
                   project=ROOT / 'runs/detect',  # save results to project/name
                   name='exp',  # save results to project/name
                   exist_ok=True,  # existing project/name ok, do not increment
                   half=False,  # use FP16 half-precision inference
                   **kwargs
                   ):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Load model
    stride, names, pt, jit, onnx, engine = yolo_model.stride, yolo_model.names, yolo_model.pt, yolo_model.jit, \
                                           yolo_model.onnx, yolo_model.engine
    yolo_model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    imgsz = tuple(check_img_size(imgsz, s=stride))  # check image size

    if source:
        v_cap = cv2.VideoCapture(source)  # initialize the video capture
    else:
        v_cap = cv2.VideoCapture(0)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # define encoding type
    size = (int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = 20.0
    video_out_path = str(save_dir / f"{out_name}.mp4")
    out = cv2.VideoWriter(video_out_path, fourcc, fps, size)  # initialize video writer

    frame_count = 0  # initialize frame count
    color = (0, 255, 255)
    texts = "PENDING"
    try:
        while True:
            frame_count += 1
            success, frame = v_cap.read()  # read frame from video
            if not success:
                print("Detect Finished")
                break
            if frame_count % detect_interval == 0:
                if detect:
                    frame = cv2.resize(frame, imgsz)
                    run_yolo(yolo_model,
                             frame,
                             device=device,
                             half=half,
                             **kwargs
                             )
                    if fall:
                        globalF = False
                        for track in tracks.tracks:
                            if track.current:
                                track.current = False
                                flag = track.__inference__(cnn3d, duration=duration)
                                globalF |= flag
                                if flag:
                                    # If fall, then draw red rectangle
                                    texts = f"FALL!!!"
                                    color = (0, 0, 255)
                                    track.__draw__(frame, color=color)
                                else:
                                    # If walk, then draw green rectangle
                                    texts = f"WALK"
                                    color = (0, 255, 0)
                                    track.__draw__(frame)
                        if globalF:
                            texts = f"FALL!!!"
                            color = (0, 0, 255)
                        if not tracks.tracks:
                            texts = f"NO PEOPLE DETECTED"
                            color = (0, 255, 255)
            cv2.putText(frame, texts, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        color, 2)
            IPython.display.clear_output(wait=True)  # clear the previous frame
            show_array(frame)
            frame = cv2.resize(frame, size)
            out.write(frame)
        v_cap.release()
        out.release()
    except KeyboardInterrupt:
        v_cap.release()
        out.release()
        print("Detect Finish")


@torch.no_grad()
def run_cnn3d(frames, model, device, class_names, duration=1, target="FALL"):
    global FLAG
    imgs = torch.stack(list(frames), dim=1).unsqueeze(0).to(device)
    out = model(imgs)
    pred_fall_idx = torch.argmax(out, dim=1)[0]
    pred_fall = class_names[pred_fall_idx]
    HISTORY.append(pred_fall)
    flag = False
    if len(HISTORY) >= duration:
        tmp_fall_history = HISTORY[-duration:]
        flag = all(t == target for t in tmp_fall_history)
    FLAG = flag
    return flag


@torch.no_grad()
def run_yolo(model,  # model.pt path(s)
             img,
             device=torch.device("cpu"),
             conf_thres=0.25,  # confidence threshold
             iou_thres=0.45,  # NMS IOU threshold
             max_det=1000,  # maximum detections per image
             classes=None,  # filter by class: --class 0, or --class 0 2 3
             agnostic_nms=False,  # class-agnostic NMS
             augment=False,  # augmented inference
             visualize=False,  # visualize features
             half=False,  # use FP16 half-precision inference
             ):
    # Load model
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA

    # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        im = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
    else:
        print("unknow image type")
        exit(-1)

    # Inference
    pred = model(im, augment=augment, visualize=visualize)
    boxes = []
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # Process predictions
    for i, det in enumerate(pred):  # per image
        im0 = img.copy()
        bbox_xyxy = []
        confs = []
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                bbox_xyxy.append(xyxy)
                confs.append(conf)
        tracks.update(torch.tensor(bbox_xyxy), confs, im0)
    return
