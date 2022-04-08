import os
import sys
from pathlib import Path
import IPython
import cv2
import torch
from IPython.core.display import display
from PIL import Image
from collections import deque

from dataset.cnn3d import DEFAULT_TRANSFORMS
from models.common import DetectMultiBackend

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from dataset.yolo.datasets import LoadImages
from utils.general import (increment_path, non_max_suppression, scale_coords, check_img_size)
from utils.plots import Annotator, colors, save_one_box


@torch.no_grad()
def live_inference(yolo_model,
                   cnn3d,
                   class_names,
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
    imgsz = check_img_size(imgsz, s=stride)  # check image size

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
    box_que = deque(maxlen=10)
    Fall_History = []
    try:
        while True:
            frame_count += 1
            success, frame = v_cap.read()  # read frame from video
            if not success:
                print("Detect Finished")
                break
            if frame_count % detect_interval == 0:
                save_path = str(save_dir / f"{out_name}.png")
                cv2.imwrite(save_path, frame)
                if detect:
                    frame, boxes = run_yolo(yolo_model,
                                            device=device,
                                            source=save_dir / f"{out_name}.png",
                                            imgsz=imgsz,
                                            project=project,
                                            name=name,
                                            exist_ok=exist_ok,
                                            half=half,
                                            **kwargs
                                            )
                    if fall:
                        """
                        TODO
                        Object tracking and Multi Object Fall detection
                        """
                        box_que.append(DEFAULT_TRANSFORMS(Image.fromarray(boxes[0])))
                        if len(box_que) == 10:
                            flag = run_cnn3d(box_que, cnn3d, device, class_names,
                                             Fall_History=Fall_History, duration=duration)
                            pass
                    out.write(frame)
        v_cap.release()
        out.release()
    except KeyboardInterrupt:
        v_cap.release()
        out.release()
        print("Detect Finish")


@torch.no_grad()
def run_cnn3d(frames, model, device, class_names, Fall_History=[], duration=1):
    imgs = torch.stack(list(frames), dim=1).unsqueeze(0).to(device)
    pred_fall_idx = torch.argmax(model(imgs), dim=1)[0]
    pred_fall = class_names[pred_fall_idx]
    Fall_History.append(pred_fall)
    if len(Fall_History) >= duration:
        tmp_fall_history = Fall_History[-duration:]
        if all(tmp_fall_history) == "FALL":
            print("DETECTING FALL ACTION")
    return True


@torch.no_grad()
def run_yolo(yolo_model,  # model.pt path(s)
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

                frame2 = Image.fromarray(boxes[0])
                display(frame2)
    return im0, boxes
