import os
import sys
from pathlib import Path
import IPython
import cv2
import numpy as np
import torch
from IPython.core.display import display
from PIL import Image
import torchvision.transforms as T

from models.common import DetectMultiBackend
from tracking import Tracks
from utils.inference_utils import show_array

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from dataset.yolo.datasets import LoadImages
from utils.general import (check_img_size, increment_path, non_max_suppression, scale_coords, bbox_rel)
from utils.plots import Annotator, colors, save_one_box, compute_color_for_labels, draw_boxes

tracks = Tracks(min_conf=0.5)
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
HISTORY = []
DATA_SHAPE = (416, 416)
DEFAULT_TRANSFORMS = T.Compose([
    T.Resize(DATA_SHAPE),
    T.ToTensor()
])


@torch.no_grad()
def live_infer_hub(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
                   source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
                   project=ROOT / 'runs/detect',  # save results to project/name
                   name='exp',
                   detect_interval=1,
                   imgsz=(640, 640),  # inference size (height, width)
                   vis=False,
                   save_crop=False,  # save cropped prediction boxes
                   line_thickness=3
                   ):
    model = torch.hub.load(str(ROOT),
                           'custom',
                           path=weights,
                           source='local')
    save_dir = increment_path(Path(project) / name, exist_ok=False)  # increment run
    if source:
        v_cap = cv2.VideoCapture(source)  # initialize the video capture
    else:
        v_cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'VP90')  # define encoding type
    fps = 30.0  # define frame rate
    video_dims = imgsz  # define output dimensions
    out = cv2.VideoWriter(f"{save_dir}", fourcc, fps, video_dims)  # initialize video writer
    frame_count = 0  # initialize frame count
    try:
        while True:
            frame_count += 1  # increment frame count
            success, frame = v_cap.read()  # read frame from video
            if not success:
                # raise Exception("Video Initialization error")
                break
            if frame_count % detect_interval == 0:
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert raw frame from BGR to RGB
                # img = torch.from_numpy(frame).to(device, dtype=torch.float)
                # img = img / 255
                # if len(img.shape) == 3:
                #     img = img.unsqueeze(0)  # expand for batch dim
                # pred = model(img)
                cv2.imwrite(save_dir / "tmp.png", frame)
                output = model(save_dir / "tmp.png")
                xyxy = output.pandas().xyxy[0]
                if save_crop:
                    output.crop()
                for i, row in xyxy.iterrows():
                    xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
                    frame = cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0),
                                          line_thickness)

            out.write(frame)  # write detected frame to output video
            if vis:
                frame_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame_array)
                display(frame)
        v_cap.release()
        out.release()
    except KeyboardInterrupt:
        v_cap.release()
        out.release()


@torch.no_grad()
def infer(model,
          img_path,
          line_thickness=3,
          project=ROOT / 'runs/detect',  # save results to project/name
          name='exp',
          imgname="tmp.png",
          exist_ok=True,
          vis=True,
          save=True):
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run

    output = model(img_path)
    xyxy = output.pandas().xyxy[0]
    frame = cv2.imread(img_path)

    for i, row in xyxy.iterrows():
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        frame = cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), line_thickness)

    if vis:
        frame_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_array)
        display(img)
    if save:
        save_path = str(save_dir / imgname)
        print(save_path)
        cv2.imwrite(save_path, frame)


@torch.no_grad()
def live_inference(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
                   source=0,  # file/dir/URL/glob, 0 for webcam
                   data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
                   out_name="tmp",
                   imgsz=(416, 416),  # inference size (height, width)
                   detect_interval=1,
                   detect=True,
                   project=ROOT / 'runs/detect',  # save results to project/name
                   name='exp',  # save results to project/name
                   exist_ok=True,  # existing project/name ok, do not increment
                   half=False,  # use FP16 half-precision inference
                   **kwargs
                   ):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = DetectMultiBackend(weights, device=device, data=data)
    # Load model
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
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
                    frame = run(model,
                                device=device,
                                source=save_dir / f"{out_name}.png",
                                imgsz=imgsz,
                                project=project,
                                name=name,
                                exist_ok=exist_ok,
                                half=half,
                                **kwargs
                                )
                    out.write(frame)
        v_cap.release()
        out.release()
    except KeyboardInterrupt:
        v_cap.release()
        out.release()
        print("Detect Finish")


@torch.no_grad()
def run(model,  # model.pt path(s)
        device=torch.device("cpu"),
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_img=True,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=True,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):
    source = str(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine

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

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img or save_img:
                frame_array = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                if save_img:
                    save_path = save_dir / source.split("/")[-1]
                    frame = Image.fromarray(frame_array)
                    frame.save(save_path)
                if view_img:
                    IPython.display.clear_output(wait=True)  # clear the previous frame
                    # display(frame)
                    show_array(frame_array)
                    # IPython.display.clear_output(wait=True)  # clear the previous frame
    return im0


@torch.no_grad()
def live_inference_fed_img(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
                           source=0,  # file/dir/URL/glob, 0 for webcam
                           data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
                           out_name="tmp",
                           imgsz=(416, 416),  # inference size (height, width)
                           detect_interval=1,
                           detect=True,
                           project=ROOT / 'runs/detect',  # save results to project/name
                           name='exp',  # save results to project/name
                           tracking=False,
                           exist_ok=True,  # existing project/name ok, do not increment
                           half=False,  # use FP16 half-precision inference
                           **kwargs
                           ):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = DetectMultiBackend(weights, device=device, data=data)
    # Load model
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
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
    try:
        while True:
            kwargs["frame_id"] = frame_count
            success, frame = v_cap.read()  # read frame from video
            if not success:
                print("Detect Finished")
                break
            if frame_count % detect_interval == 0:

                if detect:
                    frame = cv2.resize(frame, imgsz)
                    if not tracking:
                        frame = run_per_img(model,
                                            frame,
                                            device=device,
                                            project=project,
                                            name=name,
                                            exist_ok=exist_ok,
                                            half=half,
                                            **kwargs
                                            )
                    else:
                        frame = run_per_img_tracking(model,
                                                     frame,
                                                     device=device,
                                                     project=project,
                                                     name=name,
                                                     exist_ok=exist_ok,
                                                     half=half,
                                                     **kwargs
                                                     )
            frame = cv2.resize(frame, size)
            out.write(frame)
            frame_count += 1
        v_cap.release()
        out.release()
    except KeyboardInterrupt:
        v_cap.release()
        out.release()
        print("Detect Finish")


@torch.no_grad()
def run_per_img(model,  # model.pt path(s)
                img,
                device=torch.device("cpu"),
                conf_thres=0.25,  # confidence threshold
                iou_thres=0.45,  # NMS IOU threshold
                max_det=1000,  # maximum detections per image
                view_img=False,  # show results
                save_txt=False,  # save results to *.txt
                save_img=True,  # save confidences in --save-txt labels
                save_crop=False,  # save cropped prediction boxes
                classes=None,  # filter by class: --class 0, or --class 0 2 3
                agnostic_nms=False,  # class-agnostic NMS
                augment=False,  # augmented inference
                visualize=False,  # visualize features
                project=ROOT / 'runs/detect',  # save results to project/name
                name='exp',  # save results to project/name
                exist_ok=True,  # existing project/name ok, do not increment
                line_thickness=3,  # bounding box thickness (pixels)
                hide_labels=False,  # hide labels
                hide_conf=False,  # hide confidences
                half=False,  # use FP16 half-precision inference
                **kwargs
                ):
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

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

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # Process predictions
    for i, det in enumerate(pred):  # per image
        im0 = img.copy()
        imc = im0.copy() if save_crop else im0  # for save_crop
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            labels = ""
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt:
                    [x1, y1, x2, y2] = xyxy
                    if "frame_id" in kwargs:
                        frame_id = kwargs["frame_id"]
                        labels += f"{frame_id} {int(cls)} {conf:.2f} {x1.int().item()} {y1.int().item()} {x2.int().item()} {y2.int().item()} "
                    else:
                        labels += f"{int(cls)} {conf:.2f} {x1.int().item()} {y1.int().item()} {x2.int().item()} {y2.int().item()} "
                if save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'detected.jpg', BGR=True)
            labels += "\n"
            if save_txt:
                with open(save_dir / "labels" / "labels.txt", "a+") as f:
                    f.write(labels)
                f.close()
        # Stream results
        im0 = annotator.result()
        if view_img or save_img:
            if save_img:
                save_path = save_dir / "detected.png"
                frame = Image.fromarray(im0)
                frame.save(save_path)
            if view_img:
                IPython.display.clear_output(wait=True)  # clear the previous frame
                show_array(im0)

    return im0


@torch.no_grad()
def run_per_img_tracking(model,  # model.pt path(s)
                         img,
                         device=torch.device("cpu"),
                         conf_thres=0.25,  # confidence threshold
                         iou_thres=0.45,  # NMS IOU threshold
                         max_det=1000,  # maximum detections per image
                         view_img=False,  # show results
                         save_txt=False,  # save results to *.txt
                         save_img=True,  # save confidences in --save-txt labels
                         save_crop=False,  # save cropped prediction boxes
                         classes=None,  # filter by class: --class 0, or --class 0 2 3
                         agnostic_nms=False,  # class-agnostic NMS
                         augment=False,  # augmented inference
                         visualize=False,  # visualize features
                         project=ROOT / 'runs/detect',  # save results to project/name
                         name='exp',  # save results to project/name
                         exist_ok=True,  # existing project/name ok, do not increment
                         half=False,  # use FP16 half-precision inference
                         **kwargs
                         ):
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

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

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # Process predictions
    for i, det in enumerate(pred):  # per image
        im0 = img.copy()
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            bbox_xyxy = []
            confs = []

            # Write results
            for *xyxy, conf, cls in reversed(det):
                bbox_xyxy.append(xyxy)
                confs.append(conf)
            tracks.update(torch.tensor(bbox_xyxy), confs, im0)

            for track in tracks.tracks:
                if track.current:
                    track.__draw__(im0)
        # Stream results
        if view_img or save_img:
            if view_img:
                IPython.display.clear_output(wait=True)  # clear the previous frame
                show_array(im0)
            if save_img:
                save_path = save_dir / "detected.png"
                cv2.imwrite(str(save_path), im0)
    return im0
