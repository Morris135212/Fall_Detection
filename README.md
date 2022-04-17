# Fall Detection
<p> In real world, people, especially elder people, are urgent to search for device that can inspect their personal health, given the fact that those people are extremely vulnerable.
Nowadays, people are easily to obtain their physical characteristics by smart device like iWatch.

However, those devices are generally unable to detect some motion characteristics like Falling. Falling is extremely dangerous when someone live alone. So this brought me the inspiration to construct this framework to detect Fall action in home.
</p>

## <div align="center">Pipeline</div>

## <div align="center">Quick Start</div>

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/Morris135212/Fall_Detection/blob/main/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/Morris135212/Fall_Detection.git  # clone
cd fall_detection
pip install -r requirements.txt  # install
```
</details>

<details open>
<summary>Training</summary>

**yolo training**

Same process as referred to [yolov5](https://github.com/Morris135212/yolov5)

```
python train.py --data custom.yaml --cfg yolov5n.yaml --weights '' --batch-size 128
                                         yolov5s                                 64
                                         yolov5m                                 40
                                         yolov5l                                 24
                                         yolov5x                          
```

**3dcnn training**

```python
from dataset.cnn3d import CustomDataset
# Load train & validation data
train_path = "./data/dataset/Fall_Detection/Train"
val_path = "./data/dataset/Fall_Detection/Test"
classes = ["Walk", "Fall"]
trainset = CustomDataset(train_path, classes)
valset = CustomDataset(val_path, classes)

# Get model
from models.cnn3d import get_model, resnet10
model = get_model(sample_size=112, sample_duration=10, num_classes=2)
# resnet = resnet10(sample_size=112, sample_duration=10, num_classes=2)

# Training
from train.cnn3d import Trainer
config = {
    "model": model,
    "train_data":trainset,
    "val_data":valset,
    "batch_size":2,
    "epochs":20,
    "step_size":200,
    "lr":1e-3,
    "interval":40
}
trainer = Trainer(**config)
trainer()
```

</details>

<details open>
<summary>Fall Detection Inference</summary>

```python
from models.common import DetectMultiBackend
import torch
from detect.fall_detection.detect import live_inference
from models.cnn3d import get_model

data = "./data/custom.yaml"
weights = "./runs/train/yolov5n/weights/best.pt"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Load Yolo model
yolo_model = DetectMultiBackend(weights, device=device, data=data)
# Load 3dCNN model
cnn3d = get_model(sample_size=112, sample_duration=10, num_classes=2)
cnn3d.load_state_dict(torch.load("./runs/cnn3d/basemodel/best.pt", map_location=device))

source = 0 # Live inference
# source = "./runs/detect/exp/Fall_Demo.mp4" # Video

live_inference(yolo_model=yolo_model,
               cnn3d=cnn3d, 
               source=source,
               class_names=["WALK", "FALL"],
               out_name="FALL_DETECTION", #Out put file name
               fall=True, # Whether to perform Fall detection
               conf_thres=0.4)
```
</details>
