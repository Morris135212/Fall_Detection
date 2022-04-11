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
<summary>Inference</summary>

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
