# YOLOv9-Face

The **YOLOv9-Face** repository provides pre-trained models designed specifically for face detection. The models have
been pre-trained by Lindevs from scratch.

## Release Notes

* **[2024-11-01]** YOLOv9t-Face, YOLOv9s-Face, YOLOv9m-Face, YOLOv9c-Face and YOLOv9e-Face models has been added.

## Pre-trained Models

The models have been trained on [WIDERFace](http://shuoyang1213.me/WIDERFACE/) dataset using NVIDIA RTX 4090.
[YOLOv9 models](https://github.com/ultralytics/ultralytics#models) were used as initial weights for training.

| Name         | Image Size<br>(pixels) | mAP<sup>val<br>50-95 | Params   | GFLOPs |
|--------------|------------------------|----------------------|----------|--------|
| YOLOv9t-Face | 640                    | 37.0                 | 1730019  | 6.4    |
| YOLOv9s-Face | 640                    | 40.6                 | 6194035  | 22.1   |
| YOLOv9m-Face | 640                    | 42.5                 | 16575715 | 60.0   |
| YOLOv9c-Face | 640                    | 42.4                 | 21146195 | 82.7   |
| YOLOv9e-Face | 640                    | 43.3                 | 53203347 | 169.5  |

* Download links:

| Name         | Model Size (MB) | Link                                                                                                                                                                                                    | SHA-256                                                                                                                              |
|--------------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| YOLOv9t-Face | 4.0<br>7.0      | [PyTorch](https://github.com/lindevs/yolov9-face/releases/latest/download/yolov9t-face-lindevs.pt)<br>[ONNX](https://github.com/lindevs/yolov9-face/releases/latest/download/yolov9t-face-lindevs.onnx) | 3914713a5353d060eadc2cd8888676cc6ea9ac59921ed8bcff42755ee75a298c<br>7766f85cecd7045a1b64cf3a89d94819c62cc5ff24b782b86bb0dec4f9e31964 |
| YOLOv9s-Face | 12.7<br>24.0    | [PyTorch](https://github.com/lindevs/yolov9-face/releases/latest/download/yolov9s-face-lindevs.pt)<br>[ONNX](https://github.com/lindevs/yolov9-face/releases/latest/download/yolov9s-face-lindevs.onnx) | f78b366a504b33f69e8b0ad0fc3c28e64153b167d71f4c1a29b903840fe67df4<br>9be9d734271868226274ea7e54f15e8c5bc2a4cf1b909a2eb6b6602987627e61 |
| YOLOv9m-Face | 32.4<br>63.5    | [PyTorch](https://github.com/lindevs/yolov9-face/releases/latest/download/yolov9m-face-lindevs.pt)<br>[ONNX](https://github.com/lindevs/yolov9-face/releases/latest/download/yolov9m-face-lindevs.onnx) | bda08ea1388ae1d37747acb4dec4b08884b0078af2fc137fe4e93d498c474d3f<br>a9a5775f869bc813402a37a690c50d9344520eda3177e4f66860370a68e5f23b |
| YOLOv9c-Face | 41.3<br>81.0    | [PyTorch](https://github.com/lindevs/yolov9-face/releases/latest/download/yolov9c-face-lindevs.pt)<br>[ONNX](https://github.com/lindevs/yolov9-face/releases/latest/download/yolov9c-face-lindevs.onnx) | b97c52c484ec873d0714615f566e31dc0bebc96eeafc43dca8204a9355802ba0<br>4e1cf66b2eade9240b5073d9563e6b737fe38123c4e53e342bec36274b530fae |
| YOLOv9e-Face | 103.9<br>203.4  | [PyTorch](https://github.com/lindevs/yolov9-face/releases/latest/download/yolov9e-face-lindevs.pt)<br>[ONNX](https://github.com/lindevs/yolov9-face/releases/latest/download/yolov9e-face-lindevs.onnx) | 8f3410c7001dd73961a9c649d7dbd62162d0ac5851b54fd99deea4b9681abeed<br>4942ffc2f41358355913b85ac2f9aa033ec3d25a56546f6141500097d4a7b4f4 |

* Training results:

| Name         | Training Time | Epochs | Batch Size | Link                                                  |
|--------------|---------------|--------|------------|-------------------------------------------------------|
| YOLOv9t-Face | 4.40 hours    | 300    | 16         | [results.txt](results/train/yolov9t-face/results.txt) |
| YOLOv9s-Face | 5.55 hours    | 300    | 16         | [results.txt](results/train/yolov9s-face/results.txt) |
| YOLOv9m-Face | 5.61 hours    | 200    | 16         | [results.txt](results/train/yolov9m-face/results.txt) |
| YOLOv9c-Face | 4.34 hours    | 130    | 16         | [results.txt](results/train/yolov9c-face/results.txt) |
| YOLOv9e-Face | 5.04 hours    | 70     | 9          | [results.txt](results/train/yolov9e-face/results.txt) |

* Evaluation results on WIDERFace dataset:

| Name         | Easy  | Medium | Hard  |
|--------------|-------|--------|-------|
| YOLOv9t-Face | 94.33 | 92.27  | 78.91 |
| YOLOv9s-Face | 95.54 | 94.08  | 83.06 |
| YOLOv9m-Face | 96.08 | 94.74  | 84.91 |
| YOLOv9c-Face | 96.28 | 95.09  | 85.47 |
| YOLOv9e-Face | 96.39 | 95.34  | 85.87 |

## Instructions

## Installation

```shell
pip install -r requirements.txt
```

## Prediction

```shell
python predict.py --weights weights/yolov9t-face-lindevs.pt --source data/images/bus.jpg
```

* OpenCV DNN

```shell
python examples/opencv-dnn-python/main.py --weights weights/yolov9t-face-lindevs.onnx --source data/images/bus.jpg
```

## Export

* Install package:

```shell
pip install onnx
```

* Export to ONNX format:

```shell
python export.py --weights weights/yolov9t-face-lindevs.pt
```

* Or export to ONNX format using dynamic axis:

```shell
python export.py --weights weights/yolov9t-face-lindevs.pt --dynamic
```

## Dataset Preparation

* Download WIDERFace dataset and annotations:

```shell
python download.py
```

* Convert annotations to YOLO format:

```shell
python annotations.py
```

* Copy `widerface.yaml.example` file to `widerface.yaml`:

```shell
python data_file.py
```

## Training

* Prepare dataset.
* Start training:

```shell
python train.py --weights yolov9t.pt --epochs 300 2>&1 | tee -a results.txt
python train.py --weights yolov9s.pt --epochs 300 2>&1 | tee -a results.txt
python train.py --weights yolov9m.pt --epochs 200 2>&1 | tee -a results.txt
python train.py --weights yolov9c.pt --epochs 130 2>&1 | tee -a results.txt
python train.py --weights yolov9e.pt --epochs 70 --batch 9 2>&1 | tee -a results.txt
```

* Or resume training:

```shell
python train.py --weights runs/detect/train/weights/last.pt --resume 2>&1 | tee -a results.txt
```

## Validation

* Prepare dataset.
* Start validation:

```shell
python validate.py --weights weights/yolov9t-face-lindevs.pt
```

## WIDERFace Evaluation

* Prepare dataset.
* Start prediction on validation set:

```shell
python widerface/predict.py --weights weights/yolov9t-face-lindevs.pt
```

* Install package:

```shell
pip install Cython
```

* Build extension:

```shell
cd widerface && python setup.py build_ext --inplace && cd ..
```

* Start evaluation:

```shell
python widerface/evaluate.py
```
