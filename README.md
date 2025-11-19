# modelfactory
ai model factory for test

## 使用ImageNet 2012数据集训练MobileNetV3
```
python train.py --data-dir /data/ImageNet/data/ImageNet2012 --export-onnx-path ./output/models --epochs 60
python train.py --data-dir /data/ImageNet/data/ImageNet2012 --export-onnx-path ./output/models --epochs 60 --pth-file temp/mobilenetv3_latest.pth
```