# TensorRT_Smoke
3D检测模型SMOKE:https://github.com/lzccccc/SMOKE 的tensorrt推理代码.

# Prerequisites
MMDetection3D (v1.0.0)
```
pip install torch==1.8.0 torchvision==0.9.0
pip install mmcv-full==1.4.0
pip install mmdet==2.19.0
pip install mmsegmentation==0.20.0
git clone -b v1.0.0rc0 https://gitee.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -v -e .  # or "python setup.py develop"
```

# Steps
Export ONNX (first comment code in mmdet3d/models/dense_heads/smoke_mono3d_head.py)
```
cd mmdetection3d
mv /path_to/dl_model_infer/workpsaces/smoke_pth2onnx.py .
python smoke_pth2onnx.py  # smoke_dla34.onnx
mv smoke_dla34.onnx /path_to/dl_model_infer/workpsaces/
```
