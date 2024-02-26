## Installation

1. Setup [yolov9](https://github.com/WongKinYiu/yolov9) and download [yolov9-c.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt) model.
2. Convert the model to onnx format:

- Copy `application/yolov9_app/export_.py` in this repo to yolov9 installation folder
- Then export the model
```
python export_.py --weights yolov9-c.pt --simplify --include onnx
```
3. Modify the output dimensions of the model
```
python yolov9_onnx_trans.py --onnx_path ./yolov9-c.onnx
```

4. Build a TensorRT engine: 

```
trtexec --onnx=./yolov9-c_transd.onnx --saveEngine=./yolov9-c_transd_fp16.trt --fp16
```