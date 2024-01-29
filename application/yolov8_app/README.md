# yolov8不同任务中onnx的导出方法
## yolov8-detect onnx的导出
1. 在 ultralytics/engine/exporter.py 文件中改动一处
    - 323 行：输出节点名修改为 output
    - 326 行：输入只让 batch 维度动态，宽高不动态
    - 331 行：输出只让 batch 维度动态，宽高不动态
```
# ========== exporter.py ==========

# ultralytics/engine/exporter.py第323行
# output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output0']
# dynamic = self.args.dynamic
# if dynamic:
#     dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
#     if isinstance(self.model, SegmentationModel):
#         dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 116, 8400)
#         dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
#     elif isinstance(self.model, DetectionModel):
#         dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 84, 8400)
# 修改为：

output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output']
dynamic = self.args.dynamic
if dynamic:
    dynamic = {'images': {0: 'batch'}}  # shape(1,3,640,640)
    if isinstance(self.model, SegmentationModel):
        dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 116, 8400)
        dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
    elif isinstance(self.model, DetectionModel):
        dynamic['output'] = {0: 'batch'}  # shape(1, 84, 8400)
```
2. 在 ultralytics/nn/modules/head.py 文件中改动一处
    - 72 行：添加 transpose 节点交换输出的第 2 和第 3 维度
```
# ========== head.py ==========

# ultralytics/nn/modules/head.py第72行，forward函数
# return y if self.export else (y, x)
# 修改为：

return y.permute(0, 2, 1) if self.export else (y, x)
```
- 新建导出文件 export.py，内容如下：
```
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

success = model.export(format="onnx", dynamic=True, simplify=True)
```
在终端执行如下指令即可完成 onnx 导出：
```
python export.py
```

## yolov8-obb onnx导出
- https://github.com/ultralytics/ultralytics
- 修改dynamic参数中对应的代码
1. 在 ultralytics/engine/exporter.py 文件中改动一处
    - 353 行：输出节点名修改为 output
    - 356 行：输入只让 batch 维度动态，宽高不动态
    - 361 行：输出只让 batch 维度动态，宽高不动态
```
# ========== exporter.py ==========

# ultralytics/engine/exporter.py第353行
# output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output0']
# dynamic = self.args.dynamic
# if dynamic:
#     dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
#     if isinstance(self.model, SegmentationModel):
#         dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 116, 8400)
#         dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
#     elif isinstance(self.model, DetectionModel):
#         dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 84, 8400)
# 修改为：

output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output']
dynamic = self.args.dynamic
if dynamic:
    dynamic = {'images': {0: 'batch'}}  # shape(1,3,640,640)
    if isinstance(self.model, SegmentationModel):
        dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 116, 8400)
        dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
    elif isinstance(self.model, DetectionModel):
        dynamic['output'] = {0: 'batch'}  # shape(1, 84, 8400)
```
2. 在 ultralytics/nn/modules/head.py 文件中改动一处
    - 141 行：添加 transpose 节点交换输出的第 2 和第 3 维度
```
# ========== head.py ==========

# ultralytics/nn/modules/head.py第141行，forward函数
# return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))
# 修改为：

return torch.cat([x, angle], 1).permute(0, 2, 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))
```
以上就是为了适配 tensorRT_Pro 而做出的代码修改，修改好以后，将预训练权重 yolov8s-obb.pt 放在 ultralytics-main 主目录下，新建导出文件 export.py，内容如下：
```
from ultralytics import YOLO
model = YOLO("yolov8s-obb.pt")
success = model.export(format="onnx", dynamic=True, simplify=True)
```
在终端执行如下指令即可完成 onnx 导出：
```
python export.py
```

## yolov8的onnx生成engine文件
- fp16量化生成的命令如下，这个精度损失不大，可以直接使用trtexec完成
```
trtexec --onnx=xxx_det_seg_pose_trans.onnx \
        --minShapes=images:1x3x640x640 \
        --maxShapes=images:16x3x640x640 \
        --optShapes=images:4x3x640x640 \
        --saveEngine=xxx_det_seg_pose_dynamic_fp16.engine \
        --avgRuns=100 \
        --fp16
```
- int8量化，这个直接用trtexec一般而言精度都有损失有的甚至无法工作，建议使用商汤ppq量化工具
  - 商汤的ppq的int8量化工具,支持tensorrt|openvino|mnn|ncnn|...
     - https://github.com/openppl-public/ppq
  - ppq不会使用的看yolov6的量化教程:
     - https://github.com/meituan/YOLOv6/tree/main/tools/quantization/ppq

# 生成Engine后直接编译运行即可
- yolov8检测程序入口：mains/main_yolov8_det.cpp
- yolov8分割程序入口：mains/main_yolov8_seg.cpp
- yolov8姿态估计程序入口：mains/main_yolov8_pose.cpp


