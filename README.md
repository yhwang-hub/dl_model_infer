# dl_model_infer
![Language](https://img.shields.io/badge/language-c++-brightgreen)
![Language](https://img.shields.io/badge/CUDA-11.1-brightgreen) 
![Language](https://img.shields.io/badge/TensorRT-8.5.1.7-brightgreen)
![Language](https://img.shields.io/badge/OpenCV-4.5.5-brightgreen) 
![Language](https://img.shields.io/badge/ubuntu-16.04-brightorigin)


# Introduce
This project is modified based on the [AIInfer](https://github.com/AiQuantPro/AiInfer), Tanks for this project.

This is a c++ version of the AI reasoning library. 
Currently, it only supports the reasoning of the tensorrt model. 
The follow-up plan supports the c++ reasoning of frameworks such as Openvino, NCNN, and MNN. 
There are two versions for pre- and post-processing, c++ version and cuda version. 
It is recommended to use the cuda version.,
This repository provides accelerated deployment cases of deep learning CV popular models, 
and cuda c supports dynamic-batch image process, infer, decode, NMS.

# Update
- 2023.05.27 update yolov5、yolov7、yolov8、yolox
- 2023.05.28 update rt_detr
- 2023.06.01 update yolov8_seg、yolov8_pose
- 2023.06.09 update yolov7_cutoff
- 2023.06.14 update yolov7-pose
- 2023.06.15 Adding Producer-Consumer Inference Model for yolov8-det
- 2023.06.24 update 3D objection detection algorithm smoke
- 2023.08.21 update 3D objection detection algorithm BEVDet
- 2023.09.06 update deploy for detr in mmdetection
- 2024.01.26 update yolov8-obb

# Environment
The following environments have been tested：
- ubuntu16.04
- cuda11.1
- cudnn8.6.0
- TensorRT-8.5.1.7
- gcc5.4.0
- cmake-3.24.0
- opencv-4.5.5
- Eigen3
- yaml

You can also use docker, How to use it is as follows:
```
docker pull longxiaowyh/dl_model_infer:v1.0
nvidia-docker run -itu root:root --name dl_model_infer --gpus all -v /your_path:/target_path -v /tmp/.X11-unix/:/tmp/.X11-unix/ -e DISPLAY=unix$DISPLAY -e GDK_SCALE -e GDK_DPI_SCALE  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility --shm-size=64g longxiaowyh/dl_model_infer:v1.0 /bin/bash
```

# Model Export Tutorial
- RT-DETR model export tutorial
  - https://zhuanlan.zhihu.com/p/623794029
- Yolov8 model export tutorial
  - https://github.com/yhwang-hub/dl_model_infer/tree/master/application/yolov8_app/README.md
- Yolov5 model export tutorial
  - https://github.com/ultralytics/yolov5
- Yolov7 model export tutorial
  - https://github.com/WongKinYiu/yolov7
- yolov7_cutoff model export tutorial
  - https://github.com/yhwang-hub/dl_model_deploy/tree/master/yolov7_cutoff_TensorRT
- yolov7-pose model export tutorial
  - https://github.com/yhwang-hub/dl_model_infer/tree/master/application/yolov7_pose_app
- smoke model export tutorial
  - https://github.com/yhwang-hub/dl_model_infer/blob/dev/application/smoke_det_app/README.md
- BEVDet model export tutorial
  - https://github.com/LCH1238/BEVDet/blob/export/README.md
  - Link your own nusecnes dataset under the application/bevdet4d_app/data path
- DETR model export tutorial
  - Put the workspaces/detr_pytorch2onnx.py file under the mmdetection path.
  - Modify the config_file and checkpoint_file paths in the detr_pytorch2onnx.py file.
  - Use the detr_pytorch2onnx.py file to generate onnx file.
  - Use trtexec to generate engine files.
# Use of CPM (wrapping the inference as producer-consumer)
- cpm.hpp Producer-consumer model
  - For direct inference tasks, cpm.hpp can be turned into an automatic multi-batch producer-consumer model
```
cpm::Instance<BoxArray, Image, yolov8_detector> cpmi;
auto result_futures = cpmi.commits(yoloimages);
for (int ib = 0; ib < result_futures.size(); ++ib)
{
    auto objs = result_futures[ib].get();
    auto image = images[ib].clone();
    for (auto& obj : objs)
    {
        process....
    }
}
```

# Quick Start
Take yolov8 target detection as an example，modify CMakeLists.txt and run the command below：
```
git clone git@github.com:yhwang-hub/dl_model_infer.git
cd dl_model_infer
mkdir build && cd build
cmake .. && make
cd ../workspaces
./infer -f yolov8n.transd.trt -i res/dog.jpg -b 10 -c 10 -o cuda_res -t yolov8_det
```
You can also use a script to execute，The above instructions are written in the compile_and_run.sh script，for example：
```
rm -rf build && mkdir -p build && cd build && cmake ..  && make -j9 && cd ..
# mkdir -p build && cd build && cmake ..  && make -j48 && cd ..
cd workspaces

rm -rf cuda_res/*

# ./infer -f yolov8n.transd.trt -i res/dog.jpg -b 10 -c 10 -o cuda_res -t yolov8_det

cd ..
```
Then execute the following command to run
```
cd dl_model_infer
bash compile_and_run.sh
```

# Project directory introduction
```
AiInfer
   |--application # Implementation of model inference application, your own model inference can be implemented in this directory
     |--yolov8_det_app # Example: A yolov8 detection implemented
     |--xxxx
   |--utils # tools directory
     |--backend # here implements the reasoning class of backend
     |--common # There are some commonly used tools in it
       |--arg_parsing.h # Command line parsing class, similar to python's argparse
       |--cuda_utils.h # There are some common tool functions of cuda in it
       |--cv_cpp_utils.h # There are some cv-related utility functions in it
       |--memory.h # Tools related to cpu and gpu memory application and release
       |--model_info.h # Commonly used parameter definitions for pre- and post-processing of the model, such as mean variance, nms threshold, etc.
       |--utils.h # Commonly used tool functions in cpp, timing, mkdir, etc.
       |--cpm.h # Producer-Consumer Inference Model
     |--post_process # Post-processing implementation directory, cuda post-processing acceleration, if you have custom post-processing, you can also write it here
     |--pre_process # pre-processing implementation directory, cuda pre-processing acceleration, if you have custom pre-processing can also be written here
     |--tracker # This is the implementation of the target detection and tracking library, which has been decoupled and can be deleted directly if you don’t want to use it
   |--workspaces # Working directory, where you can put some test pictures/videos, models, and then directly use the relative path in main.cpp
   |--mains # This is the collection of main.cpp, where each app corresponds to a main file, which is easy to understand, and it is too redundant to write together
   |--main.cpp # Project entry
```

# onnx downloads
| model  | baiduyun |
| ------------- | ------------- |
| yolov5  | 链接: https://pan.baidu.com/s/1Bwwo8--JS8Vkw6METz2dIw 提取码: 47ax  |
| yolov7  | 链接: https://pan.baidu.com/s/1gb0W177xhnrseJF6CfdMEA 提取码: rvg5  |
| yolov7_cutoff  | 链接: https://pan.baidu.com/s/16bKgt_DWNmk26q-utLyCfA 提取码: q7kf  |
| yolov8  | 链接: https://pan.baidu.com/s/18Cm-tN21cus3XyirqLE_eg 提取码: j8br  |
| yolov8-seg  | 链接: https://pan.baidu.com/s/1s2Gp_Jedhi9-p_Z2utJV-Q 提取码: wr5t  |
| yolov8-pose  | 链接: https://pan.baidu.com/s/1lP8kiKu2a6h_FAZSSkhUgg 提取码: 7p6a  |
| yolox  | 链接: https://pan.baidu.com/s/1U0gzW_YMbvNMtKzo4_cluA 提取码: 4xct  |
| rt-detr  | 链接: https://pan.baidu.com/s/1Ft0-ewuCTK2BxTS1q1Vdtw 提取码: ekms  |
| yolov7-pose  | 链接: https://pan.baidu.com/s/1uI6u5oKDrnroluQufIWF7w 提取码: ed9r  |
| BEVDet | 链接：https://drive.google.com/drive/folders/1jSGT0PhKOmW3fibp6fvlJ7EY6mIBVv6i?usp=drive_link |
| DETR | 链接: https://pan.baidu.com/s/1_PQVPKy0QiFWJaB7HhyTSg 提取码: j7fs  |
| yolov8-obb| 链接: https://pan.baidu.com/s/1bMZuZPtTNjo5tl5heOdJRw 提取码: fudj |

# Reference
Thanks for the following items
- https://github.com/AiQuantPro/AiInfer
- https://github.com/shouxieai/tensorRT_Pro
- https://github.com/shouxieai/infer
- https://github.com/openppl-public/ppq
- https://github.com/nanmi/yolov7-pose
- https://github.com/LCH1238/bevdet-tensorrt-cpp
- https://github.com/LCH1238/BEVDet
- https://github.com/Melody-Zhou/tensorRT_Pro-YOLOv8
