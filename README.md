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
- 2023.06.08 update pointpillar

# Environment
The following environments have been tested：
- ubuntu16.04
- cuda11.1
- cudnn8.6.0
- TensorRT-8.5.1.7
- gcc5.4.0
- cmake-3.24.0
- opencv-4.5.5

# Model Export Tutorial
- RT-DETR model export tutorial
  - https://zhuanlan.zhihu.com/p/623794029
- Yolov8 model export tutorial
  - https://github.com/yhwang-hub/dl_model_infer/tree/master/application/yolov8_app/README.md
- Yolov5 model export tutorial
  - https://github.com/ultralytics/yolov5
- Yolov7 model export tutorial
  - https://github.com/WongKinYiu/yolov7
- pointpillar model export tutorial
  - https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars
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
       |--arg_parsing.hpp # Command line parsing class, similar to python's argparse
       |--cuda_utils.hpp # There are some common tool functions of cuda in it
       |--cv_cpp_utils.hpp # There are some cv-related utility functions in it
       |--memory.hpp # Tools related to cpu and gpu memory application and release
       |--model_info.hpp # Commonly used parameter definitions for pre- and post-processing of the model, such as mean variance, nms threshold, etc.
       |--utils.hpp # Commonly used tool functions in cpp, timing, mkdir, etc.
     |--post_process # Post-processing implementation directory, cuda post-processing acceleration, if you have custom post-processing, you can also write it here
     |--pre_process # pre-processing implementation directory, cuda pre-processing acceleration, if you have custom pre-processing can also be written here
     |--tracker # This is the implementation of the target detection and tracking library, which has been decoupled and can be deleted directly if you don’t want to use it
   |--workspaces # Working directory, where you can put some test pictures/videos, models, and then directly use the relative path in main.cpp
   |--mains # This is the collection of main.cpp, where each app corresponds to a main file, which is easy to understand, and it is too redundant to write together
   |--main.cpp # Project entry
```

# Reference
Thanks for the following items
- https://github.com/AiQuantPro/AiInfer
- https://github.com/shouxieai/tensorRT_Pro
- https://github.com/shouxieai/infer
- https://github.com/openppl-public/ppq
