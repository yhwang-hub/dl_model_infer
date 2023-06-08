cmake_minimum_required(VERSION 3.10.0)
project(infer)
add_definitions(-std=c++11 -w)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# 1. 设置工作目录,里面会放测试图片和模型，生成的可执行文件也会在该目录下
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspaces)
set(CMAKE_INSTALL_PREFIX ${EXECUTABLE_OUTPUT_PATH}/install/) # make install时的存储路径

# 2. 设置显卡算力，如果你是不同显卡，请设置为显卡对应的号码参考下面的链接，我这里是RTX 3090,对应的是sm_86：
# https://developer.nvidia.com/zh-cn/cuda-gpus#compute
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_86,code=sm_86;-G;-g;-O0;-w")

# 3. 寻找cuda和opencv库
find_package(CUDA REQUIRED) # 这个默认你本机已经安装
find_package(OpenCV REQUIRED) # 如果你没安装，sudo apt-get install libopencv-dev
# find_package(OpenCV 4 REQUIRED)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# 4. 设置tensorrt的主目录,支持tensorrt7.xx和tensorrt8.xx
# set(TensorRT_ROOT "/home/uisee/disk/TensorRT-8.6.0.12") # 设置tensorrt8.xx根目录，改为你自己的即可
set(TensorRT_ROOT "/home/uisee/disk/TensorRT-8.5.1.7") # 设置tensorrt8.xx根目录，改为你自己的即可

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -O0")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# 5. include所有要用到的hpp文件路径
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}

    # tensorrt
    ${TensorRT_ROOT}/include
    ${TensorRT_ROOT}/samples/common # 导入这个主要是为了适应于trt多版本[v7.xx,v8.xx]的logger导入

    # 项目里要用到的
    ${PROJECT_SOURCE_DIR}/utils
    ${PROJECT_SOURCE_DIR}/application
)

# 6. link要用到的so库路径
# 补充：具体的cuda_lib库命名可以看 https://cmake.org/cmake/help/latest/module/FindCUDA.html
link_directories(
    # ${CUDA_LIBRARIES}
    # ${CUDA_cublas_LIBRARY}
    # ${CUDA_cudart_static_LIBRARY}
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64

    # tensorrt
    ${TensorRT_ROOT}/lib
)


# 7. 将utils里写好的cu文件和cpp文件编译成so库，方便后面调用
file(GLOB_RECURSE cpp_cuda_srcs
    ${PROJECT_SOURCE_DIR}/main.cpp
    ${PROJECT_SOURCE_DIR}/application/rt_detr_app/*.cpp
    ${PROJECT_SOURCE_DIR}/application/yolov7_app/*.cpp
    ${PROJECT_SOURCE_DIR}/application/yolov5_app/*.cpp
    ${PROJECT_SOURCE_DIR}/application/yolox_mmdet_app/*.cpp
    ${PROJECT_SOURCE_DIR}/application/pointpillar_app/*.cpp
    ${PROJECT_SOURCE_DIR}/application/pointpillar_app/*.cu
    ${PROJECT_SOURCE_DIR}/application/yolov8_app/yolov8_det_app/*.cpp
    ${PROJECT_SOURCE_DIR}/application/yolov8_app/yolov8_seg_app/*.cpp
    ${PROJECT_SOURCE_DIR}/application/yolov8_app/yolov8_pose_app/*.cpp
    ${PROJECT_SOURCE_DIR}/utils/common/*.cpp
    ${PROJECT_SOURCE_DIR}/utils/tracker/ByteTracker/*.cpp
    ${PROJECT_SOURCE_DIR}/utils/backend/tensorrt/*.cpp
    ${PROJECT_SOURCE_DIR}/utils/preprocess/*.cu
    ${PROJECT_SOURCE_DIR}/utils/preprocess/pointpillar/*.cu
    ${PROJECT_SOURCE_DIR}/utils/preprocess/pointpillar/*.cpp
    ${PROJECT_SOURCE_DIR}/utils/postprocess/*.cu
    ${PROJECT_SOURCE_DIR}/utils/postprocess/pointpillar/*.cu
    ${PROJECT_SOURCE_DIR}/utils/postprocess/pointpillar/*.cpp
    ${TensorRT_ROOT}/samples/common/logger.cpp # 引用对应版本的logger.cpp，用来适应多版本
    ${TensorRT_ROOT}/samples/common/sampleOptions.cpp 
    ${TensorRT_ROOT}/samples/common/sampleUtils.cpp
)
cuda_add_library(utils_cu_cpp SHARED ${cpp_cuda_srcs})

add_executable(infer
     ${PROJECT_SOURCE_DIR}/mains/main_rt_detr.cpp
     ${PROJECT_SOURCE_DIR}/mains/main_yolov7_det.cpp
     ${PROJECT_SOURCE_DIR}/mains/main_yolov5_det.cpp
     ${PROJECT_SOURCE_DIR}/mains/main_yolox_det.cpp
     ${PROJECT_SOURCE_DIR}/mains/main_yolov8_det.cpp
     ${PROJECT_SOURCE_DIR}/mains/main_yolov8_seg.cpp
     ${PROJECT_SOURCE_DIR}/mains/main_yolov8_pose.cpp
     ${PROJECT_SOURCE_DIR}/mains/main_track_yolov8_det.cpp
     ${PROJECT_SOURCE_DIR}/mains/main_pointpillar_det.cpp
)
# 8. 链接要所有要用到的so库
target_link_libraries(infer
    utils_cu_cpp # 调用上面编译好的so库
    cuda
    cudart
    cudnn
    pthread
    ${OpenCV_LIBS}
    nvinfer 
    nvinfer_plugin
    nvonnxparser    
)

# make install 时需要用到
install(TARGETS infer  utils_cu_cpp
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)

install(DIRECTORY
        ${PROJECT_SOURCE_DIR}/mains
        ${PROJECT_SOURCE_DIR}/utils/backend
        ${PROJECT_SOURCE_DIR}/utils/backend/tensorrt
        ${PROJECT_SOURCE_DIR}/utils/common
        ${PROJECT_SOURCE_DIR}/utils/postprocess
        ${PROJECT_SOURCE_DIR}/utils/postprocess/pointpillar
        ${PROJECT_SOURCE_DIR}/utils/preprocess
        ${PROJECT_SOURCE_DIR}/utils/preprocess/pointpillar
        ${PROJECT_SOURCE_DIR}/application/
        ${PROJECT_SOURCE_DIR}/application/rt_detr_app
        ${PROJECT_SOURCE_DIR}/application/yolov7_app
        ${PROJECT_SOURCE_DIR}/application/yolov5_app
        ${PROJECT_SOURCE_DIR}/application/yolox_mmdet_app
        ${PROJECT_SOURCE_DIR}/application/pointpillar_app
        ${PROJECT_SOURCE_DIR}/application/yolov8_app/yolov8_det_app
        ${PROJECT_SOURCE_DIR}/application/yolov8_app/yolov8_seg_app
        ${PROJECT_SOURCE_DIR}/application/yolov8_app/yolov8_pose_app
        DESTINATION include/
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h" PATTERN "*.cu")

# 通过make auto -j 来编译和运行程序
# add_custom_target(
#     auto
#     DEPENDS infer
#     WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspaces
#     COMMAND ./infer
# )
