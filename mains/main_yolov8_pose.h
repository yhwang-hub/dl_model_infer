#ifndef _MAIN_YOLOV8_POSE_H
#define _MAIN_YOLOV8_POSE_H


#include <opencv2/opencv.hpp>
#include "../utils/common/arg_parsing.h"
#include "../utils/common/cv_cpp_utils.h"
#include "../application/yolov8_app/yolov8_pose_app/yolov8_pose_detector.h"

void yolov8_pose_trt_inference(ai::arg_parsing::Settings *s);

#endif