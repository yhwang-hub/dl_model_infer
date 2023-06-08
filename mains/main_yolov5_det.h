#ifndef _MAIN_YOLOV5_DET_H
#define _MAIN_YOLOV5_DET_H

#include <opencv2/opencv.hpp>
#include "../utils/common/arg_parsing.h"
#include "../utils/common/cv_cpp_utils.h"
#include "../application/yolov5_app/yolov5_detector.h"
#include "../utils/common/utils.h"

void yolov5_det_trt_inference(ai::arg_parsing::Settings *s);

#endif