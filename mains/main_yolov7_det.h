#ifndef _MAIN_YOLOV7_DET_H
#define _MAIN_YOLOV7_DET_H

#include <opencv2/opencv.hpp>
#include "../utils/common/arg_parsing.h"
#include "../utils/common/cv_cpp_utils.h"
#include "../application/yolov7_app/yolov7_detector.h"
#include "../utils/common/utils.h"

void yolov7_det_trt_inference(ai::arg_parsing::Settings *s);

#endif