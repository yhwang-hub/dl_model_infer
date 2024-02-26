#ifndef _MAIN_YOLOV9_DET_H
#define _MAIN_YOLOV9_DET_H

#include <opencv2/opencv.hpp>
#include "../utils/common/arg_parsing.h"
#include "../utils/common/cv_cpp_utils.h"
#include "../application/yolov9_app/yolov9_detector.h"
#include "../utils/common/utils.h"

void yolov9_det_trt_inference(ai::arg_parsing::Settings *s);

#endif