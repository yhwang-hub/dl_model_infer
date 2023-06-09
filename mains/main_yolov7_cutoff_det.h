#ifndef _MAIN_YOLOV7_CUTOFF_DET_H
#define _MAIN_YOLOV7_CUTOFF_DET_H

#include <opencv2/opencv.hpp>
#include "../utils/common/arg_parsing.h"
#include "../utils/common/cv_cpp_utils.h"
#include "../utils/common/utils.h"
#include "../application/yolov7_cutoff_app/yolov7_cutoff_detector.h"


void yolov7_cutoff_det_trt_inference(ai::arg_parsing::Settings *s);

#endif