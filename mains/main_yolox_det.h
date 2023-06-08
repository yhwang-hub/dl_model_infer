#ifndef _MAIN_YOLOX_DET_H
#define _MAIN_YOLOX_DET_H

#include <opencv2/opencv.hpp>
#include "../utils/common/arg_parsing.h"
#include "../utils/common/cv_cpp_utils.h"
#include "../application/yolox_mmdet_app/yolox_detector.h"
#include "../utils/common/utils.h"

void yolox_det_trt_inference(ai::arg_parsing::Settings *s);

#endif