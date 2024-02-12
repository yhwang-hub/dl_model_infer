#ifndef _MAIN_YOLOP_DET_H
#define _MAIN_YOLOP_DET_H

#include <opencv2/opencv.hpp>
#include "../utils/common/arg_parsing.h"
#include "../utils/common/cv_cpp_utils.h"
#include "../application/yolop_app/yolop_detector.h"
#include "../utils/common/utils.h"

void yolop_det_trt_inference(ai::arg_parsing::Settings *s);

#endif