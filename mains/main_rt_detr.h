#ifndef _MAIN_RT_DETR_H
#define _MAIN_RT_DETR_H


#include <opencv2/opencv.hpp>
#include "../utils/common/arg_parsing.h"
#include "../utils/common/cv_cpp_utils.h"
#include "../application/rt_detr_app/rt_detr_detector.h"
#include "../utils/common/utils.h"

void rt_detr_trt_inference(ai::arg_parsing::Settings *s);

#endif