#ifndef _MAIN_DETR_H
#define _MAIN_DETR_H


#include <opencv2/opencv.hpp>
#include "../utils/common/arg_parsing.h"
#include "../utils/common/cv_cpp_utils.h"
#include "../application/detr_app/detr_detector.h"
#include "../utils/common/utils.h"

void detr_trt_inference(ai::arg_parsing::Settings *s);

#endif