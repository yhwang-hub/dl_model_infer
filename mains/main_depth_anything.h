#ifndef _MAIN_DEPTH_ANYTHING_H
#define _MAIN_DEPTH_ANYTHING_H

#include <opencv2/opencv.hpp>
#include "../utils/common/arg_parsing.h"
#include "../utils/common/cv_cpp_utils.h"
#include "../application/depth_anything_app/depth_anything_detector.h"
#include "../utils/common/utils.h"

void depth_anything_trt_inference(ai::arg_parsing::Settings *s);

#endif