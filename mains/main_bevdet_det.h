#ifndef _MAIN_BEVDET_DET_H
#define _MAIN_BEVDET_DET_H

#include <opencv2/opencv.hpp>
#include "../utils/common/arg_parsing.h"
#include "../utils/common/cv_cpp_utils.h"
#include "../utils/common/utils.h"
#include "../application/bevdet_app/bevdet_detector.h"
#include "../utils/common/cpu_jpegdecoder.h"


void bevdet_trt_inference();

#endif