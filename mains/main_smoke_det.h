#ifndef _MAIN_SMOKE_DET_H
#define _MAIN_SMOKE_DET_H

#include <opencv2/opencv.hpp>
#include "../utils/common/arg_parsing.h"
#include "../utils/common/cv_cpp_utils.h"
#include "../utils/common/utils.h"
#include "../application/smoke_det_app/smoke_detector.h"

void smoke_det_trt_inference(ai::arg_parsing::Settings *s);

#endif