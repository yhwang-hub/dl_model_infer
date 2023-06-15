#ifndef _MAIN_POINTPILLAR_DET_H
#define _MAIN_POINTPILLAR_DET_H

#include <opencv2/opencv.hpp>
#include "../utils/common/arg_parsing.h"
#include "../utils/common/cv_cpp_utils.h"
#include "../utils/common/utils.h"
#include "../application/pointpillar_app/pointpillar.h"

void pointpillar_det_trt_inference(ai::arg_parsing::Settings *s);

#endif