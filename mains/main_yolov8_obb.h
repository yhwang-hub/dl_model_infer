#ifndef _MAIN_YOLOV8_OBB_H
#define _MAIN_YOLOV8_OBB_H

#include <opencv2/opencv.hpp>
#include "../utils/common/arg_parsing.h"
#include "../utils/common/cv_cpp_utils.h"
#include "../application/yolov8_app/yolov8_obb_app/yolov8_obb_detector.h"
#include "../utils/common/cpm.h"

using namespace ai::utils;
using namespace ai::cvUtil;
using namespace tensorrt_infer::yolov8_infer;

void yolov8_obb_trt_inference(ai::arg_parsing::Settings *s);

#endif