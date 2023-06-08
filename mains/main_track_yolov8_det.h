#ifndef _MAIN_TRACK_YOLOV8_DET_H
#define _MAIN_TRACK_YOLOV8_DET_H

#include <opencv2/opencv.hpp>
#include "../utils/common/arg_parsing.h"
#include "../utils/common/cv_cpp_utils.h"
#include "../utils/tracker/ByteTracker/byte_tracker.hpp"
#include "../utils/tracker/ByteTracker/strack.hpp"
#include "../application/yolov8_app/yolov8_det_app/yolov8_detector.h"


void yolov8_det_track_trt_inference(ai::arg_parsing::Settings *s);


#endif