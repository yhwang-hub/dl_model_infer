#include "mains/main_rt_detr.h"
#include "mains/main_yolov8_det.h"
#include "mains/main_yolov8_seg.h"
#include "mains/main_yolov8_pose.h"
#include "mains/main_track_yolov8_det.h"
#include "mains/main_yolov7_det.h"
#include "mains/main_yolox_det.h"
#include "mains/main_yolov5_det.h"
#include "mains/main_yolov7_cutoff_det.h"
#include "mains/main_yolov7_pose_det.h"
#include "mains/main_smoke_det.h"
#include "mains/main_bevdet4d_det.h"
#include "mains/main_detr_det.h"


int main(int argc, char *argv[])
{
    ai::arg_parsing::Settings s;
    if (parseArgs(argc, argv, &s) == RETURN_FAIL)
    {
        INFO("Failed to parse the args\n");
        return RETURN_FAIL;
    }
    ai::arg_parsing::printArgs(&s);

    CHECK(cudaSetDevice(s.device_id)); // 设置你用哪块gpu

    if (s.infer_task == "rt_detr_det")
    {
        rt_detr_trt_inference(&s);
    }
    else if (s.infer_task == "yolov8_det")
    {
        if (s.perf)
        {
            yolov8_trt_inference_perf(&s);
        }
        else
        {
            yolov8_trt_inference(&s);
        }
    }
    else if (s.infer_task == "yolov8_seg")
    {
        yolov8_seg_trt_inference(&s);
    }
    else if (s.infer_task == "yolov8_pose")
    {
        yolov8_pose_trt_inference(&s);
    }
    else if (s.infer_task == "yolov8_det_track")
    {
        yolov8_det_track_trt_inference(&s);
    }
    else if (s.infer_task == "yolov7_det")
    {
        yolov7_det_trt_inference(&s);
    }
    else if (s.infer_task == "yolox_det")
    {
        yolox_det_trt_inference(&s);
    }
    else if (s.infer_task == "yolov5_det")
    {
        yolov5_det_trt_inference(&s);
    }
    else if (s.infer_task == "yolov7_cutoff_det")
    {
        yolov7_cutoff_det_trt_inference(&s);
    }
    else if (s.infer_task == "yolov7_pose")
    {
        yolov7_pose_trt_inference(&s);
    }
    else if (s.infer_task == "smoke_det")
    {
        smoke_det_trt_inference(&s);
    }
    else if (s.infer_task == "bevdet4d")
    {
        bevdet4d_trt_inference();
    }
    else if (s.infer_task == "detr")
    {
        detr_trt_inference(&s);
    }

    return RETURN_SUCCESS;
}