#ifndef _POST_PROCESS_HPP_CUDA_
#define _POST_PROCESS_HPP_CUDA_

#include <iostream>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"
#include "../common/cv_cpp_utils.h"

#define BLOCK_SIZE 32

#define CUDA_NUM_THREADS        256
#define DIV_THEN_CEIL(x, y)     (((x) + (y) - 1) / (y))

namespace ai
{
    namespace postprocess
    {
        using namespace ai::utils;
        using namespace ai::cvUtil;

        // 一般用于对yolov3/v5/v7/yolox的解析，如果你有其他任务模型的后处理需要cuda加速，也可写在这个地方
        // 默认一张图片最多的检测框是1024，可以通过传参或者直接修改默认参数改变
        void decode_detect_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                          float confidence_threshold, float *invert_affine_matrix,
                                          float *parray, int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT, cudaStream_t stream);
        // nms的cuda实现
        void nms_kernel_invoker(float *parray, float nms_threshold, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream);

        // 旋转bbox nms实现
        void rotatebbox_nms_kernel_invoker(float* parray, float nms_threshold, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream);

        // yolov8 detect后处理解析
        void decode_detect_yolov8_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                                 float confidence_threshold, float *invert_affine_matrix,
                                                 float *parray, int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT, cudaStream_t stream);

        // yolov8 obb后处理解析
        void decode_yolov8_obb_kernel_invoker(float* predict, int num_bboxes, int num_classes,
                                              float confidence_threshold, float* invert_affine_matrix,
                                              float* parray, int MAX_IMAGE_BOXES, int NUM_ROTATEBOX_ELEMENT, cudaStream_t stream);

        // yolov8 segment分支后处理
        void decode_single_mask(float left, float top, float *mask_weights, float *mask_predict,
                                int mask_width, int mask_height, unsigned char *mask_out,
                                int mask_dim, int out_width, int out_height, cudaStream_t stream);

        // yolov8 pose后处理解析
        void decode_pose_yolov8_kernel_invoker(float *predict, int num_bboxes, int pose_num, int output_cdim,
                                               float confidence_threshold, float *invert_affine_matrix,
                                               float *parray, int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT, cudaStream_t stream);

        // rtdetr后处理解析
        void decode_detect_rtdetr_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                                 float confidence_threshold, int scale_expand, float *parray, int MAX_IMAGE_BOXES,
                                                 int NUM_BOX_ELEMENT, cudaStream_t stream);
        
        void decode_kernel_yolox_invoker(const float* cls_data, const float* obj_data, const float* bbox_data,
                        const int batchsize, const int det_obj_len, const int det_bbox_len, const int det_cls_len,
                        const int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT,
                        const int input_h, const int input_w, const int stride,
                        const float confThreshold, const float nmsThreshold,
                        float *invert_affine_matrix, float* output, cudaStream_t stream);

        void decode_detect_detr_kernel_invoker(float *bbox_predict, float *label_predict,
                                            int input_h, int input_w, int num_bboxes,
                                            float confidence_threshold, float *parray, int MAX_IMAGE_BOXES,
                                            int NUM_BOX_ELEMENT, cudaStream_t stream);

        void decode_yolop_mask_kernel_invoker(float* pred_drive, float* pred_lane,
                                    uint8_t* pimage_out, uint8_t* pdrive_mask_out, uint8_t* plane_mask_out,
                                    int in_width, int in_height, float* affine_matrix,
                                    int dst_width, int dst_height, DetectorType type, cudaStream_t stream);

        float box_iou_cpu(
            float aleft, float atop, float aright, float abottom,
            float bleft, float btop, float bright, float bbottom);
        void fast_nms_cpu(float* bboxes, float threshold, int max_objects, int NUM_BOX_ELEMENT);
    }
}
#endif // _POST_PROCESS_HPP_CUDA_