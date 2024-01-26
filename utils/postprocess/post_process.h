#ifndef _POST_PROCESS_HPP_CUDA_
#define _POST_PROCESS_HPP_CUDA_

#include <iostream>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"
#include "../common/cv_cpp_utils.h"
#include "../kernels/iou3d_nms/iou3d_nms.h"

#define BLOCK_SIZE 32

#define CUDA_NUM_THREADS        256
#define DIV_THEN_CEIL(x, y)     (((x) + (y) - 1) / (y))

namespace ai
{
    namespace postprocess
    {
        using namespace ai::utils;

        class BEVDetPostprocessGPU
        {
        public:
            BEVDetPostprocessGPU(){};
            BEVDetPostprocessGPU(const int _class_num, 
                            const float _score_thresh,
                            const float _nms_thresh, 
                            const int _nms_pre_maxnum,
                            const int _nms_post_maxnum, 
                            const int _down_sample, 
                            const int _output_h, 
                            const int _output_w, 
                            const float _x_step, 
                            const float _y_step,
                            const float _x_start, 
                            const float _y_start,
                            const std::vector<int>& _class_num_pre_task,
                            const std::vector<float>& _nms_rescale_factor);

            void DoPostprocess(std::vector<void*> bev_buffer, std::vector<bevBox>& out_detections);
            ~BEVDetPostprocessGPU();

        private:
            int class_num;
            float score_thresh;
            float nms_thresh;
            int nms_pre_maxnum;
            int nms_post_maxnum;
            int down_sample;
            int output_h;
            int output_w;
            float x_step;
            float y_step;
            float x_start;
            float y_start;
            int map_size;
            int task_num;

            std::vector<int> class_num_pre_task;
            std::vector<float> nms_rescale_factor;

            std::unique_ptr<Iou3dNmsCuda> iou3d_nms;

            float* boxes_dev = nullptr;
            float* score_dev = nullptr;
            int* cls_dev = nullptr;
            int* valid_box_num = nullptr;
            int* sorted_indices_dev = nullptr;
            long* keep_data_host = nullptr;
            int* sorted_indices_host = nullptr;
            float* boxes_host = nullptr;
            float* score_host = nullptr;
            int* cls_host = nullptr;

            float* nms_rescale_factor_dev = nullptr;

        };

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

        // void decode_detect_detr_kernel_invoker(float *bbox_predict, float *label_predict,
        //                                     int input_h, int input_w, int num_bboxes,
        //                                     float confidence_threshold, float *invert_affine_matrix, float *parray, int MAX_IMAGE_BOXES,
        //                                     int NUM_BOX_ELEMENT, cudaStream_t stream);
        void decode_detect_detr_kernel_invoker(float *bbox_predict, float *label_predict,
                                            int input_h, int input_w, int num_bboxes,
                                            float confidence_threshold, float *parray, int MAX_IMAGE_BOXES,
                                            int NUM_BOX_ELEMENT, cudaStream_t stream);

        float box_iou_cpu(
            float aleft, float atop, float aright, float abottom,
            float bleft, float btop, float bright, float bbottom);
        void fast_nms_cpu(float* bboxes, float threshold, int max_objects, int NUM_BOX_ELEMENT);
    }
}
#endif // _POST_PROCESS_HPP_CUDA_