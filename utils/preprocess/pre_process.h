#ifndef _PRE_PROCESS_HPP_CUDA_
#define _PRE_PROCESS_HPP_CUDA_

#include <iostream>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"
#include "../common/cv_cpp_utils.h"

namespace ai
{
    namespace preprocess
    {
        using namespace ai::cvUtil;
        // 使用cuda实现opencv的resize，双线性插值方式
        void resize_bilinear_and_normalize(
            uint8_t *src, int src_line_size, int src_width, int src_height, float *dst, int dst_width, int dst_height,
            const Norm &norm,
            cudaStream_t stream);

        // 仿射变换[平移和尺度缩放]+双线性插值实现类似resize功能，等比缩放后边缘区域填充固定const_value，常用于yolo系列的图片处理
        void warp_affine_bilinear_and_normalize_plane(
            uint8_t *src, int src_line_size, int src_width, int src_height,
            float *dst, int dst_width, int dst_height,
            float *matrix_2_3, uint8_t const_value, const Norm &norm,
            cudaStream_t stream);

        // focus层芯片不好处理，有的干脆写成cpu处理，这里将focus融入到图片处理中，一般由于yolox，yolov5-version<6.0
        void warp_affine_bilinear_and_normalize_focus(
            uint8_t *src, int src_line_size, int src_width, int src_height,
            float *dst, int dst_width, int dst_height,
            float *matrix_2_3, uint8_t const_value, const Norm &norm,
            cudaStream_t stream);

        // 有些芯片的输出图片格式nv12格式的，但我们的tensorrt模型一般要求输出rgb/bgr的，所以可以通过该函数来转换，rgb稍微修改即可
        void convert_nv12_to_bgr_invoke(
            const uint8_t *y, const uint8_t *uv, int width, int height,
            int linesize, uint8_t *dst,
            cudaStream_t stream);

        // 下面是多添加的一些功能，你也可以按照你自己的想法修改或添加新功能[cuda实现]
        // 透视变换
        void warp_perspective(
            uint8_t *src, int src_line_size, int src_width, int src_height, float *dst, int dst_width, int dst_height,
            float *matrix_3_3, uint8_t const_value, const Norm &norm, cudaStream_t stream);

        // 特征图的归一化
        void norm_feature(
            float *feature_array, int num_feature, int feature_length,
            cudaStream_t stream);
    }
}
#endif // _PRE_PROCESS_HPP_CUDA_