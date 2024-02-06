#ifndef _DEPTH_ANYTHING_DETECTOR_H
#define _DEPTH_ANYTHING_DETECTOR_H

#include <memory>
#include "../../utils/backend/tensorrt/trt_infer.h"
#include "../../utils/common/model_info.h"
#include "../../utils/common/utils.h"
#include "../../utils/common/cv_cpp_utils.h"
#include "../../utils/common/cuda_utils.h"
#include "../../utils/common/memory.h"
#include "../../utils/preprocess/pre_process.h"
#include "../../utils/postprocess/post_process.h"

namespace tensorrt_infer
{
    namespace depth_anything_infer
    {
        using namespace ai::modelInfo;
        using namespace ai::utils;
        using namespace ai::cvUtil;
        using namespace ai::memory;
        using namespace ai::preprocess;
        using namespace ai::postprocess;

        class depth_anything_detector
        {
        public:
            depth_anything_detector() = default;
            ~depth_anything_detector();

            bool initParameters(const std::string& engine_file);
            void adjust_memory(int batch_size);

            cv::Mat forward(const Image& image);
            std::vector<cv::Mat> forwards(const std::vector<Image>& images);

            void preprocess_gpu(int ibatch, const Image& image,
                                std::shared_ptr<Memory<unsigned char>> preprocess_buffer, AffineMatrix& affine,
                                cudaStream_t stream_);
            void postprocess_gpu(int ibatch, cudaStream_t stream_);

            std::vector<cv::Mat> parser_depthvalue(int num_image);
        private:
            int output_height_;
            int output_width_;
            float norm_mean[3] = { 123.675, 116.28, 103.53 };
            float norm_std[3]  = { 58.395, 57.12, 57.375 };

            std::shared_ptr<ai::backend::Infer> model_;
            std::shared_ptr<ModelInfo> model_info = nullptr;

            std::vector<AffineMatrix> affine_matrixs;
            const uint8_t const_value = 114;

            std::vector<std::shared_ptr<Memory<unsigned char>>> preprocess_buffers_;
            Memory<float> input_buffer_;
            Memory<int> depth_predict_;

            std::vector<std::shared_ptr<Memory<unsigned char>>> prerprocess_buffers_;
            
            cudaStream_t cu_stream;

            Timer timer;
        };
    }
}

#endif