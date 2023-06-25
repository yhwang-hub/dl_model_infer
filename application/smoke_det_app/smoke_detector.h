#ifndef SMOKE_DETECTOR_H
#define SMOKE_DETECTOR_H

#include <memory>
#include <algorithm>
#include <iostream>
#include "../../utils/backend/tensorrt/trt_infer.h"
#include "../../utils/common/model_info.h"
#include "../../utils/common/utils.h"
#include "../../utils/common/cv_cpp_utils.h"
#include "../../utils/common/memory.h"
#include "../../utils/preprocess/pre_process.h"
#include "../../utils/postprocess/post_process.h"
#include "../../utils/backend/backend_infer.h"
#include "../../utils/plugins/modulated_deform_conv/trt_modulated_deform_conv.hpp"

namespace tensorrt_infer
{
    namespace smoke_det_infer
    {
        using namespace ai::modelInfo;
        using namespace ai::utils;
        using namespace ai::cvUtil;
        using namespace ai::memory;
        using namespace ai::preprocess;
        using namespace ai::postprocess;

        class smoke_detector
        {
        public:
            smoke_detector() = default;
            ~smoke_detector();

            void initParameters(
                const std::string& modelFile,
                float score_thr
            );
            void adjust_memory(int batch_size);
            CubeArray forward(const Image& image);
            BatchCubeArray forwards(const std::vector<Image>& images);
            void preprocess_gpu(int ibatch, const Image& image,
                                std::shared_ptr<Memory<unsigned char>> preprocess_buffer,
                                AffineMatrix& affine,
                                cudaStream_t stream_);
            void postprocess_gpu(int ibatch, cudaStream_t stream_);
            BatchCubeArray parser_box(int num_image);

        private:
            int topk = 100;
            float mean_rgb[3] = {123.675f, 116.280f, 103.530f};
            float std_rgb[3]  = {58.395f, 57.120f, 57.375f};
            cv::Mat intrinsic_;
            std::vector<float> base_depth;
            std::vector<BboxDim> base_dims;

            std::shared_ptr<ai::backend::Infer> model_;
            std::shared_ptr<ModelInfo> model_info = nullptr;

            std::vector<AffineMatrix> affine_matrixs;
            // const uint8_t const_value = 114;
            const uint8_t const_value = 0;

            std::vector<std::shared_ptr<Memory<unsigned char>>> preprocess_buffers_;

            Memory<float> input_buffer_, output_cubearray_;
            Memory<float> det_bbox_predicts_, det_scores_predicts_, det_indices_predicts_;

            int bbox_preds_buffer_size_;
            int topk_scores_buffer_size_;
            int topk_indices_buffer_size_;

            cudaStream_t cu_stream;

            Timer timer;
        };
    }
}

#endif