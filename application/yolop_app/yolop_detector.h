#ifndef _YOLOP_DETECTOR_H
#define _YOLOP_DETECTOR_H

#include <memory>
#include <algorithm>
#include "../../utils/backend/tensorrt/trt_infer.h"
#include "../../utils/common/model_info.h"
#include "../../utils/common/utils.h"
#include "../../utils/common/cv_cpp_utils.h"
#include "../../utils/common/memory.h"
#include "../../utils/preprocess/pre_process.h"
#include "../../utils/postprocess/post_process.h"
#include "../../utils/backend/backend_infer.h"

namespace tensorrt_infer
{
    namespace yolop_infer
    {
        using namespace ai::modelInfo;
        using namespace ai::utils;
        using namespace ai::cvUtil;
        using namespace ai::memory;
        using namespace ai::preprocess;
        using namespace ai::postprocess;

        class yolop_detector
        {
        public:
            yolop_detector() = default;
            ~yolop_detector();
            void initParameters(
                const std::string& engine_file,
                const DetectorType type,
                float score_thr = 0.6f,
                float nms_thr = 0.5f);
            void adjust_memory(int batch_size, const std::vector<Image>& images);

            PTMM forward(const Image& image);
            BatchPTMM forwards(const std::vector<Image>& images);

            void preprocess_gpu(int ibatch, const Image& image,
                                std::shared_ptr<Memory<unsigned char>> preprocess_buffer,
                                AffineMatrix& affine,
                                cudaStream_t stream_);
            void postprocess_gpu(int ibatch, const Image& image, cudaStream_t stream_);
            BatchPTMM parser_box(int num_image, const std::vector<Image>& images);

        private:
            std::shared_ptr<ai::backend::Infer> model_;
            std::shared_ptr<ModelInfo> model_info = nullptr;

            std::vector<AffineMatrix> affine_matrixs;
            const uint8_t const_value = 114;

            std::vector<std::shared_ptr<Memory<unsigned char>>> preprocess_buffers_;
            Memory<float> input_buffer_, output_boxarray_;
            Memory<float> bbox_predicts_, drive_area_seg_predicts_, lane_seg_predicts_;
            // Memory<uint8_t> drive_lane_mat_, drive_mask_mat_, lane_mask_mat_;
            std::vector<std::shared_ptr<Memory<unsigned char>>> drive_lane_mat_, drive_mask_mat_, lane_mask_mat_;
            cudaStream_t cu_stream;

            Timer timer;
        };
    }
}


#endif