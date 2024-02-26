#ifndef _YOLOV9_DETECTOR_H
#define _YOLOV9_DETECTOR_H

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
    namespace yolov9_infer
    {
        using namespace ai::modelInfo;
        using namespace ai::utils;
        using namespace ai::cvUtil;
        using namespace ai::memory;
        using namespace ai::preprocess;
        using namespace ai::postprocess;

        class yolov9_detector
        {
        public:
            yolov9_detector() = default;
            ~yolov9_detector();

            void initParameters(const std::string& engine_file, float score_thr = 0.5f, float nms_thr = 0.5f);
            void adjust_memory(int batch_size);

            BoxArray forward(const Image& image);
            BatchBoxArray forwards(const std::vector<Image>& images);

            void preprocess_gpu(int ibatch, const Image& image,
                                std::shared_ptr<Memory<unsigned char>> preprocess_buffer, AffineMatrix& affine,
                                cudaStream_t stream_);
            void postprocess_gpu(int ibatch, cudaStream_t stream_);
            BatchBoxArray parser_box(int num_image);

        private:
            std::shared_ptr<ai::backend::Infer> model_;
            std::shared_ptr<ModelInfo> model_info = nullptr;

            std::vector<AffineMatrix> affine_matrixs;
            const uint8_t const_value = 114;

            std::vector<std::shared_ptr<Memory<unsigned char>>> preprocess_buffers_;
            Memory<float> input_buffer_, bbox_predict1_, bbox_predict2_, output_boxarray_;

            cudaStream_t cu_stream;

            Timer timer;
        };
    }
}

#endif