#ifndef _DETR_H
#define _DETR_H

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
    namespace detr_infer
    {
        using namespace ai::modelInfo;
        using namespace ai::utils;
        using namespace ai::cvUtil;
        using namespace ai::memory;
        using namespace ai::preprocess;
        using namespace ai::postprocess;

        class detr_detector
        {
        public:
            detr_detector() = default;
            ~detr_detector();
            void initParameters(const std::string &engine_file, float score_thr = 0.5f);
            void adjust_memory(int batch_size);

            BoxArray forward(const Image &image);
            BatchBoxArray forwards(const std::vector<Image> &images);

            void preprocess_gpu(int ibatch, const Image& image,
                                std::shared_ptr<Memory<unsigned char>> preprocess_buffer, AffineMatrix& affine,
                                cudaStream_t stream_);
            void postprocess_gpu(int ibatch, cudaStream_t stream_);
            BatchBoxArray parser_box(const std::vector<Image> &images);
        private:
            int NUM_QUERY = 100;
            float mean_rgb[3] {123.675f, 116.280f, 103.530f};
            float std_rgb[3] = {58.395f, 57.120f, 57.375f};

            std::shared_ptr<ai::backend::Infer> model_;
            std::shared_ptr<ModelInfo> model_info = nullptr;

            std::vector<AffineMatrix> affine_matrixs;
            const uint8_t const_value = 114;

            std::vector<std::shared_ptr<Memory<unsigned char>>> preprocess_buffers_;
            Memory<float> input_buffer_, output_boxarray_;
            Memory<float> det_bboxes_predicts_, det_labels_predicts_;

            cudaStream_t cu_stream;

            Timer timer;
        };
    }
}

#endif