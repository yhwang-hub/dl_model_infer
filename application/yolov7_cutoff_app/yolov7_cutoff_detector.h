#ifndef _YOLOV7_CUTOFF_DETECTOR_H
#define _YOLOV7_CUTOFF_DETECTOR_H

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

namespace tensorrt_infer
{
    namespace yolov7_cutoff_infer
    {
        using namespace ai::modelInfo;
        using namespace ai::utils;
        using namespace ai::cvUtil;
        using namespace ai::memory;
        using namespace ai::preprocess;
        using namespace ai::postprocess;

        class yolov7_cutoff_detector
        {
        public:
            yolov7_cutoff_detector() = default;
            ~yolov7_cutoff_detector();

            void initParameters(
                const std::string& engine_file,
                float score_thr = 0.5f,
                float nms_thr = 0.5f
            );
            void adjust_memory(int batch_size);

            BoxArray forward(const Image& image);
            BatchBoxArray forwards(const std::vector<Image>& images);

            void preprocess_gpu(int ibatch, const Image& image,
                                std::shared_ptr<Memory<unsigned char>> preprocess_buffer,
                                AffineMatrix& affine,
                                cudaStream_t stream_);
            void postprocess_gpu(int ibatch, cudaStream_t stream_);
            BatchBoxArray parser_box(int num_image);

        private:
            static const int num_stages = 3;
            const int det_obj_len = 1;
            const int det_bbox_len = 4;
            const int det_cls_len = 80;
            const int det_len = (det_cls_len + det_bbox_len + det_obj_len) * 3;
            const int strides[num_stages] = {8, 16, 32};
            const int netAnchors[3][6] = {
                {12, 16, 19, 36, 40, 28},
                {36, 75, 76, 55, 72, 146},
                {142, 110, 192, 243, 459, 401}
            };

            int det_output_buffer_size[num_stages];

            std::shared_ptr<ai::backend::Infer> model_;
            std::shared_ptr<ModelInfo> model_info = nullptr;

            std::vector<AffineMatrix> affine_matrixs;
            const uint8_t const_value = 114;

            std::vector<std::shared_ptr<Memory<unsigned char>>> preprocess_buffers_;
            Memory<float> input_buffer_, output_boxarray_;

            Memory<float> det_output_predicts_[num_stages];

            cudaStream_t cu_stream;

            Timer timer;

            float sigmoid_x(float x)
            {
                return static_cast<float>(1.f / (1.f + exp(-x)));
            }
        };
    }
}

#endif