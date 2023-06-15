#ifndef _YOLOX_DETECTOR_H
#define _YOLOX_DETECTOR_H

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
    namespace yolox_infer
    {
        using namespace ai::modelInfo;
        using namespace ai::utils;
        using namespace ai::cvUtil;
        using namespace ai::memory;
        using namespace ai::preprocess;
        using namespace ai::postprocess;

        class yolox_detector
        {
        public:
            yolox_detector() = default;
            ~yolox_detector();

            void initParameters(const std::string& engine_file, float score_thr = 0.5f, float nms_thr = 0.5f); // 初始化参数
            void adjust_memory(int batch_size); // 由于batch_size是动态的，所以需要对gpu/cpu内存进行动态的申请

            // forward
            BoxArray forward(const Image& image);
            BatchBoxArray forwards(const std::vector<Image>& images);

            // 模型的预处理和后处理
            void preprocess_gpu(int ibatch, const Image& image,
                                std::shared_ptr<Memory<unsigned char>> preprocess_buffer, AffineMatrix& affine,
                                cudaStream_t stream_);
            void postprocess_gpu(int ibatch, cudaStream_t stream_);
            BatchBoxArray parser_box(int num_image);

        private:
            static const int num_stage = 3;
            const int det_obj_len = 1;
            const int det_bbox_len = 4;
            const int det_cls_len = 80;
            const int strides[num_stage] = {8, 16, 32};

            int det_output_cls_buffer_size[num_stage];
            int det_output_obj_buffer_size[num_stage];
            int det_output_bbox_buffer_size[num_stage];

            std::shared_ptr<ai::backend::Infer> model_;
            std::shared_ptr<ModelInfo> model_info = nullptr;

            // 仿射矩阵的声明
            std::vector<AffineMatrix> affine_matrixs;
            const uint8_t const_value = 114; // 图片resize补边时的值

            // 使用自定义的Memory类用来申请gpu/cpu内存
            std::vector<std::shared_ptr<Memory<unsigned char>>> preprocess_buffers_;
            Memory<float> input_buffer_, output_boxarray_;
            Memory<float> det_cls_predicts_[num_stage];
            Memory<float> det_bbox_predicts_[num_stage];
            Memory<float> det_obj_predicts_[num_stage];

            // 使用cuda流进行操作
            cudaStream_t cu_stream;

            // time
            Timer timer;
        };
    }
}

#endif