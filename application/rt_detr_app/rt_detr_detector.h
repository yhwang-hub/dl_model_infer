#ifndef _RTDETR_DETECT_CUDA_HPP_
#define _RTDETR_DETECT_CUDA_HPP_

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
    namespace rtdetr_cuda
    {
        using namespace ai::modelInfo;
        using namespace ai::utils;
        using namespace ai::cvUtil;
        using namespace ai::memory;
        using namespace ai::preprocess;
        using namespace ai::postprocess;

        class RTDETRDetect
        {
        public:
            RTDETRDetect() = default;
            ~RTDETRDetect();
            void initParameters(const std::string &engine_file, float score_thr = 0.5f); // 初始化参数
            void adjust_memory(int batch_size);                                          // 由于batch size是动态的，所以需要对gpu/cpu内存进行动态的申请

            // forward
            BoxArray forward(const Image &image);
            BatchBoxArray forwards(const std::vector<Image> &images);

            // 模型前后处理
            void preprocess_gpu(int ibatch, const Image &image,
                                shared_ptr<Memory<unsigned char>> preprocess_buffer, cudaStream_t stream_);
            void postprocess_gpu(int ibatch, cudaStream_t stream_);
            BatchBoxArray parser_box(const std::vector<Image> &images);

        private:
            std::shared_ptr<ai::backend::Infer> model_;
            std::shared_ptr<ModelInfo> model_info = nullptr;

            // 使用自定义的Memory类用来申请gpu/cpu内存
            std::vector<std::shared_ptr<Memory<unsigned char>>> preprocess_buffers_;
            Memory<float> input_buffer_, bbox_predict_, output_boxarray_;

            // 使用cuda流进行操作
            cudaStream_t cu_stream;

            // time
            Timer timer;
        };
    }
}

#endif // _RTDETR_DETECT_CUDA_HPP_