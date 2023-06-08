#ifndef _YOLOV8_POSE_DETECTOR_H
#define _YOLOV8_POSE_DETECTOR_H


#include <memory>
#include "../../../utils/backend/tensorrt/trt_infer.h"
#include "../../../utils/common/model_info.h"
#include "../../../utils/common/utils.h"
#include "../../../utils/common/cv_cpp_utils.h"
#include "../../../utils/common/memory.h"
#include "../../../utils/preprocess/pre_process.h"
#include "../../../utils/postprocess/post_process.h"

namespace tensorrt_infer
{
    namespace yolov8_infer
    {
        using namespace ai::modelInfo;
        using namespace ai::utils;
        using namespace ai::cvUtil;
        using namespace ai::memory;
        using namespace ai::preprocess;
        using namespace ai::postprocess;
        
        class yolov8_pose_detector
        {
        public:
            yolov8_pose_detector() = default;
            ~yolov8_pose_detector();
            void initParameters(
                const std::string& engine_file,
                float score_thr = 0.5f,
                float nms_thr = 0.45f
            );

            void adjust_memory(int batch_size);

            // forward
            PoseBoxArray forward(const Image& image);
            BatchPoseBoxArray forwards(const std::vector<Image>& images);

            // 模型预处理和后处理
            void preprocess_gpu(
                int ibatch, const Image& image,
                std::shared_ptr<Memory<unsigned char>> preprocess_buffer,
                AffineMatrix& affine,
                cudaStream_t stream_
            );

            void postprocess_gpu(int ibatch, cudaStream_t stream_);
            BatchPoseBoxArray parser_box(int num_image);

        private:
            std::shared_ptr<ai::backend::Infer> model_;
            std::shared_ptr<ModelInfo> model_info = nullptr;

            // 仿射矩阵的声明
            std::vector<AffineMatrix> affine_matrixs;
            const uint8_t const_value = 114;

            std::vector<std::shared_ptr<Memory<unsigned char>>> preprocess_buffers_;
            Memory<float> input_buffer_, bbox_predict_, output_boxarray_;

            cudaStream_t cu_stream;

            Timer timer;
        };
    }
}

#endif // _YOLOv8_POSE_DETECTOR_H