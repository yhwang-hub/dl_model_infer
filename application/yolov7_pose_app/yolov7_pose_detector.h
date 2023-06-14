#ifndef _YOLOV7_POSE_DETECTOR_H
#define _YOLOV7_POSE_DETECTOR_H

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
    namespace yolov7_pose_infer
    {
        using namespace ai::modelInfo;
        using namespace ai::utils;
        using namespace ai::cvUtil;
        using namespace ai::memory;
        using namespace ai::preprocess;
        using namespace ai::postprocess;

        class yolov7_pose_detector
        {
        public:
            yolov7_pose_detector() = default;
            ~yolov7_pose_detector();

            void initParameters(
                const std::string& engine_file,
                float score_thr = 0.5f,
                float nms_thr = 0.45f
            );

            void adjust_memory(int batch_size);

            PoseBoxArray forward(const Image& image);
            BatchPoseBoxArray forwards(const std::vector<Image>& images);

            void preprocess_gpu(
                int ibatch, const Image& image,
                std::shared_ptr<Memory<unsigned char>> preprocess_buffer,
                AffineMatrix& affine,
                cudaStream_t stream_
            );

            void postprocess_gpu(int ibatch, cudaStream_t stream_);
            BatchPoseBoxArray parser_box(int num_image);\

        private:
            static const int num_stages = 4;
            const int det_obj_len = 1;
            const int det_bbox_len = 4;
            const int det_cls_len = 1;
            const int det_info_len_i = det_cls_len + det_bbox_len + det_obj_len;
            const int det_info_len_kpt = 17 * 3;
            const int strides[num_stages] = {8, 16, 32, 64};
            int det_output_buffer_size[num_stages];
            const int netAnchors[num_stages][6] = {
                {19.0f,  27.0f,  44.0f,40.0f,  38.0f,  94.0f},
                {96.0f,  68.0f,  86.0f,152.0f, 180.0f,137.0f},
                {140.0f,301.0f, 303.0f,264.0f, 238.0f,542.0f},
                {436.0f,615.0f, 739.0f,380.0f, 925.0f,792.0f},
            };

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
                return static_cast<float>(1.f / (1.f + expf(-x)));
            }
        };
    }
}

#endif // _YOLOV7_POSE_DETECTOR_H