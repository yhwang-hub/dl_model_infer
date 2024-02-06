#ifndef _MODEL_INFO_HPP_
#define _MODEL_INFO_HPP_

#include <string>
#include <vector>
#include "cv_cpp_utils.h"

namespace ai
{
    namespace modelInfo
    {
        struct PreprocessImageConfig
        {
            int32_t infer_batch_size{0};        // 模型输入的batch,自动获取，无需配置
            int32_t network_input_width_{0};    // 模型输入的宽,自动获取，无需配置
            int32_t network_input_height_{0};   // 模型输入的高,自动获取，无需配置
            int32_t network_input_channels_{0}; // 模型输入的通道数,自动获取，无需配置
            bool isdynamic_model_ = false;      // 是否是动态模型,自动获取，无需配置

            size_t network_input_numel{0}; // 模型输入h*w*c的大小，无需配置

            ai::cvUtil::Norm normalize_ = ai::cvUtil::Norm::None(); // 对输入图片的预处理进行配置
        };

        struct PostprocessImageConfig
        {
            float confidence_threshold_{0.5f};
            float nms_threshold_{0.45f};

            // 检测分支
            std::vector<int> bbox_head_dims_;
            size_t bbox_head_dims_output_numel_{0}; // 模型输出的大小，无需配置

            // 分割分支
            std::vector<int> seg_head_dims_;
            size_t seg_head_dims_output_numel_{0}; // 模型输出的大小，无需配置

            // pose分支
            int pose_num_ = 0;

            std::vector<int> depth_head_dims_;
            size_t depth_head_dims_output_numel_{0};

            // 模型输出结果解析时的一些参数设置,最好设置为const类型，以免改变
            int MAX_IMAGE_BOXES = 1024;
            int NUM_BOX_ELEMENT = 7;               // left, top, right, bottom, confidence, class, keepflag.一般是固定值，常不修改
            int NUM_ROTATEBOX_ELEMENT = 8;         // enter_x, center_y, width, height, angle, confidence, class, keepflag.一般是固定值，常不修改
            size_t IMAGE_MAX_BOXES_ADD_ELEMENT{0}; // MAX_IMAGE_BOXES * NUM_BOX_ELEMENT

            int MAX_IMAGE_CUBES = 100;
            int NUM_CUBE_ELEMENT = 18;
            size_t IMAGE_MAX_CUBES_ADD_ELEMENT{0};

            int num_classes_ = 0; // 类别，可以通过模型输出维度自动推出，也可以设置
        };

        struct ModelInfo
        {
            std::string m_modelPath; // engine 路径,传参获取，无需配置

            // 后面这两个根据你自己的任务进行参数配置,下面是常用的一些基础配置
            PreprocessImageConfig m_preProcCfg;   // 预处理配置
            PostprocessImageConfig m_postProcCfg; // 后处理配置
        };      
    }
}
#endif // _MODEL_INFO_HPP_