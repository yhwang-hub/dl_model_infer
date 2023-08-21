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

            // 模型输出结果解析时的一些参数设置,最好设置为const类型，以免改变
            int MAX_IMAGE_BOXES = 1024;
            int NUM_BOX_ELEMENT = 7;               // left, top, right, bottom, confidence, class,keepflag.一般是固定值，常不修改
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

        struct BEVPreprocessConfig
        {
            int N_img;

            int src_img_h;
            int src_img_w;
            int input_img_h;
            int input_img_w;
            int crop_h;
            int crop_w;
            float resize_radio;
            int down_sample;
            int feat_h;
            int feat_w;
            int bev_h;
            int bev_w;
            int bevpool_channel;

            float depth_start;
            float depth_end;
            float depth_step;
            int depth_num;

            float x_start;
            float x_end;
            float x_step;
            int xgrid_num;

            float y_start;
            float y_end;
            float y_step;
            int ygrid_num;

            float z_start;
            float z_end;
            float z_step;
            int zgrid_num;

            ai::cvUtil::BboxDim mean;
            ai::cvUtil::BboxDim std;

            ai::cvUtil::Sampler pre_sample;

            std::vector<Eigen::Matrix3f> cams_intrin;
            std::vector<Eigen::Quaternion<float>> cams2ego_rot;
            std::vector<Eigen::Translation3f> cams2ego_trans;

            Eigen::Matrix3f post_rot;
            Eigen::Translation3f post_trans;

            int valid_feat_num;
            int unique_bev_num;
        };

        struct BEVPostprocessConfig
        {
            bool use_depth;
            bool use_adj;
            int adj_num;

            int class_num;
            float score_thresh;
            float nms_overlap_thresh;
            int nms_pre_maxnum;
            int nms_post_maxnum;
            int num_points;
            std::vector<float> nms_rescale_factor;
            std::vector<int> class_num_pre_task;
            std::map<std::string, int> out_num_task_head;
        };

        struct BEVModelInfo
        {
            std::string imgstage_modelPath;
            std::string bevstage_modelPath; 

            BEVPreprocessConfig preProCfg;
            BEVPostprocessConfig postProCfg;
        };        
    }
}
#endif // _MODEL_INFO_HPP_