#ifndef _BEVDET_DETECTOR_H
#define _BEVDET_DETECTOR_H


#include <memory>
#include <algorithm>
#include <iostream>
#include <chrono>
#include "../../utils/backend/tensorrt/trt_infer.h"
#include "../../utils/common/model_info.h"
#include "../../utils/common/utils.h"
#include "../../utils/common/cv_cpp_utils.h"
#include "../../utils/common/memory.h"
#include "../../utils/preprocess/pre_process.h"
#include "../../utils/postprocess/post_process.h"
#include "../../utils/kernels/bevpool/bevpool.h"
#include "../../utils/kernels/grid_sampler/grid_sampler.h"
#include "../../utils/backend/backend_infer.h"

namespace tensorrt_infer
{
    namespace bevdet_det_infer
    {
        using namespace ai::modelInfo;
        using namespace ai::utils;
        using namespace ai::cvUtil;
        using namespace ai::memory;
        using namespace ai::preprocess;
        using namespace ai::postprocess;

        struct adjFrame
        {
            adjFrame(){}
            adjFrame(int _n,
                    int _map_size, 
                    int _bev_channel) : 
                    n(_n), 
                    map_size(_map_size), 
                    bev_channel(_bev_channel),
                    scenes_token(_n),
                    ego2global_rot(_n),
                    ego2global_trans(_n)
            {
                CHECK(cudaMalloc((void**)&adj_buffer, _n * _map_size * _bev_channel * sizeof(float)));
            }  
            const std::string& lastScenesToken() const
            {
                return scenes_token[last];
            }

            void reset()
            {
                last = -1;
                buffer_num = 0;
            }

            void saveFrameBuffer(const float* curr_buffer, const std::string &curr_token, 
                                const Eigen::Quaternion<float> &_ego2global_rot,
                                const Eigen::Translation3f &_ego2global_trans)
            {
                last = (last + 1) % n;
                CHECK(cudaMemcpy(adj_buffer + last * map_size * bev_channel, curr_buffer,
                                map_size * bev_channel * sizeof(float), cudaMemcpyDeviceToDevice));
                scenes_token[last] = curr_token;
                ego2global_rot[last] = _ego2global_rot;
                ego2global_trans[last] = _ego2global_trans;
                buffer_num = std::min(buffer_num + 1, n);
            }
            const float* getFrameBuffer(int idx)
            {
                idx = (-idx + last + n) % n;
                return adj_buffer + idx * map_size * bev_channel;
            }
            void getEgo2Global(int idx, Eigen::Quaternion<float> &adj_ego2global_rot, 
                            Eigen::Translation3f &adj_ego2global_trans)
            {
                idx = (-idx + last + n) % n;
                adj_ego2global_rot = ego2global_rot[idx];
                adj_ego2global_trans = ego2global_trans[idx];
            }

            ~adjFrame()
            {
                CHECK(cudaFree(adj_buffer));
            }

            int n;
            int map_size;
            int bev_channel;

            int last;
            int buffer_num;

            std::vector<std::string> scenes_token;
            std::vector<Eigen::Quaternion<float>> ego2global_rot;
            std::vector<Eigen::Translation3f> ego2global_trans;

            float* adj_buffer;
        };

        class bevdet_detector
        {
        public:
            bevdet_detector();
            bevdet_detector(const std::string &model_config_file, int n_img,
                            std::vector<Eigen::Matrix3f> _cams_intrin,
                            std::vector<Eigen::Quaternion<float>> _cams2ego_rot,
                            std::vector<Eigen::Translation3f> _cams2ego_trans,
                            const std::string &imgstage_file, 
                            const std::string &bevstage_file);
            ~bevdet_detector();

            void initParameters(const std::string &model_config_file,
                                std::vector<Eigen::Matrix3f> _cams_intrin,
                                std::vector<Eigen::Quaternion<float>> _cams2ego_rot,
                                std::vector<Eigen::Translation3f> _cams2ego_trans,
                                const std::string &imgstage_file,
                                const std::string &bevstage_file);
            void InitViewTransformer();
            void InitDepth(const std::vector<Eigen::Quaternion<float>> &curr_cams2ego_rot,
                           const std::vector<Eigen::Translation3f> &curr_cams2ego_trans,
                           const std::vector<Eigen::Matrix3f> &cur_cams_intrin);
            void GetAdjFrameFeature(const std::string &curr_scene_token,
                                    const Eigen::Quaternion<float> &ego2global_rot,
                                    const Eigen::Translation3f &ego2global_trans,
                                    float* bev_buffer, cudaStream_t strea);
            void AlignBEVFeature(const Eigen::Quaternion<float> &curr_ego2global_rot,
                                 const Eigen::Quaternion<float> &adj_ego2global_rot,
                                 const Eigen::Translation3f &curr_ego2global_trans,
                                 const Eigen::Translation3f &adj_ego2global_trans,
                                 const float* input_bev,
                                 float* output_bev,
                                 cudaStream_t stream);
            void adjust_memory();
            void forward(const camsData& cam_data, std::vector<bevBox>& out_detections, float &cost_time, int idx = -1);
            void preprocess_gpu(const camsData& cam_data, int idx);
            void postprocess_gpu();

        private:
            std::shared_ptr<ai::backend::Infer> imgstage_model_;
            std::shared_ptr<ai::backend::Infer> bevstage_model_;

            std::shared_ptr<BEVModelInfo> bev_model_info;

            std::unique_ptr<adjFrame> adj_frame_ptr;
            std::unique_ptr<BEVDetPostprocessGPU> postprocess_ptr;

            Memory<float> img_input_buffer_, rot_input_buffer_, trans_input_buffer_,
                        intrin_input_buffer_, post_rot_input_buffer_, post_trans_input_buffer_, bda_input_buffer_;
            Memory<float> images_feat_output_buffer_, depth_output_buffer_;
            Memory<float> bev_feat_input_buffer_;
            Memory<float> reg_output_buffer_, height_output_buffer_, dim_output_buffer_,
                        rot_output_buffer_, vel_output_buffer_, heatmap_output_buffer_;
            Memory<int> ranks_bev_dev_buffer_, ranks_depth_dev_buffer_, ranks_feat_dev_buffer_,
                        interval_starts_dev_buffer_, interval_lengths_dev_buffer_;
            Memory<uchar> src_imgs_dev_buffer_;

            int img_input_buffer_size_;
            int rot_buffer_size_;
            int trans_buffer_size_;
            int intrin_buffer_size_;
            int post_rot_buffer_size_;
            int post_trans_buffer_size_;
            int bda_buffer_size_;
            int images_feat_buffer_size_;
            int depth_buffer_size_;

            int bev_feat_input_buffer_size_;
            int reg_output_buffer_size_;
            int height_output_buffer_size_;
            int dim_output_buffer_size_;
            int rot_output_buffer_size_;
            int vel_output_buffer_size_;
            int heatmap_output_buffer_size_;

            int ranks_bev_buffer_size_;
            int ranks_depth_buffer_size_;
            int ranks_feat_buffer_size_;
            int interval_starts_buffer_size_;
            int interval_lengths_buffer_size_;

            int src_imgs_buffer_size_;

            const std::string images_input_name = "images";
            const std::string rot_input_name = "rot";
            const std::string trans_input_name = "trans";
            const std::string intrin_input_name = "intrin";
            const std::string post_rot_input_name = "post_rot";
            const std::string post_trans_input_name = "post_trans";
            const std::string bda_input_name = "bda";
            const std::string images_feat_output_name = "images_feat";
            const std::string depth_output_name = "depth";

            const std::string bev_feat_input_name = "BEV_feat";
            const std::string reg_output_name = "reg_0";
            const std::string height_output_name = "height_0";
            const std::string dim_output_name = "dim_0";
            const std::string rot_output_name = "rot_0";
            const std::string vel_output_name = "vel_0";
            const std::string heatmap_output_name = "heatmap_0";

            cudaStream_t imgstage_cu_stream;
            cudaStream_t bevstage_cu_stream;
            cudaStream_t bev_cu_stream;

            Timer timer;
        };
    }
}

#endif