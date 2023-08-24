#include "bevdet4d_detector.h"

namespace tensorrt_infer
{
    namespace bevdet4d_infer
    {
        bevdet4d_detector::bevdet4d_detector(const std::string &model_config_file, int n_img,
                            std::vector<Eigen::Matrix3f> _cams_intrin,
                            std::vector<Eigen::Quaternion<float>> _cams2ego_rot,
                            std::vector<Eigen::Translation3f> _cams2ego_trans,
                            const std::string &imgstage_file, 
                            const std::string &bevstage_file)
        {
            auto start1 = std::chrono::high_resolution_clock::now();
            initParameters(model_config_file, _cams_intrin, _cams2ego_rot, _cams2ego_trans,
                        imgstage_file, bevstage_file);
            if (n_img != bev_model_info->preProCfg.N_img)
            {
                printf("BEVDet need %d images, but given %d images!", bev_model_info->preProCfg.N_img, n_img);
            }
            auto start2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> t1 = start2 - start1;
            printf("[initParameters cost time]: %.4lf ms\n", t1.count() * 1000);
            InitViewTransformer();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> t2 = end - start2;
            printf("[InitVewTransformer cost time]: %.4lf ms\n", t2.count() * 1000);
        }

        void bevdet4d_detector::initParameters(const std::string &model_config_file,
                                std::vector<Eigen::Matrix3f> _cams_intrin,
                                std::vector<Eigen::Quaternion<float>> _cams2ego_rot,
                                std::vector<Eigen::Translation3f> _cams2ego_trans,
                                const std::string &imgstage_file,
                                const std::string &bevstage_file)
        {
            this->bev_model_info = std::make_shared<BEVModelInfo>();

            this->imgstage_model_ = trt::infer::load(imgstage_file);
            this->bevstage_model_ = trt::infer::load(bevstage_file);
            this->imgstage_model_->print();
            this->bevstage_model_->print();

            bev_model_info->preProCfg.cams_intrin = _cams_intrin;
            bev_model_info->preProCfg.cams2ego_rot = _cams2ego_rot;
            bev_model_info->preProCfg.cams2ego_trans = _cams2ego_trans;

            YAML::Node model_config = YAML::LoadFile(model_config_file);
            bev_model_info->preProCfg.N_img = model_config["data_config"]["Ncams"].as<int>();
            bev_model_info->preProCfg.src_img_h = model_config["data_config"]["src_size"][0].as<int>();
            bev_model_info->preProCfg.src_img_w = model_config["data_config"]["src_size"][1].as<int>();
            bev_model_info->preProCfg.input_img_h = model_config["data_config"]["input_size"][0].as<int>();
            bev_model_info->preProCfg.input_img_w = model_config["data_config"]["input_size"][1].as<int>();
            bev_model_info->preProCfg.crop_h = model_config["data_config"]["crop"][0].as<int>();
            bev_model_info->preProCfg.crop_w = model_config["data_config"]["crop"][1].as<int>();
            bev_model_info->preProCfg.mean.x = model_config["mean"][0].as<float>();
            bev_model_info->preProCfg.mean.y = model_config["mean"][1].as<float>();
            bev_model_info->preProCfg.mean.z = model_config["mean"][2].as<float>();
            bev_model_info->preProCfg.std.x = model_config["std"][0].as<float>();
            bev_model_info->preProCfg.std.y = model_config["std"][1].as<float>();
            bev_model_info->preProCfg.std.z = model_config["std"][2].as<float>();
            bev_model_info->preProCfg.down_sample = model_config["model"]["down_sample"].as<int>();
            bev_model_info->preProCfg.depth_start = model_config["grid_config"]["depth"][0].as<float>();
            bev_model_info->preProCfg.depth_end = model_config["grid_config"]["depth"][1].as<float>();
            bev_model_info->preProCfg.depth_step = model_config["grid_config"]["depth"][2].as<float>();
            bev_model_info->preProCfg.x_start = model_config["grid_config"]["x"][0].as<float>();
            bev_model_info->preProCfg.x_end = model_config["grid_config"]["x"][1].as<float>();
            bev_model_info->preProCfg.x_step = model_config["grid_config"]["x"][2].as<float>();
            bev_model_info->preProCfg.y_start = model_config["grid_config"]["y"][0].as<float>();
            bev_model_info->preProCfg.y_end = model_config["grid_config"]["y"][1].as<float>();
            bev_model_info->preProCfg.y_step = model_config["grid_config"]["y"][2].as<float>();
            bev_model_info->preProCfg.z_start = model_config["grid_config"]["z"][0].as<float>();
            bev_model_info->preProCfg.z_end = model_config["grid_config"]["z"][1].as<float>();
            bev_model_info->preProCfg.z_step = model_config["grid_config"]["z"][2].as<float>();
            bev_model_info->preProCfg.bevpool_channel = model_config["model"]["bevpool_channels"].as<int>();
            bev_model_info->postProCfg.nms_pre_maxnum = model_config["test_cfg"]["max_per_img"].as<int>();
            bev_model_info->postProCfg.nms_post_maxnum = model_config["test_cfg"]["post_max_size"].as<int>();
            bev_model_info->postProCfg.score_thresh = model_config["test_cfg"]["score_threshold"].as<float>();
            bev_model_info->postProCfg.nms_overlap_thresh = model_config["test_cfg"]["nms_thr"][0].as<float>();
            bev_model_info->postProCfg.use_depth = model_config["use_depth"].as<bool>();
            bev_model_info->postProCfg.use_adj = model_config["use_adj"].as<bool>();

            if (model_config["sampling"].as<std::string>() == "bicubic")
            {
                bev_model_info->preProCfg.pre_sample = Sampler::bicubic;
            }
            else
            {
                bev_model_info->preProCfg.pre_sample = Sampler::nearest;
            }

            std::vector<std::vector<float>> nms_factor_temp = \
                    model_config["test_cfg"]["nms_rescale_factor"].as<std::vector<std::vector<float>>>();
            bev_model_info->postProCfg.nms_rescale_factor.clear();
            for (auto task_factors : nms_factor_temp)
            {
                for (float factor : task_factors)
                {
                    bev_model_info->postProCfg.nms_rescale_factor.push_back(factor);
                }
            }
            
            std::vector<std::vector<std::string>> class_name_pre_task;
            bev_model_info->postProCfg.class_num = 0;
            YAML::Node tasks = model_config["model"]["tasks"];
            bev_model_info->postProCfg.class_num_pre_task = std::vector<int>();
            for(auto it : tasks)
            {
                int num = it["num_class"].as<int>();
                bev_model_info->postProCfg.class_num_pre_task.push_back(num);
                bev_model_info->postProCfg.class_num += num;
                class_name_pre_task.push_back(it["class_names"].as<std::vector<std::string>>());
            }

            YAML::Node common_head_channel = model_config["model"]["common_head"]["channels"];
            YAML::Node common_head_name = model_config["model"]["common_head"]["names"];
            for (size_t i = 0; i < common_head_channel.size(); i++)
            {
                bev_model_info->postProCfg.out_num_task_head[common_head_name[i].as<std::string>()] = common_head_channel[i].as<int>();
            }

            bev_model_info->preProCfg.resize_radio = (float)bev_model_info->preProCfg.input_img_w /
                                                            bev_model_info->preProCfg.src_img_w;
            bev_model_info->preProCfg.feat_h = bev_model_info->preProCfg.input_img_h /
                                                            bev_model_info->preProCfg.down_sample;
            bev_model_info->preProCfg.feat_w = bev_model_info->preProCfg.input_img_w /
                                                            bev_model_info->preProCfg.down_sample;
            bev_model_info->preProCfg.depth_num = (bev_model_info->preProCfg.depth_end - bev_model_info->preProCfg.depth_start) /
                                                            bev_model_info->preProCfg.depth_step;
            bev_model_info->preProCfg.xgrid_num = (bev_model_info->preProCfg.x_end - bev_model_info->preProCfg.x_start) /
                                                            bev_model_info->preProCfg.x_step;
            bev_model_info->preProCfg.ygrid_num = (bev_model_info->preProCfg.y_end - bev_model_info->preProCfg.y_start) /
                                                            bev_model_info->preProCfg.y_step;
            bev_model_info->preProCfg.zgrid_num = (bev_model_info->preProCfg.z_end - bev_model_info->preProCfg.z_start) /
                                                            bev_model_info->preProCfg.z_step;

            bev_model_info->preProCfg.bev_h = bev_model_info->preProCfg.ygrid_num;
            bev_model_info->preProCfg.bev_w = bev_model_info->preProCfg.xgrid_num;

            bev_model_info->preProCfg.post_rot << bev_model_info->preProCfg.resize_radio, 0, 0,
                                                  0, bev_model_info->preProCfg.resize_radio, 0,
                                                  0,               0,                        1;
            bev_model_info->preProCfg.post_trans.translation() << -bev_model_info->preProCfg.crop_w, -bev_model_info->preProCfg.crop_h, 0;

            bev_model_info->postProCfg.num_points = bev_model_info->preProCfg.N_img *
                                                bev_model_info->preProCfg.depth_num *
                                                bev_model_info->preProCfg.feat_h *
                                                bev_model_info->preProCfg.feat_w;

            img_input_buffer_size_ = bev_model_info->preProCfg.N_img * 3 *
                                    bev_model_info->preProCfg.input_img_h *
                                    bev_model_info->preProCfg.input_img_w;

            rot_buffer_size_ = bev_model_info->preProCfg.N_img * 3 * 3;
            trans_buffer_size_ = bev_model_info->preProCfg.N_img * 3;
            intrin_buffer_size_ = bev_model_info->preProCfg.N_img * 3 * 3;
            post_rot_buffer_size_ = bev_model_info->preProCfg.N_img * 3 * 3;
            post_trans_buffer_size_ = bev_model_info->preProCfg.N_img * 3;
            bda_buffer_size_ = 1 * 3 * 3;
            src_imgs_buffer_size_ = bev_model_info->preProCfg.N_img * 3 *
                                bev_model_info->preProCfg.src_img_h *
                                bev_model_info->preProCfg.src_img_w;

            images_feat_buffer_size_ = 1;
            auto images_feat_dim = imgstage_model_->get_network_dims(images_feat_output_name);
            for (int i = 0; i < images_feat_dim.size(); i++)
            {
               images_feat_buffer_size_ *= images_feat_dim[i];
            }

            depth_buffer_size_ = 1;
            auto depth_dim = imgstage_model_->get_network_dims(depth_output_name);
            for (int i = 0; i < depth_dim.size(); i++)
            {
               depth_buffer_size_ *= depth_dim[i];
            }

            bev_feat_input_buffer_size_ = 1;
            auto bev_feat_input_dim = bevstage_model_->get_network_dims(bev_feat_input_name);
            for (int i = 0; i < bev_feat_input_dim.size(); i++)
            {
               bev_feat_input_buffer_size_ *= bev_feat_input_dim[i];
            }

            reg_output_buffer_size_ = 1;
            auto reg_output_dim = bevstage_model_->get_network_dims(reg_output_name);
            for (int i = 0; i < reg_output_dim.size(); i++)
            {
               reg_output_buffer_size_ *= reg_output_dim[i];
            }

            height_output_buffer_size_ = 1;
            auto height_output_dim = bevstage_model_->get_network_dims(height_output_name);
            for (int i = 0; i < height_output_dim.size(); i++)
            {
               height_output_buffer_size_ *= height_output_dim[i];
            }

            dim_output_buffer_size_ = 1;
            auto dim_output_dim = bevstage_model_->get_network_dims(dim_output_name);
            for (int i = 0; i < dim_output_dim.size(); i++)
            {
               dim_output_buffer_size_ *= dim_output_dim[i];
            }

            rot_output_buffer_size_ = 1;
            auto rot_output_dim = bevstage_model_->get_network_dims(rot_output_name);
            for (int i = 0; i < rot_output_dim.size(); i++)
            {
               rot_output_buffer_size_ *= rot_output_dim[i];
            }

            vel_output_buffer_size_ = 1;
            auto vel_output_dim = bevstage_model_->get_network_dims(vel_output_name);
            for (int i = 0; i < vel_output_dim.size(); i++)
            {
               vel_output_buffer_size_ *= vel_output_dim[i];
            }

            heatmap_output_buffer_size_ = 1;
            auto heatmap_output_dim = bevstage_model_->get_network_dims(heatmap_output_name);
            for (int i = 0; i < heatmap_output_dim.size(); i++)
            {
               heatmap_output_buffer_size_ *= heatmap_output_dim[i];
            }

            bev_model_info->postProCfg.adj_num = 0;
            if(bev_model_info->postProCfg.use_adj)
            {
                bev_model_info->postProCfg.adj_num = model_config["adj_num"].as<int>();
                adj_frame_ptr.reset(
                    new adjFrame(
                        bev_model_info->postProCfg.adj_num,
                        bev_model_info->preProCfg.bev_h * bev_model_info->preProCfg.bev_w,
                        bev_model_info->preProCfg.bevpool_channel
                    )
                );
            }

            postprocess_ptr.reset(
                new BEVDetPostprocessGPU(
                    bev_model_info->postProCfg.class_num,
                    bev_model_info->postProCfg.score_thresh,
                    bev_model_info->postProCfg.nms_overlap_thresh,
                    bev_model_info->postProCfg.nms_pre_maxnum,
                    bev_model_info->postProCfg.nms_post_maxnum,
                    bev_model_info->preProCfg.down_sample,
                    bev_model_info->preProCfg.bev_h, bev_model_info->preProCfg.bev_w,
                    bev_model_info->preProCfg.x_step, bev_model_info->preProCfg.y_step,
                    bev_model_info->preProCfg.x_start, bev_model_info->preProCfg.y_start,
                    bev_model_info->postProCfg.class_num_pre_task,
                    bev_model_info->postProCfg.nms_rescale_factor
                )
            );

            CHECK(cudaStreamCreate(&bev_cu_stream));
        }

        bevdet4d_detector::~bevdet4d_detector()
        {
            CHECK(cudaStreamDestroy(bev_cu_stream));
        }

        void bevdet4d_detector::adjust_memory()
        {
            src_imgs_dev_buffer_.gpu(src_imgs_buffer_size_);

            img_input_buffer_.gpu(img_input_buffer_size_);
            rot_input_buffer_.gpu(rot_buffer_size_);
            trans_input_buffer_.gpu(trans_buffer_size_);
            intrin_input_buffer_.gpu(intrin_buffer_size_);
            post_rot_input_buffer_.gpu(post_rot_buffer_size_);
            post_trans_input_buffer_.gpu(post_trans_buffer_size_);
            bda_input_buffer_.gpu(bda_buffer_size_);
            images_feat_output_buffer_.gpu(images_feat_buffer_size_);
            depth_output_buffer_.gpu(depth_buffer_size_);

            bev_feat_input_buffer_.gpu(bev_feat_input_buffer_size_);
            reg_output_buffer_.gpu(reg_output_buffer_size_);
            height_output_buffer_.gpu(height_output_buffer_size_);
            dim_output_buffer_.gpu(dim_output_buffer_size_);
            rot_output_buffer_.gpu(rot_output_buffer_size_);
            vel_output_buffer_.gpu(vel_output_buffer_size_);
            heatmap_output_buffer_.gpu(heatmap_output_buffer_size_);
        }

        void bevdet4d_detector::InitViewTransformer()
        {
            Eigen::Vector3f* frustum = new Eigen::Vector3f[bev_model_info->postProCfg.num_points];
            for (int i = 0; i < bev_model_info->preProCfg.N_img; i++)
            {
                for (int d_ = 0; d_ < bev_model_info->preProCfg.depth_num; d_++)
                {
                    for (int h_ = 0; h_ < bev_model_info->preProCfg.feat_h; h_++)
                    {
                        for (int w_ = 0; w_ < bev_model_info->preProCfg.feat_w; w_++)
                        {
                            int offset = i * bev_model_info->preProCfg.depth_num *
                                         bev_model_info->preProCfg.feat_h * bev_model_info->preProCfg.feat_w +
                                         d_ * bev_model_info->preProCfg.feat_h * bev_model_info->preProCfg.feat_w +
                                         h_ * bev_model_info->preProCfg.feat_w + w_;

                            (frustum + offset)->x() = (float)w_ * (bev_model_info->preProCfg.input_img_w - 1) /
                                                                (bev_model_info->preProCfg.feat_w - 1);
                            (frustum + offset)->y() = (float)h_ * (bev_model_info->preProCfg.input_img_h - 1) /
                                                                (bev_model_info->preProCfg.feat_h - 1);
                            (frustum + offset)->z() = (float)d_ * bev_model_info->preProCfg.depth_step +
                                                                bev_model_info->preProCfg.depth_start;

                            // eliminate post tranformation
                            *(frustum + offset) -= bev_model_info->preProCfg.post_trans.translation();
                            *(frustum + offset) = bev_model_info->preProCfg.post_rot.inverse() * (*(frustum + offset));

                            (frustum + offset)->x() *= (frustum + offset)->z();
                            (frustum + offset)->y() *= (frustum + offset)->z();

                            // img to ego -> rot -> trans
                            *(frustum + offset) = bev_model_info->preProCfg.cams2ego_rot[i] * bev_model_info->preProCfg.cams_intrin[i].inverse()
                                                *  (*(frustum + offset)) + bev_model_info->preProCfg.cams2ego_trans[i].translation();

                            // voxelization
                            *(frustum + offset) -= Eigen::Vector3f(bev_model_info->preProCfg.x_start,
                                                                   bev_model_info->preProCfg.y_start, 
                                                                   bev_model_info->preProCfg.z_start);

                            (frustum + offset)->x() = (int)((frustum + offset)->x() / bev_model_info->preProCfg.x_step);
                            (frustum + offset)->y() = (int)((frustum + offset)->y() / bev_model_info->preProCfg.y_step);
                            (frustum + offset)->z() = (int)((frustum + offset)->z() / bev_model_info->preProCfg.z_step);
                        }   
                    }
                }
            }

            int* _ranks_depth = new int[bev_model_info->postProCfg.num_points];
            int* _ranks_feat = new int[bev_model_info->postProCfg.num_points];

            for (int i = 0; i < bev_model_info->postProCfg.num_points; i++)
            {
                _ranks_depth[i] = i;
            }
            for (int i = 0; i < bev_model_info->preProCfg.N_img; i++)
            {
                for (int d_ = 0; d_ < bev_model_info->preProCfg.depth_num; d_++)
                {
                    for (int u = 0; u < bev_model_info->preProCfg.feat_h * bev_model_info->preProCfg.feat_w; u++)
                    {
                        int offset = i * (bev_model_info->preProCfg.depth_num * bev_model_info->preProCfg.feat_h * bev_model_info->preProCfg.feat_w) +
                                    d_ * (bev_model_info->preProCfg.feat_h * bev_model_info->preProCfg.feat_w) + u;
                        _ranks_feat[offset] = i * bev_model_info->preProCfg.feat_h * bev_model_info->preProCfg.feat_w + u;
                    }
                }
            }

            std::vector<int> kept;
            for (int i = 0; i < bev_model_info->postProCfg.num_points; i++)
            {
                if ((int)(frustum + i)->x() >= 0 && (int)(frustum + i)->x() < bev_model_info->preProCfg.xgrid_num &&
                    (int)(frustum + i)->y() >= 0 && (int)(frustum + i)->y() < bev_model_info->preProCfg.ygrid_num &&
                    (int)(frustum + i)->z() >= 0 && (int)(frustum + i)->z() < bev_model_info->preProCfg.zgrid_num)
                {
                    kept.push_back(i);
                }
            }

            
            bev_model_info->preProCfg.valid_feat_num = kept.size();

            int* ranks_depth_host = new int[bev_model_info->preProCfg.valid_feat_num];
            int* ranks_feat_host = new int[bev_model_info->preProCfg.valid_feat_num];
            int* ranks_bev_host = new int[bev_model_info->preProCfg.valid_feat_num];
            int* order = new int[bev_model_info->preProCfg.valid_feat_num];

            for(int i = 0; i < bev_model_info->preProCfg.valid_feat_num; i++)
            {
                Eigen::Vector3f &p = frustum[kept[i]];
                ranks_bev_host[i] = (int)p.z() * bev_model_info->preProCfg.xgrid_num * bev_model_info->preProCfg.ygrid_num + 
                                    (int)p.y() * bev_model_info->preProCfg.xgrid_num + (int)p.x();
                order[i] = i;
            }

            thrust::sort_by_key(ranks_bev_host, ranks_bev_host + bev_model_info->preProCfg.valid_feat_num, order);
            for(int i = 0; i < bev_model_info->preProCfg.valid_feat_num; i++)
            {
                ranks_depth_host[i] = _ranks_depth[kept[order[i]]];
                ranks_feat_host[i] = _ranks_feat[kept[order[i]]];
            }

            delete[] _ranks_depth;
            delete[] _ranks_feat;
            delete[] frustum;
            delete[] order;

            std::vector<int> interval_starts_host;
            std::vector<int> interval_lengths_host;

            interval_starts_host.push_back(0);
            int len = 1;
            for(int i = 1; i < bev_model_info->preProCfg.valid_feat_num; i++)
            {
                if(ranks_bev_host[i] != ranks_bev_host[i - 1])
                {
                    interval_starts_host.push_back(i);
                    interval_lengths_host.push_back(len);
                    len=1;
                }
                else
                {
                    len++;
                }
            }

            interval_lengths_host.push_back(len);
            bev_model_info->preProCfg.unique_bev_num = interval_lengths_host.size();

            ranks_bev_buffer_size_ = bev_model_info->preProCfg.valid_feat_num;
            ranks_depth_buffer_size_ = bev_model_info->preProCfg.valid_feat_num;
            ranks_feat_buffer_size_ = bev_model_info->preProCfg.valid_feat_num;
            interval_starts_buffer_size_ = bev_model_info->preProCfg.unique_bev_num;
            interval_lengths_buffer_size_ = bev_model_info->preProCfg.unique_bev_num;

            ranks_bev_dev_buffer_.gpu(ranks_bev_buffer_size_);
            ranks_depth_dev_buffer_.gpu(ranks_depth_buffer_size_);
            ranks_feat_dev_buffer_.gpu(ranks_feat_buffer_size_);
            interval_starts_dev_buffer_.gpu(interval_lengths_buffer_size_);
            interval_lengths_dev_buffer_.gpu(interval_lengths_buffer_size_);

            int* ranks_bev_dev_device = ranks_bev_dev_buffer_.gpu();
            int* ranks_depth_dev_device = ranks_depth_dev_buffer_.gpu();
            int* ranks_feat_dev_device = ranks_feat_dev_buffer_.gpu();
            int* interval_starts_dev_device = interval_starts_dev_buffer_.gpu();
            int* interval_lengths_dev_device = interval_lengths_dev_buffer_.gpu();

            CHECK(cudaMemcpy(ranks_bev_dev_device, ranks_bev_host, ranks_bev_buffer_size_ * sizeof(int), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(ranks_depth_dev_device, ranks_depth_host, ranks_depth_buffer_size_ * sizeof(int), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(ranks_feat_dev_device, ranks_feat_host, ranks_feat_buffer_size_ * sizeof(int), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(interval_starts_dev_device, interval_starts_host.data(), interval_lengths_buffer_size_ * sizeof(int), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(interval_lengths_dev_device, interval_lengths_host.data(), interval_lengths_buffer_size_ * sizeof(int), cudaMemcpyHostToDevice));

            delete[] ranks_bev_host;
            delete[] ranks_depth_host;
            delete[] ranks_feat_host;
        }

        void bevdet4d_detector::InitDepth(const std::vector<Eigen::Quaternion<float>> &curr_cams2ego_rot,
                           const std::vector<Eigen::Translation3f> &curr_cams2ego_trans,
                           const std::vector<Eigen::Matrix3f> &cur_cams_intrin)
        {
            float* rot_host = new float[bev_model_info->preProCfg.N_img * 3 * 3];
            float* trans_host = new float[bev_model_info->preProCfg.N_img * 3];
            float* intrin_host = new float[bev_model_info->preProCfg.N_img * 3 * 3];
            float* post_rot_host = new float[bev_model_info->preProCfg.N_img * 3 * 3];
            float* post_trans_host = new float[bev_model_info->preProCfg.N_img * 3];
            float* bda_host = new float[1 * 3 * 3];

            for(int i = 0; i < bev_model_info->preProCfg.N_img; i++)
            {
                for(int j = 0; j < 3; j++)
                {
                    for(int k = 0; k < 3; k++)
                    {
                        rot_host[i * 9 + j * 3 + k] = curr_cams2ego_rot[i].matrix()(j, k);
                        intrin_host[i * 9 + j * 3 + k] = cur_cams_intrin[i](j, k);
                        post_rot_host[i * 9 + j * 3 + k] = bev_model_info->preProCfg.post_rot(j, k);
                    }
                    trans_host[i * 3 + j] = curr_cams2ego_trans[i].translation()(j);
                    post_trans_host[i * 3 + j] = bev_model_info->preProCfg.post_trans.translation()(j);
                }
            }

            for(int i = 0; i < 3; i++)
            {
                for(int j = 0; j < 3; j++)
                {
                    if(i == j)
                    {
                        bda_host[i * 3 + j] = 1.0;
                    }
                    else
                    {
                        bda_host[i * 3 + j] = 0.0;
                    }
                }
            }

            float* rot_input_device = rot_input_buffer_.gpu();
            float* trans_input_device = trans_input_buffer_.gpu();
            float* intrin_input_device = intrin_input_buffer_.gpu();
            float* post_rot_input_device = post_rot_input_buffer_.gpu();
            float* post_trans_input_device = post_trans_input_buffer_.gpu();
            float* bda_input_device = bda_input_buffer_.gpu();

            CHECK(cudaMemcpy(rot_input_device, rot_host, rot_buffer_size_ * sizeof(float), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(trans_input_device, trans_host, trans_buffer_size_ * sizeof(float), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(intrin_input_device, intrin_host, intrin_buffer_size_ * sizeof(float), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(post_rot_input_device, post_rot_host, post_rot_buffer_size_ * sizeof(float), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(post_trans_input_device, post_trans_host, post_trans_buffer_size_ * sizeof(float), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(bda_input_device, bda_host, bda_buffer_size_ * sizeof(float), cudaMemcpyHostToDevice));

            delete[] rot_host;
            delete[] trans_host;
            delete[] intrin_host;
            delete[] post_rot_host;
            delete[] post_trans_host;
            delete[] bda_host;
        }

        void bevdet4d_detector::GetAdjFrameFeature(const std::string &curr_scene_token, 
                                    const Eigen::Quaternion<float> &ego2global_rot,
                                    const Eigen::Translation3f &ego2global_trans,
                                    float* bev_buffer, cudaStream_t stream)
        {
            /* bev_buffer : 720 * 128 x 128
            */
            bool reset = false;
            if(adj_frame_ptr->buffer_num == 0 || adj_frame_ptr->lastScenesToken() != curr_scene_token)
            {
                adj_frame_ptr->reset();
                for(int i = 0; i < bev_model_info->postProCfg.adj_num; i++)
                {
                    adj_frame_ptr->saveFrameBuffer(bev_buffer, curr_scene_token, ego2global_rot, ego2global_trans);
                }
                reset = true;
            }

            for(int i = 0; i < bev_model_info->postProCfg.adj_num; i++){
                const float* adj_buffer = adj_frame_ptr->getFrameBuffer(i);

                Eigen::Quaternion<float> adj_ego2global_rot;
                Eigen::Translation3f adj_ego2global_trans;
                adj_frame_ptr->getEgo2Global(i, adj_ego2global_rot, adj_ego2global_trans);

                // cudaStream_t stream;
                CHECK(cudaStreamCreate(&stream));
                AlignBEVFeature(ego2global_rot, adj_ego2global_rot, ego2global_trans,
                                adj_ego2global_trans, adj_buffer,
                                bev_buffer + (i + 1) *
                                bev_model_info->preProCfg.bev_w *
                                bev_model_info->preProCfg.bev_h *
                                bev_model_info->preProCfg.bevpool_channel, stream);
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaStreamDestroy(stream));
            }

            if(!reset)
            {
                adj_frame_ptr->saveFrameBuffer(bev_buffer, curr_scene_token, ego2global_rot, ego2global_trans);
            }
        }

        void bevdet4d_detector::AlignBEVFeature(const Eigen::Quaternion<float> &curr_ego2global_rot,
                                    const Eigen::Quaternion<float> &adj_ego2global_rot,
                                    const Eigen::Translation3f &curr_ego2global_trans,
                                    const Eigen::Translation3f &adj_ego2global_trans,
                                    const float* input_bev,
                                    float* output_bev,
                                    cudaStream_t stream)
        {
            Eigen::Matrix4f curr_e2g_transform;
            Eigen::Matrix4f adj_e2g_transform;

            for(int i = 0; i < 3; i++)
            {
                for(int j = 0; j < 3; j++)
                {
                    curr_e2g_transform(i, j) = curr_ego2global_rot.matrix()(i, j);
                    adj_e2g_transform(i, j) = adj_ego2global_rot.matrix()(i, j);
                }
            }
            for(int i = 0; i < 3; i++)
            {
                curr_e2g_transform(i, 3) = curr_ego2global_trans.vector()(i);
                adj_e2g_transform(i, 3) = adj_ego2global_trans.vector()(i);

                curr_e2g_transform(3, i) = 0.0;
                adj_e2g_transform(3, i) = 0.0;
            }
            curr_e2g_transform(3, 3) = 1.0;
            adj_e2g_transform(3, 3) = 1.0;

            Eigen::Matrix4f currEgo2adjEgo = adj_e2g_transform.inverse() * curr_e2g_transform;
            Eigen::Matrix3f currEgo2adjEgo_2d;
            for(int i = 0; i < 2; i++)
            {
                for(int j = 0; j < 2; j++)
                {
                    currEgo2adjEgo_2d(i, j) = currEgo2adjEgo(i, j);
                }
            }
            currEgo2adjEgo_2d(2, 0) = 0.0;
            currEgo2adjEgo_2d(2, 1) = 0.0;
            currEgo2adjEgo_2d(2, 2) = 1.0;
            currEgo2adjEgo_2d(0, 2) = currEgo2adjEgo(0, 3);
            currEgo2adjEgo_2d(1, 2) = currEgo2adjEgo(1, 3);

            Eigen::Matrix3f gridbev2egobev;
            gridbev2egobev(0, 0) = bev_model_info->preProCfg.x_step;
            gridbev2egobev(1, 1) = bev_model_info->preProCfg.y_step;
            gridbev2egobev(0, 2) = bev_model_info->preProCfg.x_start;
            gridbev2egobev(1, 2) = bev_model_info->preProCfg.y_start;
            gridbev2egobev(2, 2) = 1.0;

            gridbev2egobev(0, 1) = 0.0;
            gridbev2egobev(1, 0) = 0.0;
            gridbev2egobev(2, 0) = 0.0;
            gridbev2egobev(2, 1) = 0.0;

            Eigen::Matrix3f currgrid2adjgrid = gridbev2egobev.inverse() * currEgo2adjEgo_2d * gridbev2egobev;


            float* grid_dev;
            float* transform_dev;
            CHECK(cudaMalloc((void**)&grid_dev,
                    bev_model_info->preProCfg.bev_h * bev_model_info->preProCfg.bev_w * 2 * sizeof(float)));
            CHECK(cudaMalloc((void**)&transform_dev, 9 * sizeof(float)));


            CHECK(cudaMemcpy(transform_dev, Eigen::Matrix3f(currgrid2adjgrid.transpose()).data(), 
                                                        9 * sizeof(float), cudaMemcpyHostToDevice));

            compute_sample_grid_cuda(grid_dev, transform_dev,
                    bev_model_info->preProCfg.bev_w, bev_model_info->preProCfg.bev_h, stream);

            int output_dim[4] = {1, bev_model_info->preProCfg.bevpool_channel, bev_model_info->preProCfg.bev_w, bev_model_info->preProCfg.bev_h};
            int input_dim[4] = {1, bev_model_info->preProCfg.bevpool_channel, bev_model_info->preProCfg.bev_w, bev_model_info->preProCfg.bev_h};
            int grid_dim[4] = {1, bev_model_info->preProCfg.bev_w, bev_model_info->preProCfg.bev_h, 2};
            

            grid_sample(output_bev, input_bev, grid_dev, output_dim, input_dim, grid_dim, 4,
                        GridSamplerInterpolation::Bilinear, GridSamplerPadding::Zeros, true, stream);
            CHECK(cudaFree(grid_dev));
            CHECK(cudaFree(transform_dev));
        }

        void bevdet4d_detector::preprocess_gpu(const camsData& cam_data, int idx)
        {
            uchar* src_img_device = src_imgs_dev_buffer_.gpu();
            float* img_input_device = (float*)img_input_buffer_.gpu();

            CHECK(cudaMemcpy(src_img_device, cam_data.imgs_dev,
                        bev_model_info->preProCfg.N_img * 3 *
                        bev_model_info->preProCfg.src_img_h *
                        bev_model_info->preProCfg.src_img_w * sizeof(uchar),
                        cudaMemcpyDeviceToDevice));
            bevdet_preprocess(src_img_device, img_input_device,
                        bev_model_info->preProCfg.N_img,
                        bev_model_info->preProCfg.src_img_h,
                        bev_model_info->preProCfg.src_img_w,
                        bev_model_info->preProCfg.input_img_h,
                        bev_model_info->preProCfg.input_img_w,
                        bev_model_info->preProCfg.resize_radio,
                        bev_model_info->preProCfg.resize_radio,
                        bev_model_info->preProCfg.crop_h,
                        bev_model_info->preProCfg.crop_w,
                        bev_model_info->preProCfg.mean,
                        bev_model_info->preProCfg.std,
                        bev_model_info->preProCfg.pre_sample);

            InitDepth(cam_data.param.cams2ego_rot, cam_data.param.cams2ego_trans, cam_data.param.cams_intrin);
        }

        void bevdet4d_detector::forward(const camsData& cam_data, std::vector<bevBox>& out_detections, float &cost_time, int idx)
        {
            printf("-------------------%d-------------------\n", idx + 1);
            printf("scenes_token : %s, timestamp : %lld\n", cam_data.param.scene_token.data(), cam_data.param.timestamp);

            auto adjust_memory_start = std::chrono::high_resolution_clock::now();
            adjust_memory();
            auto adjust_memory_stop = std::chrono::high_resolution_clock::now();

            auto pre_start = std::chrono::high_resolution_clock::now();

            preprocess_gpu(cam_data, idx);
            CHECK(cudaDeviceSynchronize());
            auto pre_end = std::chrono::high_resolution_clock::now();

            std::vector<void*> imgstage_bindings{img_input_buffer_.gpu()};
            imgstage_bindings.push_back(rot_input_buffer_.gpu());
            imgstage_bindings.push_back(trans_input_buffer_.gpu());
            imgstage_bindings.push_back(intrin_input_buffer_.gpu());
            imgstage_bindings.push_back(post_rot_input_buffer_.gpu());
            imgstage_bindings.push_back(post_trans_input_buffer_.gpu());
            imgstage_bindings.push_back(bda_input_buffer_.gpu());
            imgstage_bindings.push_back(depth_output_buffer_.gpu());
            imgstage_bindings.push_back(images_feat_output_buffer_.gpu());

            std::cout << "----------start imgstage infer---------" << std::endl;

            if(!this->imgstage_model_->forward(imgstage_bindings, bev_cu_stream))
            {
                INFO("Failed to img stage forward.");
                return;
            }
            CHECK(cudaDeviceSynchronize());
            auto imgstage_end = std::chrono::high_resolution_clock::now();

            float* depth_output_device = depth_output_buffer_.gpu();
            float* images_feat_output_device = images_feat_output_buffer_.gpu();
            int* ranks_depth_dev_device = ranks_depth_dev_buffer_.gpu();
            int* ranks_feat_dev_device = ranks_feat_dev_buffer_.gpu();
            int* ranks_bev_dev_device = ranks_bev_dev_buffer_.gpu();
            int* interval_starts_dev_device = interval_starts_dev_buffer_.gpu();
            int* interval_lengths_dev_device = interval_lengths_dev_buffer_.gpu();
            float* bev_feat_input_device = bev_feat_input_buffer_.gpu();

            std::cout << "----------start bevpoolv2---------" << std::endl;

            bev_pool_v2(bev_model_info->preProCfg.bevpool_channel,
                    bev_model_info->preProCfg.unique_bev_num,
                    bev_model_info->preProCfg.bev_h * bev_model_info->preProCfg.bev_w,
                    depth_output_device,
                    images_feat_output_device,
                    ranks_depth_dev_device,
                    ranks_feat_dev_device,
                    ranks_bev_dev_device,
                    interval_starts_dev_device,
                    interval_lengths_dev_device,
                    bev_feat_input_device
            );
            CHECK(cudaDeviceSynchronize());
            auto bevpool_end = std::chrono::high_resolution_clock::now();

            std::cout << "----------start GetAdjFrameFeature---------" << std::endl;

            if(bev_model_info->postProCfg.use_adj)
            {
                GetAdjFrameFeature(cam_data.param.scene_token,
                                   cam_data.param.ego2global_rot, 
                                   cam_data.param.ego2global_trans,
                                   bev_feat_input_device,
                                   bev_cu_stream);
                CHECK(cudaDeviceSynchronize());
            }
            auto align_feat_end = std::chrono::high_resolution_clock::now();

            std::cout << "----------start bevstage infer---------" << std::endl;

            std::vector<void*> bevstage_bindings{bev_feat_input_buffer_.gpu()};
            bevstage_bindings.push_back(reg_output_buffer_.gpu());
            bevstage_bindings.push_back(height_output_buffer_.gpu());
            bevstage_bindings.push_back(dim_output_buffer_.gpu());
            bevstage_bindings.push_back(rot_output_buffer_.gpu());
            bevstage_bindings.push_back(vel_output_buffer_.gpu());
            bevstage_bindings.push_back(heatmap_output_buffer_.gpu());
            if(!this->bevstage_model_->forward(bevstage_bindings, bev_cu_stream))
            {
                INFO("Failed to bev stage forward.");
                return;
            }
            CHECK(cudaDeviceSynchronize());
            auto bevstage_end = std::chrono::high_resolution_clock::now();

            std::cout << "----------start postprocess---------" << std::endl;

            postprocess_ptr->DoPostprocess(bevstage_bindings, out_detections);
            CHECK(cudaDeviceSynchronize());
            auto post_end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> pre_t = pre_end - pre_start;
            std::chrono::duration<double> imgstage_t = imgstage_end - pre_end;
            std::chrono::duration<double> bevpool_t = bevpool_end - imgstage_end;
            std::chrono::duration<double> align_t = std::chrono::duration<double>(0);
            std::chrono::duration<double> bevstage_t;
            if(bev_model_info->postProCfg.use_adj)
            {
                align_t = align_feat_end - bevpool_end;
                bevstage_t = bevstage_end - align_feat_end;
            }
            else
            {
                bevstage_t = bevstage_end - bevpool_end;
            }
            std::chrono::duration<double> post_t = post_end - bevstage_end;

            std::chrono::duration<double> adjust_memory_t = adjust_memory_stop - adjust_memory_start;
            printf("[Adjust Memory   ] cost time: %5.3lf ms\n", adjust_memory_t.count() * 1000);

            printf("[Preprocess   ] cost time: %5.3lf ms\n", pre_t.count() * 1000);
            printf("[Image stage  ] cost time: %5.3lf ms\n", imgstage_t.count() * 1000);
            printf("[BEV pool     ] cost time: %5.3lf ms\n", bevpool_t.count() * 1000);
            if(bev_model_info->postProCfg.use_adj)
            {
                printf("[Align Feature] cost time: %5.3lf ms\n", align_t.count() * 1000);
            }
            printf("[BEV stage    ] cost time: %5.3lf ms\n", bevstage_t.count() * 1000);
            printf("[Postprocess  ] cost time: %5.3lf ms\n", post_t.count() * 1000);

            std::chrono::duration<double> sum_time = post_end - pre_start;
            cost_time = sum_time.count() * 1000;
            printf("[Infer total  ] cost time: %5.3lf ms\n", cost_time);

            printf("Detect %ld objects\n", out_detections.size());
        }
    }
}