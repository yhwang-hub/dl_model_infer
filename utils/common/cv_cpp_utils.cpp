#include "cv_cpp_utils.h"

namespace ai
{
    namespace cvUtil
    {
        using std::chrono::high_resolution_clock;
        using std::chrono::duration;

        Image cvimg_trans_func(const cv::Mat &image)
        {
            return Image(image.data, image.cols, image.rows, image.channels());
        }

        Norm Norm::mean_std(const float mean[3], const float std[3], float alpha, ChannelType channel_type)
        {
            Norm out;
            out.type = NormType::MeanStd;
            out.alpha = alpha;
            out.channel_type = channel_type;
            memcpy(out.mean, mean, sizeof(out.mean));
            memcpy(out.std, std, sizeof(out.std));
            return out;
        }

        Norm Norm::alpha_beta(float alpha, float beta, ChannelType channel_type)
        {
            Norm out;
            out.type = NormType::AlphaBeta;
            out.alpha = alpha;
            out.beta = beta;
            out.channel_type = channel_type;
            return out;
        }

        Norm Norm::None()
        {
            return Norm();
        };

        void AffineMatrix::compute(const std::tuple<int, int> &from,
                                const std::tuple<int, int> &to,
                                DetectorType detectortype)
        {
            float scale_x = get<0>(to) / (float)get<0>(from);
            float scale_y = get<1>(to) / (float)get<1>(from);
            float scale = std::min(scale_x, scale_y);
            i2d[0] = scale;
            i2d[1] = 0;
            i2d[2] = -scale * get<0>(from) * 0.5 + get<0>(to) * 0.5 + scale * 0.5 - 0.5;
            i2d[3] = 0;
            i2d[4] = scale;
            i2d[5] = -scale * get<1>(from) * 0.5 + get<1>(to) * 0.5 + scale * 0.5 - 0.5;

            if (detectortype == DetectorType::V7 ||
                detectortype == DetectorType::V5 ||
                detectortype == DetectorType::V8 ||
                detectortype == DetectorType::X ||
                detectortype == DetectorType::SMOKE
                )
            {
                i2d[2] = 0;
                i2d[5] = 0;
            }

            double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
            D = D != 0. ? double(1.) / D : double(0.);
            double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
            double b1 = -A11 * i2d[2] - A12 * i2d[5];
            double b2 = -A21 * i2d[2] - A22 * i2d[5];

            d2i[0] = A11;
            d2i[1] = A12;
            d2i[2] = b1;
            d2i[3] = A21;
            d2i[4] = A22;
            d2i[5] = b2;
        }

        InstanceSegmentMap::InstanceSegmentMap(int width, int height)
        {
            this->width = width;
            this->height = height;
            checkRuntime(cudaMallocHost(&this->data, width * height));
        }

        InstanceSegmentMap::~InstanceSegmentMap()
        {
            if (this->data)
            {
                checkRuntime(cudaFreeHost(this->data));
                this->data = nullptr;
            }
            this->width = 0;
            this->height = 0;
        }

        void draw_batch_rectangle(std::vector<cv::Mat> &images, BatchBoxArray &batched_result, const std::string &save_dir, const std::vector<std::string> &classlabels)
        {
            for (int ib = 0; ib < (int)batched_result.size(); ++ib)
            {
                auto &objs = batched_result[ib];
                auto &image = images[ib];
                for (auto &obj : objs)
                {
                    // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.class_label, obj.confidence,\
                            obj.left, obj.top, obj.right, obj.bottom);

                    uint8_t b, g, r;
                    tie(b, g, r) = random_color(obj.class_label);
                    cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                                  cv::Scalar(b, g, r), 2);
                    auto name = classlabels[obj.class_label];
                    auto caption = cv::format("%s %.2f", name.c_str(), obj.confidence);
                    int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                    cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                                  cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
                    cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2,
                                16);
                }
                if (mkdirs(save_dir))
                {
                    std::string save_path = path_join("%s/Infer_%d.jpg", save_dir.c_str(), ib);
                    cv::imwrite(save_path, image);
                }
            }
        }

        static std::vector<cv::Point> xywhr2xyxyxyxy(const RotateBox& box)
        {
            float cos_value = std::cos(box.angle);
            float sin_value = std::sin(box.angle);

            float w_2 = box.width / 2, h_2 = box.height / 2;
            float vec1_x =  w_2 * cos_value, vec1_y = w_2 * sin_value;
            float vec2_x = -h_2 * sin_value, vec2_y = h_2 * cos_value;

            std::vector<cv::Point> corners;
            corners.push_back(cv::Point(box.center_x + vec1_x + vec2_x, box.center_y + vec1_y + vec2_y));
            corners.push_back(cv::Point(box.center_x + vec1_x - vec2_x, box.center_y + vec1_y - vec2_y));
            corners.push_back(cv::Point(box.center_x - vec1_x - vec2_x, box.center_y - vec1_y - vec2_y));
            corners.push_back(cv::Point(box.center_x - vec1_x + vec2_x, box.center_y - vec1_y + vec2_y));

            return corners;
        }

        void draw_batch_rotaterectangle(std::vector<cv::Mat> &images, BatchRotateBoxArray &batched_result, const std::string &save_dir, const std::vector<std::string> &dotalabels)
        {
            for (int ib = 0; ib < (int)batched_result.size(); ++ib)
            {
                auto &objs = batched_result[ib];
                auto &image = images[ib];
                for (auto &obj : objs)
                {
                    // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.class_label, obj.confidence,\
                            obj.left, obj.top, obj.right, obj.bottom);

                    uint8_t b, g, r;
                    tie(b, g, r) = random_color(obj.class_label);
                    auto corners = xywhr2xyxyxyxy(obj);
                    cv::polylines(image, vector<vector<cv::Point>>{corners}, true, cv::Scalar(b, g, r), 2, 16);

                    auto name = dotalabels[obj.class_label];
                    auto caption = cv::format("%s %.2f", name.c_str(), obj.confidence);
                    int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                    cv::rectangle(image, cv::Point(corners[0].x-3, corners[0].y-33), cv::Point(corners[0].x-3 + width, corners[0].y), cv::Scalar(b, g, r), -1);
                    cv::putText(image, caption, cv::Point(corners[0].x-3, corners[0].y-5), 0, 1, cv::Scalar::all(0), 2, 16);
                }
                if (mkdirs(save_dir))
                {
                    std::string save_path = path_join("%s/Infer_%d.jpg", save_dir.c_str(), ib);
                    cv::imwrite(save_path, image);
                }
            }
        }

        void draw_one_image_rectangle(cv::Mat &image, BoxArray &result, const std::string &save_dir, const std::vector<std::string> &classlabels)
        {

            for (auto &obj : result)
            {
                uint8_t b, g, r;
                tie(b, g, r) = random_color(obj.class_label);
                cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                              cv::Scalar(b, g, r), 2);
                auto name = classlabels[obj.class_label];
                auto caption = cv::format("%s %.2f", name.c_str(), obj.confidence);
                int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                              cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
                cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2,
                            16);
            }
            if (mkdirs(save_dir))
            {
                std::string save_path = path_join("%s/Infer_one.jpg", save_dir.c_str());
                cv::imwrite(save_path, image);
            }
        }

        void draw_batch_segment(std::vector<cv::Mat> &images, BatchSegBoxArray &batched_result, const std::string &save_dir,
                                const std::vector<std::string> &classlabels, int img_mask_wh, int network_input_wh)
        {
            for (int ib = 0; ib < (int)batched_result.size(); ++ib)
            {
                auto &objs = batched_result[ib];
                auto &image = images[ib];
                cv::Mat img_mask = cv::Mat::zeros(img_mask_wh, img_mask_wh, CV_8UC3);

                float scale_img_mask_x = (img_mask_wh * 1.0) / image.cols;
                float scale_img_mask_y = (img_mask_wh * 1.0) / image.rows;
                float scale_img_mask_xy = std::min(scale_img_mask_x, scale_img_mask_y);

                int padw = 0, padh = 0;
                if (image.cols > image.rows)
                    padh = (int)(image.cols - image.rows) / 2;
                else
                    padw = (int)(image.rows - image.cols) / 2;

                for (auto &obj : objs)
                {
                    uint8_t b, g, r;
                    tie(b, g, r) = random_color(obj.class_label);
                    cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                                  cv::Scalar(b, g, r), 2);
                    auto name = classlabels[obj.class_label];
                    auto caption = cv::format("%s %.2f", name.c_str(), obj.confidence);
                    int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                    cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                                  cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
                    cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
                    // seg
                    cv::Mat seg_image(obj.seg->height, obj.seg->width, CV_8UC3);
                    cv::Mat seg_image_mask(obj.seg->height, obj.seg->width, CV_8UC1, obj.seg->data);
                    cv::cvtColor(seg_image_mask, seg_image, cv::COLOR_GRAY2BGR);
                    seg_image = seg_image.mul(cv::Scalar(b, g, r));
                    cv::Rect img_mask_rect = cv::Rect((obj.left + padw) * scale_img_mask_xy, (obj.top + padh) * scale_img_mask_xy, obj.seg->width, obj.seg->height);
                    cv::add(img_mask(img_mask_rect), seg_image, img_mask(img_mask_rect));
                }
                cv::resize(img_mask, img_mask, cv::Size(), 1 / scale_img_mask_xy, 1 / scale_img_mask_xy);
                if (mkdirs(save_dir))
                {
                    cv::Mat img_seg_mask = img_mask(cv::Rect(padw, padh, image.cols, image.rows));
                    std::string save_path = path_join("%s/Infer_%d.jpg", save_dir.c_str(), ib);
                    std::string save_seg_path = path_join("%s/Infer_%d_seg.jpg", save_dir.c_str(), ib);
                    cv::imwrite(save_seg_path, img_seg_mask);
                    cv::imwrite(save_path, 0.8 * image + 0.2 * img_seg_mask);
                }
            }
        }

        void draw_batch_pose(std::vector<cv::Mat> &images, BatchPoseBoxArray &batched_result, const std::string &save_dir,
                            const std::vector<std::string> &classlabels, const float pose_thr)
        {
            for (int ib = 0; ib < (int)batched_result.size(); ++ib)
            {
                auto &objs = batched_result[ib];
                auto &image = images[ib];
                for (auto &obj : objs)
                {
                    uint8_t b, g, r;
                    tie(b, g, r) = random_color(obj.class_label);
                    cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                                  cv::Scalar(b, g, r), 2);
                    auto name = classlabels[obj.class_label];
                    auto caption = cv::format("%s %.2f", name.c_str(), obj.confidence);
                    int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                    cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                                  cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
                    cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2,
                                16);

                    // draw point
                    for (int i = 0; i < obj.pose->pose_data.size(); i++)
                    {
                        float pose_x = obj.pose->pose_data[i][0];
                        float pose_y = obj.pose->pose_data[i][1];
                        float pose_score = obj.pose->pose_data[i][2];
                        if (pose_score >= pose_thr)
                            cv::circle(image, cv::Point(pose_x, pose_y), 4, cv::Scalar(b, g, r), -1, 16);
                    }

                    // draw line
                    for (auto &pair : obj.pose->skeleton)
                    {
                        if (obj.pose->pose_data[pair[0]][0] > 0. && obj.pose->pose_data[pair[0]][1] > 0. && obj.pose->pose_data[pair[0]][2] >= pose_thr &&
                            obj.pose->pose_data[pair[1]][0] > 0. && obj.pose->pose_data[pair[1]][0] > 0. && obj.pose->pose_data[pair[1]][2] >= pose_thr)
                            cv::line(image, cv::Point(obj.pose->pose_data[pair[0]][0], obj.pose->pose_data[pair[0]][1]),
                                     cv::Point(obj.pose->pose_data[pair[1]][0], obj.pose->pose_data[pair[1]][1]),
                                     cv::Scalar(r, g, b), 2);
                    }
                }
                if (mkdirs(save_dir))
                {
                    std::string save_path = path_join("%s/Infer_%d.jpg", save_dir.c_str(), ib);
                    cv::imwrite(save_path, image);
                }
            }
        }

        camParams::camParams(const YAML::Node &config, int n, std::vector<std::string> &cams_name) :
                            N_img(n)
        {
            if((size_t)n != cams_name.size())
            {
                std::cerr << "Error! Need " << n << " camera param, bug given " << cams_name.size() << " camera names!" << std::endl;
            }
            ego2global_rot = fromYamlQuater(config["ego2global_rotation"]);
            ego2global_trans = fromYamlTrans(config["ego2global_translation"]);

            lidar2ego_rot = fromYamlQuater(config["lidar2ego_rotation"]);
            lidar2ego_trans = fromYamlTrans(config["lidar2ego_translation"]);

            timestamp = config["timestamp"].as<unsigned long long>();
            scene_token = config["scene_token"].as<std::string>();

            imgs_file.clear();

            cams_intrin.clear();
            cams2ego_rot.clear();
            cams2ego_trans.clear();
            
            for(std::string name : cams_name)
            {
                // imgs_file.push_back("." + config["cams"][name]["data_path"].as<std::string>());
                imgs_file.push_back("/home/uisee/dl_model_infer/application/bevdet4d_app/" + config["cams"][name]["data_path"].as<std::string>());

                //
                cams_intrin.push_back(fromYamlMatrix3f(config["cams"][name]["cam_intrinsic"]));
                cams2ego_rot.push_back(fromYamlQuater(config["cams"][name]["sensor2ego_rotation"]));
                cams2ego_trans.push_back(fromYamlTrans(config["cams"][name]["sensor2ego_translation"]));
                //
            }
        }

        DataLoader::DataLoader(int _n_img, 
                            int _h, 
                            int _w,
                            const std::string &_data_infos_path,
                            const std::vector<std::string> &_cams_name,
                            bool _sep):
                            n_img(_n_img),
                            cams_intrin(_n_img), 
                            cams2ego_rot(_n_img),
                            cams2ego_trans(_n_img),
#ifdef __HAVE_NVJPEG__
                            nvdecoder(_n_img, DECODE_BGR),  
#endif
                            img_h(_h),
                            img_w(_w),
                            cams_name(_cams_name),
                            data_infos_path(_data_infos_path),
                            separate(_sep)
        {
            YAML::Node temp_seq = YAML::LoadFile(data_infos_path + "/time_sequence.yaml");
            printf("Successful load config : %s!\n", (data_infos_path + "/time_sequence.yaml").c_str());
            time_sequence = temp_seq["time_sequence"].as<std::vector<int>>();
            sample_num = time_sequence.size();

            cams_param.resize(sample_num);

            if(separate == false)
            {
                YAML::Node infos = YAML::LoadFile(data_infos_path + "/samples_info/samples_info.yaml");

                for(size_t i = 0; i < cams_name.size(); i++)
                {
                    cams_intrin[i] = fromYamlMatrix3f(infos[0]["cams"][cams_name[i]]["cam_intrinsic"]);
                    cams2ego_rot[i] = fromYamlQuater(infos[0]["cams"][cams_name[i]]
                                                                    ["sensor2ego_rotation"]);
                    cams2ego_trans[i] = fromYamlTrans(infos[0]["cams"][cams_name[i]]
                                                                    ["sensor2ego_translation"]);
                }
                lidar2ego_rot = fromYamlQuater(infos[0]["lidar2ego_rotation"]);
                lidar2ego_trans = fromYamlTrans(infos[0]["lidar2ego_translation"]);

                for(int i = 0; i < sample_num; i++)
                {
                    cams_param[i] = camParams(infos[i], n_img, cams_name);
                }
            }
            else
            {
                YAML::Node config0 = YAML::LoadFile(data_infos_path + "/samples_info/sample0000.yaml");

                for(size_t i = 0; i < cams_name.size(); i++)
                {
                    cams_intrin[i] = fromYamlMatrix3f(config0["cams"][cams_name[i]]["cam_intrinsic"]);
                    cams2ego_rot[i] = fromYamlQuater(config0["cams"][cams_name[i]]
                                                                    ["sensor2ego_rotation"]);
                    cams2ego_trans[i] = fromYamlTrans(config0["cams"][cams_name[i]]
                                                                    ["sensor2ego_translation"]);
                }
                lidar2ego_rot = fromYamlQuater(config0["lidar2ego_rotation"]);
                lidar2ego_trans = fromYamlTrans(config0["lidar2ego_translation"]);
            }

            CHECK(cudaMalloc((void**)&imgs_dev, n_img * img_h * img_w * 3 * sizeof(uchar)));
        }


        const camsData& DataLoader::data(int idx, bool time_order)
        {
            if(time_order)
            {
                idx = time_sequence[idx];
            }
            printf("------time_sequence idx : %d ---------\n", idx);
            if(separate == false)
            {
                cams_data.param = cams_param[idx];
            }
            else
            {
                char str_idx[50];
                sprintf(str_idx, "/samples_info/sample%04d.yaml", idx);
                std::cout << "str_idx: " << str_idx << std::endl;
                YAML::Node config_idx = YAML::LoadFile(data_infos_path + str_idx);
                cams_data.param = camParams(config_idx, n_img, cams_name);
            }
            imgs_data.clear();
            if(read_sample(cams_data.param.imgs_file, imgs_data))
            {
                exit(1);
            }
#ifdef __HAVE_NVJPEG__
            nvdecoder.decode(imgs_data, imgs_dev);
#else
            decode_cpu(imgs_data, imgs_dev, img_w, img_h);
            printf("decode on cpu!\n");
#endif
            cams_data.imgs_dev = imgs_dev;
            return cams_data;
        }

        DataLoader::~DataLoader()
        {
            CHECK(cudaFree(imgs_dev));
        }

        int read_image(std::string &image_names, std::vector<char> &raw_data)
        {
            std::ifstream input(image_names.c_str(), std::ios::in | std::ios::binary | std::ios::ate);

            if (!(input.is_open()))
            {
                std::cerr << "Cannot open image: " << image_names << std::endl;
                return EXIT_FAILURE;
            }

            std::streamsize file_size = input.tellg();
            input.seekg(0, std::ios::beg);
            if (raw_data.size() < (size_t)file_size)
            {
                raw_data.resize(file_size);
            }
            if (!input.read(raw_data.data(), file_size))
            {
                std::cerr << "Cannot read from file: " << image_names << std::endl;
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }

        int read_sample(std::vector<std::string> &imgs_file, std::vector<std::vector<char>> &imgs_data)
        {
            imgs_data.resize(imgs_file.size());

            for(size_t i = 0; i < imgs_data.size(); i++)
            {
                if(read_image(imgs_file[i], imgs_data[i]))
                {
                    return EXIT_FAILURE;
                }
            }
            return EXIT_SUCCESS;
        }

        Eigen::Translation3f fromYamlTrans(YAML::Node x)
        {
            std::vector<float> trans = x.as<std::vector<float>>();
            return Eigen::Translation3f(trans[0], trans[1], trans[2]);
        }

        Eigen::Quaternion<float> fromYamlQuater(YAML::Node x)
        {
            std::vector<float> quater = x.as<std::vector<float>>();
            return Eigen::Quaternion<float>(quater[0], quater[1], quater[2], quater[3]);
        }

        Eigen::Matrix3f fromYamlMatrix3f(YAML::Node x)
        {
            std::vector<std::vector<float>> m = x.as<std::vector<std::vector<float>>>();
            Eigen::Matrix3f mat;
            for(size_t i = 0; i < m.size(); i++)
            {
                for(size_t j = 0; j < m[0].size(); j++)
                {
                    mat(i, j) = m[i][j];
                }
            }
            return mat;
        }
    }
}