#include "cv_cpp_utils.h"

namespace ai
{
    namespace cvUtil
    {
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
    }
}