#ifndef _CV_CPP_UTILS_HPP_
#define _CV_CPP_UTILS_HPP_

#include <iostream>
#include <tuple>
#include <string.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "utils.h"

namespace ai
{
    namespace cvUtil
    {
        using namespace std;
        using namespace ai::utils;

        // 统一模型的输入格式，方便后续进行输入的配置
        struct Image
        {
            const void *bgrptr = nullptr;
            int width = 0, height = 0, channels = 0;

            Image() = default;
            Image(const void *bgrptr, int width, int height, int channels) : bgrptr(bgrptr), width(width), height(height), channels(channels) {}
        };

        Image cvimg_trans_func(const cv::Mat &image);

        // 对输入进行尺度缩放的flage配置
        enum class NormType : int
        {
            None = 0,
            MeanStd = 1,  // out = (x * alpha - mean) / std
            AlphaBeta = 2 // out = x * alpha + beta
        };

        // 设置输入通道是RGB还是BGR
        enum class ChannelType : int
        {
            BGR = 0,
            RGB = 1
        };

        enum class DetectorType : int
        {
            V5 = 0,
            X = 1,
            V3 = 2,
            V7 = 3,
            V8 = 5,
            V8Seg = 6, // yolov8 instance segmentation
            V8Pose = 7,
            V7Pose = 8,
            SMOKE = 9
        };

        // 可以通过该结构体来初始化对输入的配置
        struct Norm
        {
            float mean[3];
            float std[3];
            float alpha, beta;
            NormType type = NormType::None;
            ChannelType channel_type = ChannelType::BGR;

            // out = (x * alpha - mean) / std
            static Norm mean_std(const float mean[3], const float std[3], float alpha = 1 / 255.0f, ChannelType channel_type = ChannelType::BGR);
            // out = x * alpha + beta
            static Norm alpha_beta(float alpha, float beta = 0.0f, ChannelType channel_type = ChannelType::BGR);
            // None
            static Norm None();
        };

        // 由于后面仿射变换使用cuda实现的，所以，这个结构体用来计算仿射变换的矩阵和逆矩阵
        struct AffineMatrix
        {
            float i2d[6]; // image to dst(network), 2x3 matrix
            float d2i[6]; // dst to image, 2x3 matrix
            void compute(const std::tuple<int, int> &from, const std::tuple<int, int> &to, DetectorType detectortype);
        };

        static void affine_project(float *matrix, float x, float y, float *ox, float *oy)
        {
            *ox = matrix[0] * x + matrix[1] * y + matrix[2];
            *oy = matrix[3] * x + matrix[4] * y + matrix[5];
        }

        // detect
        struct Box
        {
            float left, top, right, bottom, confidence;
            int class_label;

            Box() = default;
            Box(float left, float top, float right, float bottom, float confidence, int class_label)
                : left(left),
                  top(top),
                  right(right),
                  bottom(bottom),
                  confidence(confidence),
                  class_label(class_label) {}
        };
        typedef std::vector<Box> BoxArray;
        typedef std::vector<BoxArray> BatchBoxArray;

        struct BboxDim
        {
            float x, y, z;

            BboxDim() = default;
            BboxDim(float x_, float y_, float z_)
                    : x(x_),
                      y(y_),
                      z(z_) {}
        };

        struct CubeBox
        {
            int class_id;
            float score;
            cv::Point2f cube_point[8];

            CubeBox() = default;
            virtual ~CubeBox() = default;
        };

        typedef std::vector<CubeBox> CubeArray;
        typedef std::vector<CubeArray> BatchCubeArray;

        struct InstanceSegmentMap
        {
            int width = 0, height = 0;     // width % 8 == 0
            unsigned char *data = nullptr; // is width * height memory

            InstanceSegmentMap(int width, int height);
            virtual ~InstanceSegmentMap();
        };

        struct SegBox
        {
            float left, top, right, bottom, confidence;
            int class_label;
            std::shared_ptr<InstanceSegmentMap> seg; // valid only in segment task

            SegBox() = default;
            SegBox(float left, float top, float right, float bottom, float confidence, int class_label)
                : left(left),
                  top(top),
                  right(right),
                  bottom(bottom),
                  confidence(confidence),
                  class_label(class_label) {}
        };

        typedef std::vector<SegBox> SegBoxArray;
        typedef std::vector<SegBoxArray> BatchSegBoxArray;

        struct InstancePose
        {
            std::vector<std::vector<float>> pose_data; // 存储骨骼点
            // 存储骨骼连接顺序
            std::vector<std::vector<int>> skeleton{{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8}, {7, 9},\
                                                   {8, 10}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}};
            InstancePose() = default;
            virtual ~InstancePose() = default;
        };

        struct PoseBox
        {
            float left, top, right, bottom, confidence;
            int class_label;
            std::shared_ptr<InstancePose> pose; // valid only in segment task

            PoseBox() = default;
            PoseBox(float left, float top, float right, float bottom, float confidence, int class_label)
                : left(left),
                  top(top),
                  right(right),
                  bottom(bottom),
                  confidence(confidence),
                  class_label(class_label) {}
        };

        typedef std::vector<PoseBox> PoseBoxArray;
        typedef std::vector<PoseBoxArray> BatchPoseBoxArray;

        // draw image
        void draw_one_image_rectangle(cv::Mat &image, BoxArray &result, const std::string &save_dir, const std::vector<std::string> &classlabels);
        void draw_batch_rectangle(std::vector<cv::Mat> &images, BatchBoxArray &batched_result, const std::string &save_dir, const std::vector<std::string> &classlabels);

        // draw seg image
        void draw_batch_segment(std::vector<cv::Mat> &images, BatchSegBoxArray &batched_result, const std::string &save_dir,
                                const std::vector<std::string> &classlabels, int img_mask_wh = 160, int network_input_wh = 640);

        // draw pose img
        void draw_batch_pose(std::vector<cv::Mat> &images, BatchPoseBoxArray &batched_result, const std::string &save_dir,
                            const std::vector<std::string> &classlabels, const float pose_thr = 0.25f);
    }
}

#endif // _CV_CPP_UTILS_HPP_