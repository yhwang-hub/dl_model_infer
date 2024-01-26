#ifndef _CV_CPP_UTILS_HPP_
#define _CV_CPP_UTILS_HPP_

#include <iostream>
#include <tuple>
#include <string.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "cpu_jpegdecoder.h"
#include "nvjpegdecoder.h"

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
            V5,
            X,
            V3,
            V7,
            V8,
            V8Seg, // yolov8 instance segmentation
            V8Pose,
            V8Obb,
            V7Pose,
            SMOKE,
            DETR
        };

        enum class Sampler
        {
            nearest,
            bicubic
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

        struct RotateBox
        {
            float center_x, center_y, width, height, angle, confidence;
            int class_label;

            RotateBox() = default;

            RotateBox(
                float center_x,
                float center_y,
                float width,
                float height,
                float angle,
                float confidence,
                int class_label) :
                    center_x(center_x),
                    center_y(center_y),
                    width(width),
                    height(height),
                    angle(angle),
                    confidence(confidence),
                    class_label(class_label){}
        };
        typedef std::vector<RotateBox> RotateBoxArray;
        typedef std::vector<RotateBoxArray> BatchRotateBoxArray;

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

        struct camParams
        {
            camParams() = default;
            camParams(const YAML::Node &config, int n, std::vector<std::string>& cams_name);

            int N_img;

            Eigen::Quaternion<float> ego2global_rot;
            Eigen::Translation3f ego2global_trans;

            Eigen::Quaternion<float> lidar2ego_rot;
            Eigen::Translation3f lidar2ego_trans;

            std::vector<Eigen::Matrix3f> cams_intrin;
            std::vector<Eigen::Quaternion<float>> cams2ego_rot;
            std::vector<Eigen::Translation3f> cams2ego_trans;

            std::vector<std::string> imgs_file;

            unsigned long long timestamp;
            std::string scene_token;
        };

        struct camsData
        {
            camsData() = default;
            camsData(const camParams &_param) :
                param(_param),
                imgs_dev(nullptr) {}
            camParams param;
            uchar* imgs_dev;
        };

        class DataLoader
        {
        public:
            DataLoader() = default;
            DataLoader(int _n_img, 
                    int _h, 
                    int _w,
                    const std::string &_data_infos_path,
                    const std::vector<std::string> &_cams_name,
                    bool _sep=true);

            const std::vector<Eigen::Matrix3f>& get_cams_intrin() const
            {
                return cams_intrin;
            }
            const std::vector<Eigen::Quaternion<float>>& get_cams2ego_rot() const
            {
                return cams2ego_rot;
            }
            const std::vector<Eigen::Translation3f>& get_cams2ego_trans() const
            {
                return cams2ego_trans;
            }

            const Eigen::Quaternion<float>& get_lidar2ego_rot() const
            {
                return lidar2ego_rot;
            }

            const Eigen::Translation3f& get_lidar2ego_trans() const
            {
                return lidar2ego_trans;
            }

            int size()
            {
                return sample_num;
            }

            const camsData& data(int idx, bool time_order=true);
            ~DataLoader();

        private:
            std::vector<int> time_sequence;
            std::string data_infos_path;
            int sample_num;

            std::vector<std::string> cams_name;
            int n_img;
            int img_h;
            int img_w;

            std::vector<camParams> cams_param; 
            camsData cams_data;

            std::vector<Eigen::Matrix3f> cams_intrin;
            std::vector<Eigen::Quaternion<float>> cams2ego_rot;
            std::vector<Eigen::Translation3f> cams2ego_trans;
            Eigen::Quaternion<float> lidar2ego_rot;
            Eigen::Translation3f lidar2ego_trans;

#ifdef __HAVE_NVJPEG__
            nvjpegDecoder nvdecoder;
#endif
            uchar *imgs_dev;
            std::vector<std::vector<char>> imgs_data;
            bool separate;
        };

        Eigen::Translation3f fromYamlTrans(YAML::Node x);
        Eigen::Quaternion<float> fromYamlQuater(YAML::Node x);
        Eigen::Matrix3f fromYamlMatrix3f(YAML::Node x);

        int read_image(std::string &image_names, std::vector<char> &raw_data);

        int read_sample(std::vector<std::string> &imgs_file, 
                        std::vector<std::vector<char>> &imgs_data);

        // draw image
        void draw_one_image_rectangle(cv::Mat &image, BoxArray &result, const std::string &save_dir, const std::vector<std::string> &classlabels);
        void draw_batch_rectangle(std::vector<cv::Mat> &images, BatchBoxArray &batched_result, const std::string &save_dir, const std::vector<std::string> &classlabels);

        void draw_batch_rotaterectangle(std::vector<cv::Mat> &images, BatchRotateBoxArray &batched_result, const std::string &save_dir, const std::vector<std::string> &dotalabels);

        // draw seg image
        void draw_batch_segment(std::vector<cv::Mat> &images, BatchSegBoxArray &batched_result, const std::string &save_dir,
                                const std::vector<std::string> &classlabels, int img_mask_wh = 160, int network_input_wh = 640);

        // draw pose img
        void draw_batch_pose(std::vector<cv::Mat> &images, BatchPoseBoxArray &batched_result, const std::string &save_dir,
                            const std::vector<std::string> &classlabels, const float pose_thr = 0.25f);
    }
}

#endif // _CV_CPP_UTILS_HPP_