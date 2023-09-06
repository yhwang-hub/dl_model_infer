#ifndef _UTILS_HPP_
#define _UTILS_HPP_
#include <iostream>
#include <string>
#include <fstream>
#include <numeric>
#include <sstream>
#include <memory>
#include <vector>
#include <stack>
#include <unordered_map>
#include <initializer_list>

#include <unistd.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <stdarg.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <yaml-cpp/yaml.h>
#include "cuda_utils.h"
#define strtok_s strtok_r

using namespace std;
typedef unsigned char uchar;

namespace ai
{
    namespace utils
    {
        struct bevBox
        {
            float x, y, z, l, w, h, r;
            float vx = 0.0f;  // optional
            float vy = 0.0f;  // optional
            float score;
            int label;
            bool is_drop;  // for nms
            bevBox() = default;
            bevBox(float x, float y, float z, float l, float w, float h, float r,
                    float vx, float vy, float score, int label, bool is_drop)
                    : x(x), y(y), z(z), l(l), w(w), h(h), r(r),
                    vx(vx), vy(vy), score(score), label(label), is_drop(is_drop){}
        };

        // 一些常用的函数的定义
        std::string file_name(const std::string &path, bool include_suffix);
        void __log_func(const char *file, int line, const char *fmt, ...);
        std::vector<unsigned char> load_file(const std::string &file);
        std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id);
        std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v);

        // 一些文件处理函数
        bool file_exist(const string &path); // 判断文件是否存在
        bool dir_mkdir(const string &path);
        bool mkdirs(const string &path); // 如果文件夹不存在则会返回创建
        bool begin_with(const string &str, const string &with);
        bool end_with(const string &str, const string &with);
        string path_join(const char *fmt, ...);
        bool rmtree(const string &directory, bool ignore_fail = false);
        bool alphabet_equal(char a, char b, bool ignore_case);
        bool pattern_match(const char *str, const char *matcher, bool igrnoe_case = true);
        bool pattern_match_body(const char *str, const char *matcher, bool igrnoe_case);
        vector<string> find_files(
            const string &directory,
            const string &filter = "*",
            bool findDirectory = false,
            bool includeSubDirectory = false);
        void Boxes2Txt(const std::vector<bevBox> &boxes, std::string file_name, bool with_vel=false);
        void Egobox2Lidarbox(const std::vector<bevBox>& ego_boxes,
                             std::vector<bevBox> &lidar_boxes,
                             const Eigen::Quaternion<float>& lidar2ego_rot,
                             const Eigen::Translation3f &lidar2ego_trans);

        // 时间计时类，
        class Timer
        {
        public:
            Timer();
            virtual ~Timer();
            void start(void *stream = nullptr);
            float stop(const char *prefix = "Timer", int loop_iters = 1, bool print = true);

        private:
            void *start_, *stop_;
            void *stream_;
        };
    }
}
static float Sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static float Clamp_cpu(float x)
{
    float x_clamp = x;
    // torch.clamp(x, 0, 1)
    if (x_clamp > 1.0f)
    {
        x_clamp = 1.0f;
    } else if (x_clamp < 0.0f)
    {
        x_clamp = 0.0f;
    }
    // std::cout << "x: " << x << ", x_clamp: " << x_clamp << std::endl;
    return x_clamp;
}
#define INFO(...) ai::utils::__log_func(__FILE__, __LINE__, __VA_ARGS__)
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#endif // _UTILS_HPP_
