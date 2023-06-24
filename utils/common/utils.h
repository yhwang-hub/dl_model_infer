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
#include "cuda_utils.h"
#define strtok_s strtok_r

using namespace std;

namespace ai
{
    namespace utils
    {
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
            const string &filter = "*", bool findDirectory = false, bool includeSubDirectory = false);

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
#define INFO(...) ai::utils::__log_func(__FILE__, __LINE__, __VA_ARGS__)
#endif // _UTILS_HPP_
