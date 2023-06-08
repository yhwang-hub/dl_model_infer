#include "utils.h"

namespace ai
{
    namespace utils
    {
        // 一些常用的函数的实现
        bool file_exist(const string &path)
        {
            return access(path.c_str(), R_OK) == 0;
        }

        bool dir_mkdir(const string &path)
        {
            return mkdir(path.c_str(), 0755) == 0;
        }

        bool mkdirs(const string &path)
        {

            if (path.empty())
                return false;
            if (file_exist(path))
                return true;

            string _path = path;
            char *dir_ptr = (char *)_path.c_str();
            char *iter_ptr = dir_ptr;

            bool keep_going = *iter_ptr not_eq 0;
            while (keep_going)
            {

                if (*iter_ptr == 0)
                    keep_going = false;

                if ((*iter_ptr == '/' and iter_ptr not_eq dir_ptr) or *iter_ptr == 0)
                {
                    char old = *iter_ptr;
                    *iter_ptr = 0;
                    if (!file_exist(dir_ptr))
                    {
                        if (!dir_mkdir(dir_ptr))
                        {
                            if (!file_exist(dir_ptr))
                            {
                                INFO("mkdirs %s return false.", dir_ptr);
                                return false;
                            }
                        }
                    }
                    *iter_ptr = old;
                }
                iter_ptr++;
            }
            return true;
        }

        bool begin_with(const string &str, const string &with)
        {

            if (str.length() < with.length())
                return false;
            return strncmp(str.c_str(), with.c_str(), with.length()) == 0;
        }

        bool end_with(const string &str, const string &with)
        {

            if (str.length() < with.length())
                return false;

            return strncmp(str.c_str() + str.length() - with.length(), with.c_str(), with.length()) == 0;
        }

        string path_join(const char *fmt, ...)
        {
            va_list vl;
            va_start(vl, fmt);
            char buffer[2048];
            vsnprintf(buffer, sizeof(buffer), fmt, vl);
            return buffer;
        }

        bool rmtree(const string &directory, bool ignore_fail)
        {

            if (directory.empty())
                return false;
            auto files = find_files(directory, "*", false);
            auto dirs = find_files(directory, "*", true);

            bool success = true;
            for (int i = 0; i < files.size(); ++i)
            {
                if (::remove(files[i].c_str()) != 0)
                {
                    success = false;

                    if (!ignore_fail)
                    {
                        return false;
                    }
                }
            }

            dirs.insert(dirs.begin(), directory);
            for (int i = (int)dirs.size() - 1; i >= 0; --i)
            {
                if (::rmdir(dirs[i].c_str()) != 0)
                {
                    success = false;
                    if (!ignore_fail)
                        return false;
                }
            }
            return success;
        }

        vector<string> find_files(const string &directory, const string &filter, bool findDirectory, bool includeSubDirectory)
        {
            string realpath = directory;
            if (realpath.empty())
                realpath = "./";

            char backchar = realpath.back();
            if (backchar not_eq '\\' and backchar not_eq '/')
                realpath += "/";

            struct dirent *fileinfo;
            DIR *handle;
            stack<string> ps;
            vector<string> out;
            ps.push(realpath);

            while (!ps.empty())
            {
                string search_path = ps.top();
                ps.pop();

                handle = opendir(search_path.c_str());
                if (handle not_eq 0)
                {
                    while (fileinfo = readdir(handle))
                    {
                        struct stat file_stat;
                        if (strcmp(fileinfo->d_name, ".") == 0 or strcmp(fileinfo->d_name, "..") == 0)
                            continue;

                        if (lstat((search_path + fileinfo->d_name).c_str(), &file_stat) < 0)
                            continue;

                        if (!findDirectory and !S_ISDIR(file_stat.st_mode) or
                            findDirectory and S_ISDIR(file_stat.st_mode))
                        {
                            if (pattern_match(fileinfo->d_name, filter.c_str()))
                                out.push_back(search_path + fileinfo->d_name);
                        }

                        if (includeSubDirectory and S_ISDIR(file_stat.st_mode))
                            ps.push(search_path + fileinfo->d_name + "/");
                    }
                    closedir(handle);
                }
            }
            return out;
        }

        bool pattern_match(const char *str, const char *matcher, bool igrnoe_case)
        {
            //   abcdefg.pnga          *.png      > false
            //   abcdefg.png           *.png      > true
            //   abcdefg.png          a?cdefg.png > true

            if (!matcher or !*matcher or !str or !*str)
                return false;

            char filter[500];
            strcpy(filter, matcher);

            vector<const char *> arr;
            char *ptr_str = filter;
            char *ptr_prev_str = ptr_str;
            while (*ptr_str)
            {
                if (*ptr_str == ';')
                {
                    *ptr_str = 0;
                    arr.push_back(ptr_prev_str);
                    ptr_prev_str = ptr_str + 1;
                }
                ptr_str++;
            }

            if (*ptr_prev_str)
                arr.push_back(ptr_prev_str);

            for (int i = 0; i < arr.size(); ++i)
            {
                if (pattern_match_body(str, arr[i], igrnoe_case))
                    return true;
            }
            return false;
        }

        bool pattern_match_body(const char *str, const char *matcher, bool igrnoe_case)
        {
            //   abcdefg.pnga          *.png      > false
            //   abcdefg.png           *.png      > true
            //   abcdefg.png          a?cdefg.png > true

            if (!matcher or !*matcher or !str or !*str)
                return false;

            const char *ptr_matcher = matcher;
            while (*str)
            {
                if (*ptr_matcher == '?')
                {
                    ptr_matcher++;
                }
                else if (*ptr_matcher == '*')
                {
                    if (*(ptr_matcher + 1))
                    {
                        if (pattern_match_body(str, ptr_matcher + 1, igrnoe_case))
                            return true;
                    }
                    else
                    {
                        return true;
                    }
                }
                else if (!alphabet_equal(*ptr_matcher, *str, igrnoe_case))
                {
                    return false;
                }
                else
                {
                    if (*ptr_matcher)
                        ptr_matcher++;
                    else
                        return false;
                }
                str++;
            }

            while (*ptr_matcher)
            {
                if (*ptr_matcher not_eq '*')
                    return false;
                ptr_matcher++;
            }
            return true;
        }

        bool alphabet_equal(char a, char b, bool ignore_case)
        {
            if (ignore_case)
            {
                a = a > 'a' and a < 'z' ? a - 'a' + 'A' : a;
                b = b > 'a' and b < 'z' ? b - 'a' + 'A' : b;
            }
            return a == b;
        }

        std::string file_name(const std::string &path, bool include_suffix)
        {
            if (path.empty())
                return "";

            int p = path.rfind('/');
            int e = path.rfind('\\');
            p = std::max(p, e);
            p += 1;

            // include suffix
            if (include_suffix)
                return path.substr(p);

            int u = path.rfind('.');
            if (u == -1)
                return path.substr(p);

            if (u <= p)
                u = path.size();
            return path.substr(p, u - p);
        }

        void __log_func(const char *file, int line, const char *fmt, ...)
        {
            va_list vl;
            va_start(vl, fmt);
            char buffer[2048];
            std::string filename = file_name(file, true);
            int n = snprintf(buffer, sizeof(buffer), "[%s:%d]: ", filename.c_str(), line);
            vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);
            fprintf(stdout, "%s\n", buffer);
        }

        std::vector<unsigned char> load_file(const std::string &file)
        {
            std::ifstream in(file, std::ios::in | std::ios::binary);
            if (!in.is_open())
            {
                return {};
            }
            in.seekg(0, std::ios::end);
            size_t length = in.tellg();

            std::vector<uint8_t> data;
            if (length > 0)
            {
                in.seekg(0, std::ios::beg);
                data.resize(length);
                in.read((char *)&data[0], length);
            }
            in.close();
            return data;
        }

        std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id)
        {
            float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
            float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
            return hsv2bgr(h_plane, s_plane, 1);
        }

        std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v)
        {
            const int h_i = static_cast<int>(h * 6);
            const float f = h * 6 - h_i;
            const float p = v * (1 - s);
            const float q = v * (1 - f * s);
            const float t = v * (1 - (1 - f) * s);
            float r, g, b;
            switch (h_i)
            {
            case 0:
                r = v, g = t, b = p;
                break;
            case 1:
                r = q, g = v, b = p;
                break;
            case 2:
                r = p, g = v, b = t;
                break;
            case 3:
                r = p, g = q, b = v;
                break;
            case 4:
                r = t, g = p, b = v;
                break;
            case 5:
                r = v, g = p, b = q;
                break;
            default:
                r = 1, g = 1, b = 1;
                break;
            }
            return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255),
                              static_cast<uint8_t>(r * 255));
        }

        // 时间类的实现
        Timer::Timer()
        {
            checkRuntime(cudaEventCreate((cudaEvent_t *)&start_));
            checkRuntime(cudaEventCreate((cudaEvent_t *)&stop_));
        }

        Timer::~Timer()
        {
            checkRuntime(cudaEventDestroy((cudaEvent_t)start_));
            checkRuntime(cudaEventDestroy((cudaEvent_t)stop_));
        }

        void Timer::start(void *stream)
        {
            stream_ = stream;
            checkRuntime(cudaEventRecord((cudaEvent_t)start_, (cudaStream_t)stream_));
        }

        float Timer::stop(const char *prefix, int loop_iters, bool print)
        {
            checkRuntime(cudaEventRecord((cudaEvent_t)stop_, (cudaStream_t)stream_));
            checkRuntime(cudaEventSynchronize((cudaEvent_t)stop_));

            float latency = 0;
            checkRuntime(cudaEventElapsedTime(&latency, (cudaEvent_t)start_, (cudaEvent_t)stop_));

            if (print)
            {
                printf("[%s]: %.5f ms\n", prefix, latency / loop_iters);
            }
            return latency;
        }
    }
}