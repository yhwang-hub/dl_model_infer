#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

#define INTER_RESIZE_COFE_SCALE (1 << 11)

struct unchar3
{
    unsigned char r, g, b;
};

void write_image_to_file(const void* data, int width, int height, const std::string& file_path)
{
    std::ofstream file(file_path, std::ios::binary | std::ios::out);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file for writing: " << file_path << std::endl;
        return;
    }

    int header[16] = {0};
    header[0] = 0xFFCC1122;
    header[1] = 0; // 0 unchar, 1 float
    header[2] = 3;
    header[3] = height;
    header[4] = width;
    header[5] = 3;
    file.write((char*)header, sizeof(header));
    file.write((char*)data, width * height * 3);
    file.close();
}

int limit(int value, int low, int high)
{
    return value < low ? low : (value > high ? high : value);
}

void bilinear_interpolation(unchar3* image, unchar3* interp_output, const int height, const int width)
{
    int tox{32}, toy{176};
    double sx = 1 / 0.48;
    double sy = 1 / 0.48;
    int output_width{704}, output_height{256};
    unchar3 rgb[4];

    for (int y = 0; y < output_height; y++)
    {
        for (int x = 0; x < output_width; x++)
        {
            float src_x = (x + tox + 0.5f) * sx - 0.5f;
            float src_y = (y + toy + 0.5f) * sy - 0.5f;

            int y_low    = floorf(src_y);
            int x_low    = floorf(src_x);
            int y_high = limit(y_low + 1, 0, height - 1);
            int x_high = limit(x_low + 1, 0, width  - 1);
            y_low = limit(y_low, 0, height - 1);
            x_low = limit(x_low, 0, width  - 1);

            int ly = rint((src_y - y_low) * INTER_RESIZE_COFE_SCALE);
            int lx = rint((src_x - x_low) * INTER_RESIZE_COFE_SCALE);
            int hy = INTER_RESIZE_COFE_SCALE - ly;
            int hx = INTER_RESIZE_COFE_SCALE - lx;

            rgb[0] = image[y_low  * width + x_low];    // 左上角
            rgb[1] = image[y_low  * width + x_high];   // 右上角
            rgb[2] = image[y_high * width + x_low];  // 左下角
            rgb[3] = image[y_high * width + x_high]; // 右下角

            interp_output[y * output_width + x].r =
                (((hy * ((hx * rgb[0].r + lx * rgb[1].r) >> 4)) >> 16) + ((ly * ((hx * rgb[2].r + lx * rgb[3].r) >> 4)) >> 16) + 2) >> 2;
            interp_output[y * output_width + x].g =
                (((hy * ((hx * rgb[0].g + lx * rgb[1].g) >> 4)) >> 16) + ((ly * ((hx * rgb[2].g + lx * rgb[3].g) >> 4)) >> 16) + 2) >> 2;
            interp_output[y * output_width + x].b =
                (((hy * ((hx * rgb[0].b + lx * rgb[1].b) >> 4)) >> 16) + ((ly * ((hx * rgb[2].b + lx * rgb[3].b) >> 4)) >> 16) + 2) >> 2;
        }
    }
}

int main()
{
    const std::string root = "../example-data";
    const char* file_names[] = {
        "0-FRONT.jpg", "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg",
        "3-BACK.jpg", "4-BACK_LEFT.jpg", "5-BACK_RIGHT.jpg"
    };
    int output_width = 704, output_height = 256;
    int new_width = 768, new_height = 432;
    unchar3* output = new unchar3[output_height * output_width];

    long total = 0;
    for (int i = 0; i < 6; i++)
    {
        char path[200];
        sprintf(path, "%s/%s", root.c_str(), ((std::string)file_names[i]).c_str());

        // 加载图像数据
        int width, height, channels;
        unsigned char* image_data = stbi_load(path, &width, &height, &channels, 0);
        if (image_data == nullptr)
        {
            std::cerr << "Failed to load image" << std::endl;
            return 1;
        }

        char ori_pixels_path[200];
        std::string original_file_name = file_names[i];
        std::string new_file_name = original_file_name.substr(0, original_file_name.size() - 4);
        sprintf(ori_pixels_path, "../CPP_OpenCV_stbi_load_Result/%s_ori.tensor", new_file_name.c_str());
        write_image_to_file(image_data, width, height, ori_pixels_path);
        
        auto start = std::chrono::high_resolution_clock::now();
        bilinear_interpolation((unchar3*)image_data, output, height, width);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        total += duration.count();

        sprintf(ori_pixels_path, "../CPP_OpenCV_stbi_load_Result/%s_resized.tensor", new_file_name.c_str());
        write_image_to_file(output, output_width, output_height, ori_pixels_path);
        stbi_image_free(image_data);
    }
    // std::cout << "Execution time: " << (float)total / 1000 << "seconds" << std::endl;

    delete [] output;
    return 0;
}