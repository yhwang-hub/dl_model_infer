#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include "include/types.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

#define checkRuntime(op) __check_cuda_runtime(op, #op, __FILE__, __LINE__)
bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
    if (code != cudaSuccess) {
        const char* name = cudaGetErrorName(code);
        const char* message = cudaGetErrorString(code);
        printf("CUDA Runtime error %s # %s, code = %s in file %s:%d\n", name, message, op, file, line);
        return false;
    }
    return true;
}

void run_bilinear_interpolation(cudaStream_t stream, nv::unchar3* image_data, int width, int height,
                            int output_width, int output_height, nv::unchar3* d_output);
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
    header[1] = 0;
    header[2] = 3;
    header[3] = height;
    header[4] = width;
    header[5] = 3;
    file.write((char*)header, sizeof(header));
    file.write((char*)data, (width * height * 3));
    file.close();
    printf("saved %s\n", file_path.c_str());
}

void convert_image(nv::unchar3* image_data_, unsigned char* image_data, int width, int height)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            image_data_[y * width + x].r = image_data[(y * width + x) * 3];
            image_data_[y * width + x].g = image_data[(y * width + x) * 3 + 1];
            image_data_[y * width + x].b = image_data[(y * width + x) * 3 + 2];
        }
    }
}

int main()
{
    const std::string root = "../example-data";
    const char* file_names[] =
    {
        "0-FRONT.jpg", "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg",
        "3-BACK.jpg", "4-BACK_LEFT.jpg", "5-BACK_RIGHT.jpg"
    };

    int output_width = 704, output_height = 256;
    nv::unchar3* output = new nv::unchar3[output_width * output_height];
    nv::unchar3* d_output = nullptr;
    checkRuntime(cudaMalloc(&d_output, output_width * output_height * sizeof(nv::unchar3)));
    nv::unchar3* d_image_data = nullptr;
    checkRuntime(cudaMalloc(&d_image_data, 1600 * 900 * sizeof(nv::unchar3)));
    nv::unchar3* converted_image_data = new nv::unchar3[1600 * 900];
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0;

    for (int i = 0; i < 6; ++i)
    {
        char path[200];
        sprintf(path, "%s/%s", root.c_str(), ((std::string)file_names[i]).c_str());

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
        sprintf(ori_pixels_path, "../CUDA_Result/%s_ori.tensor", new_file_name.c_str());
        write_image_to_file(image_data, width, height, ori_pixels_path);
        convert_image(converted_image_data, image_data, width, height);
        checkRuntime(cudaMemcpy(d_image_data, converted_image_data, width * height * sizeof(nv::unchar3), cudaMemcpyHostToDevice));

        cudaEventRecord(start, stream);
        run_bilinear_interpolation(stream, d_image_data, width, height, output_width, output_height, d_output);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;

        checkRuntime(cudaMemcpy(output, d_output, output_width * output_height * sizeof(nv::unchar3), cudaMemcpyDeviceToHost));
        char interp_pixels_path[200];
        sprintf(interp_pixels_path, "../CUDA_Result/%s_cuda.tensor", new_file_name.c_str());
        write_image_to_file(output, output_width , output_height, interp_pixels_path);

        stbi_image_free(image_data);
    }

    printf("6 image bilinear_interpolation total time: %f\n", total_time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] output;
    delete[] converted_image_data;
    checkRuntime(cudaFree(d_output));
    checkRuntime(cudaFree(d_image_data));
    checkRuntime(cudaStreamDestroy(stream));

    return 0;
}