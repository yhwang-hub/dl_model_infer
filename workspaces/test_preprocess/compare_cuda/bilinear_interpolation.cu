#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "include/types.hpp"
#include <cuda_fp16.h>

#define INTER_RESIEZE_CORE_SCALE (1 << 11)

__device__ int limit(int value, int low, int high)
{
    return value < low ? low : (value > high ? high : value);
}

__global__ void bilinear_interpolation_kernel(cudaStream_t stream, nv::unchar3* image_data,
                                    int width, int height, int output_width, int output_height,
                                    int tox, int toy, double sx, double sy, nv::unchar3* output)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix >= output_width || iy >= output_height)
    {
        return;
    }
    nv::unchar3 rgb[4];

    double src_x = (ix + tox + 0.5f) * sx - 0.5f;
    double src_y = (iy + toy + 0.5f) * sy - 0.5f;

    int y_low  = floorf(src_y);
    int x_low  = floorf(src_x);
    int y_high = limit(y_low + 1, 0, height - 1);
    int x_high = limit(x_low + 1, 0, width  - 1);
    y_low = limit(y_low, 0, height - 1);
    x_low = limit(x_low, 0, width  - 1);

    int ly = rint((src_y - y_low) * INTER_RESIEZE_CORE_SCALE);
    int lx = rint((src_x - x_low) * INTER_RESIEZE_CORE_SCALE);
    int hy = INTER_RESIEZE_CORE_SCALE - ly;
    int hx = INTER_RESIEZE_CORE_SCALE - lx;

    rgb[0] = image_data[y_low * width + x_low];
    rgb[1] = image_data[y_low * width + x_high];
    rgb[2] = image_data[y_high * width + x_low];
    rgb[3] = image_data[y_high * width + x_high];

     output[iy * output_width + ix].r = 
        (((hy * ((hx * rgb[0].r + lx * rgb[1].r) >> 4)) >> 16) + ((ly * ((hx * rgb[2].r + lx * rgb[3].r) >> 4)) >> 16) + 2) >> 2;
    output[iy * output_width + ix].g = 
        (((hy * ((hx * rgb[0].g + lx * rgb[1].g) >> 4)) >> 16) + ((ly * ((hx * rgb[2].g + lx * rgb[3].g) >> 4)) >> 16) + 2) >> 2;
    output[iy * output_width + ix].b = 
        (((hy * ((hx * rgb[0].b + lx * rgb[1].b) >> 4)) >> 16) + ((ly * ((hx * rgb[2].b + lx * rgb[3].b) >> 4)) >> 16) + 2) >> 2;
}

void run_bilinear_interpolation(cudaStream_t stream, nv::unchar3* image_data,
                        int width, int height, int output_width, int output_height, nv::unchar3* output)
{
    int tox{32}, toy{176};
    double sx = 1 / 0.48;
    double sy = 1 / 0.48;

    int thread_nums = 32;
    dim3 __threads__{thread_nums, thread_nums};
    dim3 __blocks__{(output_width + thread_nums - 1) / thread_nums, (output_height + thread_nums - 1) / thread_nums};
    bilinear_interpolation_kernel<<<__blocks__, __threads__, 0, stream>>>(
        stream, image_data, width, height,
        output_width, output_height,
        tox, toy, sx, sy, output
    );
}