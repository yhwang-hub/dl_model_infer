#ifndef _GRID_SAMPLER_H
#define _GRID_SAMPLER_H

#pragma once

#include <cuda_runtime.h>

#include <cuda_fp16.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "../../common/cv_cpp_utils.h"
#include "../../common/cuda_utils.h"

struct TensorDesc
{
  int shape[MAXTENSORDIMS];
  int stride[MAXTENSORDIMS];
  int dim;
};


inline int GET_BLOCKS(const int N)
{
  int optimal_block_num = DIVUP(N, NUM_THREADS);
  int max_block_num = 4096;
  return std::min(optimal_block_num, max_block_num);
}


enum class GridSamplerInterpolation { Bilinear, Nearest };
enum class GridSamplerPadding { Zeros, Border, Reflection };

template <typename T>
void grid_sample(T *output, const T *input, const T *grid, int *output_dims, int *input_dims,
                 int *grid_dims, int nb_dims, GridSamplerInterpolation interp,
                 GridSamplerPadding padding, bool align_corners, cudaStream_t stream);

void compute_sample_grid_cuda(float* grid_dev, const float* transform,
                          int bev_w, int bev_h, cudaStream_t stream);

#endif