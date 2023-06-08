// 一般而言直接在hpp中进行函数的定义，最好用static修饰，否则会报错：multiple definition of x
#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP
#include <string>
#include <stdarg.h>
#include <cmath>
#include <cuda_runtime.h>
#include "utils.h"

#define GPU_BLOCK_THREADS 512 // 512=32*16
#define INTER_RESIZE_COEF_BITS 11
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS) // 2048
#define CAST_BITS (INTER_RESIZE_COEF_BITS << 1)

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

#define checkCudaKernel(...)                                                                        \
    __VA_ARGS__;                                                                                    \
    do                                                                                              \
    {                                                                                               \
        cudaError_t cudaStatus = cudaPeekAtLastError();                                             \
        if (cudaStatus != cudaSuccess)                                                              \
        {                                                                                           \
            std::cerr << "kernel function failed: " << cudaGetErrorString(cudaStatus) << std::endl; \
        }                                                                                           \
    } while (0);

#define checkRuntime(call)                                                                       \
    do                                                                                           \
    {                                                                                            \
        auto ___call__ret_code__ = (call);                                                       \
        if (___call__ret_code__ != cudaSuccess)                                                  \
        {                                                                                        \
            INFO("CUDA Runtime error💥 %s # %s, code = %s [ %d ]", #call,                         \
                 cudaGetErrorString(___call__ret_code__), cudaGetErrorName(___call__ret_code__), \
                 ___call__ret_code__);                                                           \
            abort();                                                                             \
        }                                                                                        \
    } while (0)

#define checkKernel(...)                     \
    do                                       \
    {                                        \
        {                                    \
            (__VA_ARGS__);                   \
        }                                    \
        checkRuntime(cudaPeekAtLastError()); \
    } while (0)

#define Assert(op)                       \
    do                                   \
    {                                    \
        bool cond = !(!(op));            \
        if (!cond)                       \
        {                                \
            INFO("Assert failed, " #op); \
            abort();                     \
        }                                \
    } while (0)

#define Assertf(op, ...)                                   \
    do                                                     \
    {                                                      \
        bool cond = !(!(op));                              \
        if (!cond)                                         \
        {                                                  \
            INFO("Assert failed, " #op " : " __VA_ARGS__); \
            abort();                                       \
        }                                                  \
    } while (0)

template <typename _T>
static __inline__ __device__ _T clipf(_T value, _T low, _T high)
{
    return value < low ? low : (value > high ? high : value);
}

static __inline__ __device__ int resize_cast(int value)
{
    return (value + (1 << (CAST_BITS - 1))) >> CAST_BITS;
}

template <typename _T>
static __inline__ __device__ _T limit(_T value, _T low, _T high)
{
    return value < low ? low : (value > high ? high : value);
}

static __host__ inline float desigmoid(float y)
{
    return -log(1.0f / y - 1.0f);
}

static __device__ inline float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

static __host__ inline int upbound(int n, int align = 32)
{
    return (n + align - 1) / align * align;
}

namespace CUDATools
{
    static dim3 grid_dims(int numJobs)
    {
        int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
        return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
    }

    static dim3 block_dims(int numJobs)
    {
        return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    }
}

#endif // CUDA_UTILS_HPP