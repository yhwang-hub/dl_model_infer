#include "memory.h"
namespace ai
{
    namespace memory
    {
        BaseMemory::BaseMemory(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes)
        {
            reference(cpu, cpu_bytes, gpu, gpu_bytes);
        }

        void BaseMemory::reference(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes)
        {
            release();

            if (cpu == nullptr || cpu_bytes == 0)
            {
                cpu = nullptr;
                cpu_bytes = 0;
            }

            if (gpu == nullptr || gpu_bytes == 0)
            {
                gpu = nullptr;
                gpu_bytes = 0;
            }

            this->cpu_ = cpu;
            this->cpu_capacity_ = cpu_bytes;
            this->cpu_bytes_ = cpu_bytes;
            this->gpu_ = gpu;
            this->gpu_capacity_ = gpu_bytes;
            this->gpu_bytes_ = gpu_bytes;

            this->owner_cpu_ = !(cpu && cpu_bytes > 0);
            this->owner_gpu_ = !(gpu && gpu_bytes > 0);
        }

        BaseMemory::~BaseMemory() { release(); }

        void *BaseMemory::gpu_realloc(size_t bytes)
        {
            if (gpu_capacity_ < bytes)
            {
                release_gpu();

                gpu_capacity_ = bytes;
                checkRuntime(cudaMalloc(&gpu_, bytes));
                // checkRuntime(cudaMemset(gpu_, 0, size)); // 将gpu申请的bytes内存置为0，加不加都行
            }
            gpu_bytes_ = bytes;
            return gpu_;
        }

        void *BaseMemory::cpu_realloc(size_t bytes)
        {
            if (cpu_capacity_ < bytes)
            {
                release_cpu();

                cpu_capacity_ = bytes;
                checkRuntime(cudaMallocHost(&cpu_, bytes)); // 注意，这里申请的是固定页内存
                Assert(cpu_ != nullptr);
                // memset(cpu_, 0, size);
            }
            cpu_bytes_ = bytes;
            return cpu_;
        }

        void BaseMemory::release_cpu()
        {
            if (cpu_)
            {
                if (owner_cpu_)
                {
                    checkRuntime(cudaFreeHost(cpu_));
                }
                cpu_ = nullptr;
            }
            cpu_capacity_ = 0;
            cpu_bytes_ = 0;
        }

        void BaseMemory::release_gpu()
        {
            if (gpu_)
            {
                if (owner_gpu_)
                {
                    checkRuntime(cudaFree(gpu_));
                }
                gpu_ = nullptr;
            }
            gpu_capacity_ = 0;
            gpu_bytes_ = 0;
        }

        void BaseMemory::release()
        {
            release_cpu();
            release_gpu();
        }
    }
}