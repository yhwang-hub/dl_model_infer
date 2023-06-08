#ifndef _MEMORY_HPP_
#define _MEMORY_HPP_
#include "utils.h"

namespace ai
{
    namespace memory
    {
        /*为了方便gpu & cpu内存的申请、复用、释放，将其功能单独用一个BaseMemory类来实现*/
        class BaseMemory
        {
        public:
            BaseMemory() = default; // c++11 的较高级用法，构造函数会默认构建
            BaseMemory(void *cpu, size_t cpu_bytes,
                       void *gpu, size_t gpu_bytes);
            virtual ~BaseMemory();
            virtual void *gpu_realloc(size_t bytes); // 申请gpu内存
            virtual void *cpu_realloc(size_t bytes); // 申请cpu内存
            void release_gpu();                      // 释放gpu内存，如果要复用，其内存不会free掉
            void release_cpu();                      // 释放cpu内存，如果要复用，其内存不会free掉
            void release();
            inline bool owner_gpu() const { return owner_gpu_; }
            inline bool owner_cpu() const { return owner_cpu_; }
            inline size_t cpu_bytes() const { return cpu_bytes_; }
            inline size_t gpu_bytes() const { return gpu_bytes_; }
            virtual inline void *get_gpu() const { return gpu_; }
            virtual inline void *get_cpu() const { return cpu_; }
            void reference(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes);

        protected:
            // cpu内存操作需要用到的一些属性
            void *cpu_ = nullptr;     // cpu内存的地址指针
            size_t cpu_bytes_ = 0;    // cpu需要申请的内存大小，一般是bytes数
            size_t cpu_capacity_ = 0; // cpu内存需要申请的最大bytes数，类似vector的capacity
            bool owner_cpu_ = true;   // cpu内存是否free的标识符，一般用这个参数来控制cpu内存的复用

            // gpu内存操作需要用到的一些属性,各属性功能同cpu
            void *gpu_ = nullptr;
            size_t gpu_bytes_ = 0, gpu_capacity_ = 0;
            bool owner_gpu_ = true;
        };

        /* 内存申请的模版类，用于对各种类型的cpu/gpu内存进行申请*/
        template <typename _DT>
        class Memory : public BaseMemory
        {
        public:
            Memory() = default;
            Memory(const Memory &other) = delete;
            Memory &operator=(const Memory &other) = delete;
            virtual _DT *gpu(size_t size) { return (_DT *)BaseMemory::gpu_realloc(size * sizeof(_DT)); } // 申请template<_TD>类型的gpu内存
            virtual _DT *cpu(size_t size) { return (_DT *)BaseMemory::cpu_realloc(size * sizeof(_DT)); } // 申请template<_TD>类型的cpu内存

            inline size_t cpu_size() const { return cpu_bytes_ / sizeof(_DT); }
            inline size_t gpu_size() const { return gpu_bytes_ / sizeof(_DT); }

            virtual inline _DT *gpu() const { return (_DT *)gpu_; } // 用来获取gpu内存的指针地址
            virtual inline _DT *cpu() const { return (_DT *)cpu_; } // 用来获取cpu内存的指针地址
        };
    }
}
#endif // _MEMORY_HPP_