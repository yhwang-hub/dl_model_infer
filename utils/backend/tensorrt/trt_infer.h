#ifndef _TRT_INFER_HPP_
#define _TRT_INFER_HPP_

// tensorrt 导入的库
#include <logger.h>
#include <parserOnnxConfig.h>
#include <NvInfer.h>
#include "NvInferPlugin.h"

#include <memory>
#include "../backend_infer.h"
#include "../../common/utils.h"
namespace trt
{
    namespace infer
    {
        using namespace nvinfer1;
        using namespace ai::backend;
        // 为了适配tensorrt7.xx和8.xx
        template <typename _T>
        static void destroy_nvidia_pointer(_T *ptr)
        {
            if (ptr)
                ptr->destroy();
        }

        /* 此类是为了初始化tensorrt推理需要用到的 rutime,engine,context */
        class __native_engine_context
        {
        public:
            __native_engine_context() = default;
            virtual ~__native_engine_context();
            bool construct(std::vector<unsigned char> &trtFile);

        private:
            void destroy();

        public:
            shared_ptr<IExecutionContext> context_;
            shared_ptr<ICudaEngine> engine_;
            shared_ptr<IRuntime> runtime_ = nullptr;
        };

        /* 此类是显示了tensorrt的一些信息配置和推理的实现，主要是对上面Infer的具体实现*/
        class InferTRT : public Infer
        {
        public:
            InferTRT() = default;
            virtual ~InferTRT() = default;

            void setup();                                                // 初始化binding_name_to_index_，用来对输入输入name和index进行绑定
            bool construct_context(std::vector<unsigned char> &trtFile); // 反序列化engine并赋值其上下文context_
            bool load(const string &engine_file);                        // 加载engine_file，里面调用了construct_context方法
            std::string format_shape(const Dims &shape);                 // 将输出shape转为str，单纯是用于打印shape，无太大意义

            virtual bool forward(const std::vector<void *> &bindings, void *stream,
                                 void *input_consum_event) override; // 执行推理操作

            virtual int index(const std::string &name) override; // 根据name寻找index
            virtual std::vector<int> get_network_dims(const std::string &name) override;
            virtual std::vector<int> get_network_dims(int ibinding) override; // 获取模型engine的输入输出的维度信息，更加常用
            virtual bool set_network_dims(const std::string &name, const std::vector<int> &dims) override;
            virtual bool set_network_dims(int ibinding, const std::vector<int> &dims) override; // 设置动态shape，很重要

            virtual bool has_dynamic_dim() override; // 判断模型输入是否是动态shape
            virtual void print() override;           // 打印当前模型的一些输入、输出维度信息等

        public:
            shared_ptr<__native_engine_context> context_ = nullptr; // 创建context指针
            unordered_map<string, int> binding_name_to_index_;      // 创建{name:index}的map对象
        };

        // 多态，加载模型并返回实例化的对象指针
        Infer *loadraw(const std::string &file);
        std::shared_ptr<Infer> load(const std::string &file);

        bool onnxToTRTModel(const std::string& modelFile, const std::string& engine_file);
    }
}

#endif // _TRT_INFER_HPP_