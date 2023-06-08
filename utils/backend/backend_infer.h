#ifndef _BACKEND_HPP_
#define _BACKEND_HPP_
#include <vector>
#include <string>
namespace ai
{
    namespace backend
    {

        /* 模型推理的基类，只能通过继承，可以通过该类配置一些信息，方便以后多类继承使用。*/
        class Infer
        {
        public:
            virtual int index(const std::string &name) = 0; // 输入输出name to index
            virtual bool forward(const std::vector<void *> &bindings, void *stream = nullptr,
                                 void *input_consum_event = nullptr) = 0;           // trt执行推理操作
            virtual std::vector<int> get_network_dims(const std::string &name) = 0; // 根据输入输出名称获取模型输入输出维度
            virtual std::vector<int> get_network_dims(int ibinding) = 0;            // 根据index获取模型输入输出维度
            virtual bool set_network_dims(const std::string &name, const std::vector<int> &dims) = 0;
            virtual bool set_network_dims(int ibinding, const std::vector<int> &dims) = 0; // 设置输入的动态shape
            virtual bool has_dynamic_dim() = 0;                                            // 判断是动态shape输入还是静态shape输入
            virtual void print() = 0;                                                      // 打印输入输出维度信息
        };
    }
}
#endif // _BACKEND_HPP_