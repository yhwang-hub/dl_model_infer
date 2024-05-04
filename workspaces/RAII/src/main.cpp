#include "infer.hpp"
#include <string>
using namespace std;

int main()
{
    auto infer = create_infer("resnet50");

    if(infer == nullptr)
    {
        printf("\033[31m Failed !!! \033[0m\n");
        return -1;
    }

    infer->forward();

    return 0;
}

/*
这就是我们建议的一个封装方式，这是我们通过 RAII + 接口模式封装的效果，通过这个效果我们来总结下几个原则：

1. 头文件，尽量只包含需要的部分

2. 外界不需要的，尽量不让它看到，保持定义的简洁

3. 不要在头文件中写 using namespace 这种，但是可以在 cpp 中写 using namespace ，对于命名空间，应当尽量少的展开

4. 不要使用构造函数去初始化，而是使用返回值是布尔值的init方法，去初始化。
*/