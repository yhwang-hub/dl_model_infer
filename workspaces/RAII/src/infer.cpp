#include "infer.hpp"
using namespace std;

class InferImpl : public InferInterface
{
public:
    bool load_model(const string& file)
    {
        if (file == "")
        {
            return false;
        }
        printf("load model %s start ....\n", file.c_str());
        context_ = file;
        printf("load model %s end ....\n", file.c_str());
        return true;
    }

    virtual void forward() override
    {
        printf("\033[32m model %s start infer \033[0m \n", context_.c_str());
    }

    void destroy()
    {
        printf("destroy model %s ....\n", context_.c_str());
        context_.clear();
    }
private:
    string context_;

};

shared_ptr<InferInterface> create_infer(const string& file)
{
    shared_ptr<InferImpl> instance (new InferImpl());
    if(!instance->load_model(file))
    {
        instance.reset();
    }

    return instance;
}